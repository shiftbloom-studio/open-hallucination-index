// tsproxy — tsnet-backed bolt TCP forwarder that lets the OHI Lambda
// reach a Tailnet-hosted Neo4j (or any other TCP service) without
// putting the Lambda inside a VPC or running a full tailscaled daemon.
//
// Why this exists: AWS Lambda has no /dev/net/tun, so we cannot run the
// standard tailscaled (which needs a tun device or root-mode
// user-space-networking with additional privileges we don't get).
// tsnet runs Tailscale entirely in userspace inside this process, which
// IS permitted in the Lambda sandbox. We use it to accept bolt
// connections on 127.0.0.1:7687 inside the Lambda and forward them
// over the Tailnet to the PC-hosted Neo4j at TS_UPSTREAM.
//
// The Python FastAPI app (adapters/neo4j.py) then opens NEO4J_URI=
// bolt://127.0.0.1:7687 and doesn't need to know about Tailscale at
// all — tsproxy is invisible to it.
//
// Environment variables:
//   TS_AUTHKEY   Tailscale auth key (reusable + ephemeral recommended).
//                If empty, tsproxy logs + exits(0); the Lambda app will
//                then start but /health/deep will report Neo4j down.
//   TS_HOSTNAME  Tailnet node hostname for this Lambda instance
//                (default: "ohi-lambda").
//   TS_UPSTREAM  "host:port" in the Tailnet to forward to
//                (e.g. "tens0rfl0w:7687"). Required when TS_AUTHKEY is
//                set; otherwise tsproxy is a no-op.
//   TS_LISTEN    "host:port" where tsproxy accepts local connections
//                (default: "127.0.0.1:7687").
//   TS_STATE_DIR Directory for tsnet state. Default "/tmp/tsnet" — the
//                only writable tree in a Lambda /var/task container.

package main

import (
	"context"
	"errors"
	"io"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"tailscale.com/tsnet"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.SetPrefix("[tsproxy] ")

	authkey := os.Getenv("TS_AUTHKEY")
	if authkey == "" {
		log.Println("TS_AUTHKEY is empty; skipping Tailnet bring-up. Lambda app will start but Neo4j will be unreachable.")
		// Exit 0 so the Lambda entrypoint doesn't treat the proxy as failed.
		os.Exit(0)
	}

	hostname := getenv("TS_HOSTNAME", "ohi-lambda")
	upstream := os.Getenv("TS_UPSTREAM")
	if upstream == "" {
		log.Println("TS_UPSTREAM unset; skipping Tailnet bring-up (nothing to forward to). Lambda will run in Neo4j-degraded mode.")
		os.Exit(0)
	}
	listen := getenv("TS_LISTEN", "127.0.0.1:7687")
	stateDir := getenv("TS_STATE_DIR", "/tmp/tsnet")

	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		log.Printf("mkdir state dir %q: %v — skipping Tailnet bring-up", stateDir, err)
		os.Exit(0)
	}

	srv := &tsnet.Server{
		Hostname:  hostname,
		AuthKey:   authkey,
		Ephemeral: true,
		Dir:       stateDir,
		// Silence tsnet's verbose logging; keep only our own.
		Logf: func(format string, args ...any) {},
	}

	// Bound Tailnet bring-up so a dead control plane, a stale auth key, or
	// a placeholder secret value doesn't stall Lambda init past its 10 s
	// ceiling. The sequential wait in entrypoint.sh gives us ~6 s here
	// after Python + Docker startup. Any failure is logged and treated as
	// "degraded mode" rather than a container crash — the Python app MUST
	// still boot, even when Tailscale is misconfigured.
	bringUpCtx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()
	if _, err := srv.Up(bringUpCtx); err != nil {
		log.Printf("tsnet Up failed (%v); Lambda will run in Neo4j-degraded mode.", err)
		_ = srv.Close()
		os.Exit(0)
	}
	defer srv.Close()
	log.Printf("tsnet up as %q; forwarding %s -> %s via tailnet", hostname, listen, upstream)

	ln, err := net.Listen("tcp", listen)
	if err != nil {
		log.Printf("listen on %s failed (%v); Lambda will run in Neo4j-degraded mode.", listen, err)
		os.Exit(0)
	}
	defer ln.Close()

	for {
		c, err := ln.Accept()
		if err != nil {
			// Listener was closed (e.g. during shutdown). Return cleanly.
			if errors.Is(err, net.ErrClosed) {
				return
			}
			log.Printf("accept: %v", err)
			continue
		}
		go handle(srv, c, upstream)
	}
}

func handle(srv *tsnet.Server, local net.Conn, upstream string) {
	defer local.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	remote, err := srv.Dial(ctx, "tcp", upstream)
	if err != nil {
		log.Printf("dial %s via tailnet: %v", upstream, err)
		return
	}
	defer remote.Close()

	// Full-duplex pipe. Return when either side finishes; do NOT
	// wait for both, so a half-closed TCP doesn't leak the goroutine.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = io.Copy(remote, local)
		// Signal upstream close to unblock the other direction.
		if tc, ok := remote.(*net.TCPConn); ok {
			_ = tc.CloseWrite()
		}
	}()
	go func() {
		defer wg.Done()
		_, _ = io.Copy(local, remote)
		if tc, ok := local.(*net.TCPConn); ok {
			_ = tc.CloseWrite()
		}
	}()
	wg.Wait()
}
