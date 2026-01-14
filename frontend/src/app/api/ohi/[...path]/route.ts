import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

const REQUIRED_ENVS = ["DEFAULT_API_URL", "DEFAULT_API_KEY"] as const;
type RequiredEnv = (typeof REQUIRED_ENVS)[number];

export const getRequiredEnvs = () => REQUIRED_ENVS;

function requireEnv(name: RequiredEnv): string {
  const value = process.env[name];
  if (!value) throw new Error(`Missing required env var: ${name}`);
  return value;
}

function joinUrl(base: string, path: string) {
  const b = base.endsWith("/") ? base.slice(0, -1) : base;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

function copyHeaders(incoming: Headers) {
  const headers = new Headers(incoming);

  // Avoid forwarding hop-by-hop headers.
  headers.delete("connection");
  headers.delete("host");
  headers.delete("content-length");

  return headers;
}

async function handle(
  req: Request,
  ctx: { params: Promise<{ path?: string[] }> }
) {
  const { path = [] } = await ctx.params;

  const baseUrl = requireEnv("DEFAULT_API_URL");
  const apiKey = requireEnv("DEFAULT_API_KEY");

  const incomingUrl = new URL(req.url);

  // Build forwarded path and remove local `api/ohi` prefix if present to avoid duplication
  let forwardedPath = path.join("/");
  if (forwardedPath.startsWith("api/ohi/")) {
    forwardedPath = forwardedPath.replace(/^api\/ohi\//, "");
  } else if (forwardedPath === "api/ohi") {
    forwardedPath = "";
  }

  // If the baseUrl already includes an `/api` segment and the forwardedPath also starts with `api/`,
  // strip the extra `api/` to avoid duplicating `/api` in the upstream URL.
  try {
    const parsedBase = new URL(baseUrl);
    if (
      parsedBase.pathname.endsWith("/api") &&
      forwardedPath.startsWith("api/")
    ) {
      forwardedPath = forwardedPath.replace(/^api\//, "");
    }
  } catch {
    // ignore if baseUrl isn't a full URL
  }

  const upstreamUrl = forwardedPath
    ? new URL(joinUrl(baseUrl, forwardedPath))
    : new URL(baseUrl);
  upstreamUrl.search = incomingUrl.search;

  const headers = copyHeaders(req.headers);
  headers.set("X-API-KEY", apiKey);

  try {
    const supabase = await createClient();
    const { data } = await supabase.auth.getUser();
    if (data?.user?.id) {
      headers.set("X-User-Id", data.user.id);
    }
  } catch {
    // ignore auth lookup failures for upstream requests
  }

  const method = req.method.toUpperCase();
  const hasBody = !["GET", "HEAD"].includes(method);

  const body = hasBody ? await req.arrayBuffer() : undefined;
  try {
    const upstream = await fetch(upstreamUrl.toString(), {
      method,
      headers,
      body,
    });

    return new Response(upstream.body, {
      status: upstream.status,
      headers: upstream.headers,
    });
  } catch (err) {
    console.error("Upstream fetch failed", {
      upstreamUrl: upstreamUrl?.toString?.(),
      err,
    });
    return NextResponse.json(
      { error: "Upstream fetch failed", details: String(err) },
      { status: 502 }
    );
  }
}

export const GET = handle;
export const POST = handle;
export const PUT = handle;
export const PATCH = handle;
export const DELETE = handle;
