# Public Internet Access Setup (HTTPS)

This guide explains how to expose the Open Hallucination Index API to the public internet securely with HTTPS. It is written to work with most routers, OSes, and DNS providers.

## Prerequisites

1. Router or gateway with port forwarding (any brand)
2. A domain name pointing to your public IP (or dynamic DNS)
3. API running on your local machine (Docker Compose recommended)
4. Ports 80 and 443 reachable from the internet
5. No CGNAT, or an alternative tunnel/VPS setup (see below)

## Quick Start (HTTPS with Let's Encrypt)

### Step 1: Generate an API Key

```bash
# Generate a random 64-character API key
openssl rand -hex 32
```

Add to your repository‑root `.env` file:
```bash
API_API_KEY=your_generated_api_key_here
```

If you use the Next.js proxy (`/api/ohi/*`), also set:
```bash
DEFAULT_API_KEY=your_generated_api_key_here
DEFAULT_API_URL=http://ohi-api:8080
```

### Step 2: Configure Router Port Forwarding

Forward **both** ports to your PC:

| Protocol | External Port | Internal Port |
|----------|---------------|---------------|
| TCP | 80 | 80 |
| TCP | 443 | 443 |

See [Router Port Forwarding](#router-port-forwarding) below for detailed steps and Fritz!Box notes.

### Step 3: Set Up DNS (Static or Dynamic)

Use any DNS provider or dynamic DNS service (e.g., Cloudflare, DuckDNS, No-IP, DynDNS, MyFRITZ!). Point your domain to your **public** IP address.

### Step 4: Initialize Let's Encrypt Certificate

```bash
# Make the script executable
chmod +x scripts/init-letsencrypt.sh

# Run the setup (replace with your domain and email)
./scripts/init-letsencrypt.sh yourdomain.example.com your@email.com
```

### Step 5: Start All Services

```bash
docker-compose up -d
```

Your API is now available. By default, public access goes through the Next.js proxy route:

- `https://yourdomain.example.com/api/ohi/api/v1/verify`

If you expose the backend directly (e.g., custom nginx location), use:

- `https://yourdomain.example.com/api/v1/verify` with `X-API-Key: <API_API_KEY>`

---

## Router Port Forwarding

### Access Router Admin Interface

1. Open your router admin page (often `http://192.168.0.1` or `http://192.168.1.1`)
2. Log in with your router credentials

### Configure Port Forwarding

1. Find **Port Forwarding**, **NAT**, or **Virtual Server** settings
2. Create two TCP rules that forward to your machine's local IP:

  **Port 80 (HTTP - for Let's Encrypt):**
  | Setting | Value |
  |---------|-------|
  | Name | OHI-HTTP |
  | Protocol | TCP |
  | External Port | 80 |
  | Internal IP | your PC IP |
  | Internal Port | 80 |

  **Port 443 (HTTPS):**
  | Setting | Value |
  |---------|-------|
  | Name | OHI-HTTPS |
  | Protocol | TCP |
  | External Port | 443 |
  | Internal IP | your PC IP |
  | Internal Port | 443 |

3. Save/apply the configuration

### Fritz!Box Notes (Optional)

If you use a Fritz!Box, navigate to **Internet** → **Freigaben** → **Portfreigaben** and add two TCP rules for ports 80 and 443. For DynDNS, **Internet** → **MyFRITZ!-Konto**.

---

## Architecture

```
Internet
    │
    ▼
┌─────────────┐
│   Router    │  Port 80/443 forwarded
└─────────────┘
    │
    ▼
┌─────────────┐
│   Nginx     │  SSL termination, rate limiting
│  (Port 443) │
└─────────────┘
    │
    ▼
┌─────────────┐
│   OHI-API   │  Internal port 8080
│  (FastAPI)  │
└─────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Neo4j │ Qdrant │ Redis │ vLLM │
└───────────────────────────────┘
```

---

## SSL Certificate Management

### Manual Renewal

Certificates auto-renew every 12 hours if needed. To manually renew:

```bash
docker-compose exec certbot certbot renew
docker-compose exec nginx nginx -s reload
```

### Test Renewal

```bash
docker-compose exec certbot certbot renew --dry-run
```

### View Certificate Info

```bash
docker-compose exec certbot certbot certificates
```

---

## API Usage with HTTPS

### curl

```bash
curl -X POST "https://yourdomain.myfritz.net/api/ohi/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Albert Einstein was born in Germany.", "strategy": "mcp_enhanced"}'
```

### Python

```python
import requests

API_URL = "https://yourdomain.myfritz.net"

response = requests.post(
  f"{API_URL}/api/ohi/api/v1/verify",
  headers={
    "Content-Type": "application/json",
  },
  json={
    "text": "Albert Einstein was born in Germany.",
    "strategy": "mcp_enhanced"
  }
)
print(response.json())
```

### JavaScript

```javascript
const response = await fetch('https://yourdomain.myfritz.net/api/ohi/api/v1/verify', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Albert Einstein was born in Germany.',
    strategy: 'mcp_enhanced'
  }),
});
const result = await response.json();
console.log(result);
```

---

## Security Features

### Built-in Protections

- ✅ **HTTPS/TLS 1.2+** - All traffic encrypted
- ✅ **API Key Authentication** - Required for backend `/api/v1/*` endpoints (proxy injects server key)
- ✅ **Rate Limiting** - 60 requests/minute per IP (Nginx)
- ✅ **Security Headers** - HSTS, X-Frame-Options, etc.
- ✅ **OCSP Stapling** - Faster TLS handshakes

### Recommended Additional Steps

- [ ] Set up fail2ban for brute-force protection
- [ ] Monitor access logs: `docker-compose logs -f nginx`
- [ ] Consider Cloudflare for DDoS protection

---

## Troubleshooting

### Certificate Request Failed

```bash
# Check if port 80 is reachable from the internet
# Use a service like https://portchecker.co/ or https://canyouseeme.org/

# Check nginx logs
docker-compose logs nginx

# Check certbot logs
docker-compose logs certbot
```

### SSL Certificate Errors

```bash
# Verify certificate exists
ls -la data/certbot/live/ohi/

# Re-run certificate setup
./scripts/init-letsencrypt.sh yourdomain.example.com your@email.com
```

### 502 Bad Gateway

```bash
# Check if API is running
docker-compose ps

# Check API logs
docker-compose logs ohi-api
```

### Connection Timeout

- Verify router port forwarding is active
- Check OS firewall allows ports 80 and 443
- Confirm Docker containers are running: `docker-compose ps`

---

## If You Are Behind CGNAT (No Public IP)

If your ISP uses CGNAT, inbound port forwarding will not work. Use one of these options:

1. **Reverse tunnel** (e.g., Cloudflare Tunnel)
2. **Public VPS** as a reverse proxy
3. **DNS-01 validation** for TLS and expose via tunnel/VPN

These options require additional setup outside this guide.
