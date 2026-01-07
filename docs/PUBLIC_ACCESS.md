# Public Internet Access Setup

This guide explains how to expose the Open Hallucination Index API to the public internet securely with HTTPS.

## Prerequisites

1. Fritz!Box router with internet access
2. A domain name pointing to your public IP (or DynDNS like MyFRITZ!)
3. API running on your local machine
4. Ports 80 and 443 forwarded to your machine

## Quick Start (HTTPS with Let's Encrypt)

### Step 1: Generate an API Key

```bash
# Generate a random 64-character API key
openssl rand -hex 32
```

Add to your `.env` file:
```bash
API_KEY=your_generated_api_key_here
```

### Step 2: Configure Fritz!Box Port Forwarding

Forward **both** ports to your PC:

| Protocol | External Port | Internal Port |
|----------|---------------|---------------|
| TCP | 80 | 80 |
| TCP | 443 | 443 |

See [Fritz!Box Configuration](#fritzbox-port-forwarding) below for detailed steps.

### Step 3: Set Up DynDNS Domain

Use MyFRITZ! or another DynDNS service to get a stable domain name.

### Step 4: Initialize Let's Encrypt Certificate

```bash
# Make the script executable
chmod +x scripts/init-letsencrypt.sh

# Run the setup (replace with your domain and email)
./scripts/init-letsencrypt.sh yourdomain.myfritz.net your@email.com
```

### Step 5: Start All Services

```bash
docker-compose up -d
```

Your API is now available at: `https://yourdomain.myfritz.net`

---

## Fritz!Box Port Forwarding

### Access Fritz!Box Admin Interface

1. Open browser: `http://fritz.box` or `http://192.168.178.1`
2. Login with your Fritz!Box password

### Configure Port Forwarding

1. Navigate to: **Internet** → **Freigaben** → **Portfreigaben**

2. Click **Gerät für Freigaben hinzufügen**

3. Select your computer from the list

4. Add two port forwards:

   **Port 80 (HTTP - for Let's Encrypt):**
   | Setting | Value |
   |---------|-------|
   | Anwendung | Other application |
   | Bezeichnung | OHI-HTTP |
   | Protokoll | TCP |
   | Port an Gerät | 80 |
   | Port extern | 80 |

   **Port 443 (HTTPS):**
   | Setting | Value |
   |---------|-------|
   | Anwendung | Other application |
   | Bezeichnung | OHI-HTTPS |
   | Protokoll | TCP |
   | Port an Gerät | 443 |
   | Port extern | 443 |

5. Click **Übernehmen**

### Configure MyFRITZ! (Recommended)

1. Go to: **Internet** → **MyFRITZ!-Konto**
2. Register or login with your AVM account
3. Your domain: `yourusername.myfritz.net`

---

## Architecture

```
Internet
    │
    ▼
┌─────────────┐
│  Fritz!Box  │  Port 80/443 forwarded
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
curl -X POST "https://yourdomain.myfritz.net/api/v1/verify" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"text": "Albert Einstein was born in Germany.", "strategy": "mcp_enhanced"}'
```

### Python

```python
import requests

API_URL = "https://yourdomain.myfritz.net"
API_KEY = "your_api_key_here"

response = requests.post(
    f"{API_URL}/api/v1/verify",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
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
const response = await fetch('https://yourdomain.myfritz.net/api/v1/verify', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your_api_key_here',
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
- ✅ **API Key Authentication** - Required for `/api/v1/*` endpoints
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
# Check if port 80 is reachable from internet
# Use a service like https://portchecker.co/

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
./scripts/init-letsencrypt.sh yourdomain.myfritz.net your@email.com
```

### 502 Bad Gateway

```bash
# Check if API is running
docker-compose ps

# Check API logs
docker-compose logs ohi-api
```

### Connection Timeout

- Verify Fritz!Box port forwarding is active
- Check Windows Firewall allows ports 80 and 443
- Confirm Docker containers are running: `docker-compose ps`
