#!/bin/bash
# Initialize Let's Encrypt certificates for the OHI API
# Usage: ./scripts/init-letsencrypt.sh yourdomain.com your@email.com

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <domain> <email>"
    echo "Example: $0 api.example.com admin@example.com"
    exit 1
fi

DOMAIN=$1
EMAIL=$2
DATA_PATH="./data/certbot"

echo "=== Let's Encrypt Certificate Setup ==="
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Create directories
mkdir -p "$DATA_PATH/conf/live/ohi"
mkdir -p "$DATA_PATH/conf/archive/ohi"
mkdir -p "$DATA_PATH/www/.well-known/acme-challenge"

# Check if certificates already exist
if [ -f "$DATA_PATH/conf/live/ohi/fullchain.pem" ] && [ -s "$DATA_PATH/conf/live/ohi/fullchain.pem" ]; then
    echo "Certificates already exist. To renew, run:"
    echo "  docker-compose exec certbot certbot renew"
    exit 0
fi

# Download recommended TLS parameters
if [ ! -f "$DATA_PATH/conf/options-ssl-nginx.conf" ]; then
    echo "Downloading recommended TLS parameters..."
    curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$DATA_PATH/conf/options-ssl-nginx.conf"
fi

if [ ! -f "$DATA_PATH/conf/ssl-dhparams.pem" ]; then
    echo "Downloading DH parameters..."
    curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem > "$DATA_PATH/conf/ssl-dhparams.pem"
fi

# Create dummy certificate for nginx to start
echo "Creating dummy certificate..."
openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
    -keyout "$DATA_PATH/conf/live/ohi/privkey.pem" \
    -out "$DATA_PATH/conf/live/ohi/fullchain.pem" \
    -subj "/CN=localhost" 2>/dev/null

echo "Starting nginx with dummy certificate..."
docker-compose up -d nginx

# Wait for nginx to start
echo "Waiting for nginx to start..."
sleep 5

# Delete dummy certificate
echo "Removing dummy certificate..."
rm -f "$DATA_PATH/conf/live/ohi/privkey.pem"
rm -f "$DATA_PATH/conf/live/ohi/fullchain.pem"

# Request real certificate
echo "Requesting Let's Encrypt certificate for $DOMAIN..."
docker-compose run --rm certbot certonly --webroot \
    -w /var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN" \
    --cert-name ohi \
    --force-renewal

echo "Reloading nginx..."
docker-compose exec nginx nginx -s reload

echo ""
echo "=== Setup Complete ==="
echo "Your API is now available at: https://$DOMAIN"
echo ""
echo "Certificates will auto-renew via the certbot container."
echo "Test renewal with: docker-compose exec certbot certbot renew --dry-run"
