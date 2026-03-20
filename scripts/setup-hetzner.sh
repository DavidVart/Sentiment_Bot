#!/usr/bin/env bash
# -------------------------------------------------------------------
# Setup script for Hetzner CPX21 (Debian 12).
# Run this ON the server after copying the project files.
#
# Usage (from your Mac):
#   1. Create a CPX21 (Debian 12) in the Hetzner Cloud console
#   2. Copy project files:
#      rsync -avz --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' \
#        ~/Desktop/Sentiment_Bot/ root@<IP>:/opt/sentiment-bot/
#   3. SSH in and run:
#      ssh root@<IP> 'bash /opt/sentiment-bot/scripts/setup-hetzner.sh'
#   4. Access: http://<IP>:8080/tasks
#
# IMPORTANT: Add the Hetzner server's IP to your Cloud SQL authorized
#            networks in GCP Console → SQL → Connections → Networking.
# -------------------------------------------------------------------
set -e

PROJECT_DIR="/opt/sentiment-bot"
cd "$PROJECT_DIR"

echo "=============================================="
echo " Sentiment Bot — Hetzner Server Setup"
echo "=============================================="

# --- 1. System packages ---
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
  python3.12 python3.12-venv python3.12-dev \
  python3-pip \
  libpq-dev gcc \
  curl git \
  ufw

# --- 2. Node.js 20 (for building frontend) ---
echo "[2/7] Installing Node.js 20..."
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
fi
echo "  Node $(node --version), npm $(npm --version)"

# --- 3. Python venv + deps ---
echo "[3/7] Setting up Python environment..."
python3.12 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
echo "  Python $(python --version)"

# --- 4. Build Angular frontend ---
echo "[4/7] Building frontend..."
cd web
npm install --silent
npm run build
cd ..
echo "  Frontend built → web/out/browser/"

# --- 5. Run DB migrations ---
echo "[5/7] Running DB migrations..."
python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from src.db import apply_migrations
apply_migrations()
print('  Migrations applied.')
"

# --- 6. Firewall ---
echo "[6/7] Configuring firewall..."
ufw allow 22/tcp   # SSH
ufw allow 8080/tcp # Dashboard
ufw --force enable
echo "  Firewall: ports 22, 8080 open"

# --- 7. Systemd service ---
echo "[7/7] Creating systemd service..."
cat > /etc/systemd/system/sentiment-bot.service << 'EOF'
[Unit]
Description=Sentiment Bot Dashboard API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sentiment-bot
EnvironmentFile=/opt/sentiment-bot/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/sentiment-bot/.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sentiment-bot
systemctl restart sentiment-bot
sleep 2

# Check if it started
if systemctl is-active --quiet sentiment-bot; then
  echo "  ✅ Service running"
else
  echo "  ❌ Service failed to start. Check: journalctl -u sentiment-bot -n 30"
  exit 1
fi

# Get external IP
EXT_IP=$(curl -s ifconfig.me || hostname -I | awk '{print $1}')

echo ""
echo "=============================================="
echo " Setup complete!"
echo "=============================================="
echo ""
echo " Dashboard:  http://${EXT_IP}:8080/"
echo " Tasks UI:   http://${EXT_IP}:8080/tasks"
echo " Health:     http://${EXT_IP}:8080/health"
echo ""
echo " ⚠️  IMPORTANT: Add ${EXT_IP} to your Cloud SQL"
echo "    authorized networks in GCP Console:"
echo "    SQL → options-agent-db → Connections → Networking"
echo ""
echo " Manage:"
echo "   systemctl status sentiment-bot"
echo "   journalctl -u sentiment-bot -f"
echo ""
