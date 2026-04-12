#!/usr/bin/env bash
# =============================================================================
# install_macos_service.sh
# Installs the Telegram Bot as a macOS launchd service (auto-start, auto-restart).
#
# Run from the project root:
#   bash deploy/install_macos_service.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_SRC="$SCRIPT_DIR/com.vassiliy.telegrambot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.vassiliy.telegrambot.plist"
SERVICE_LABEL="com.vassiliy.telegrambot"

echo "🤖 Telegram Bot — macOS Service Installer"
echo "==========================================="
echo "Project: $PROJECT_DIR"
echo ""

# ── Check for .env ────────────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "❌ .env file not found in project root!"
    echo "   Copy .env.example to .env and fill in your keys:"
    echo "   cp .env.example .env"
    exit 1
fi
echo "✅ .env found"

# ── Check python3 ─────────────────────────────────────────────────
PYTHON="$(which python3 || true)"
if [ -z "$PYTHON" ]; then
    echo "❌ python3 not found. Install from https://python.org"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# ── Install Python dependencies ───────────────────────────────────
echo ""
echo "📦 Installing Python dependencies..."
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo "✅ Dependencies installed"

# ── Patch plist with actual project path ─────────────────────────
echo ""
echo "📝 Configuring service with path: $PROJECT_DIR"
sed "s|/Users/imac/Desktop/telegram-bot-v2|$PROJECT_DIR|g" \
    "$PLIST_SRC" > "$PLIST_DST"
echo "✅ Plist written to $PLIST_DST"

# ── Unload existing service if running ───────────────────────────
if launchctl list | grep -q "$SERVICE_LABEL"; then
    echo ""
    echo "🔄 Stopping existing service..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    sleep 2
fi

# ── Load service ─────────────────────────────────────────────────
echo ""
echo "🚀 Loading service..."
launchctl load "$PLIST_DST"
sleep 3

# ── Status check ─────────────────────────────────────────────────
echo ""
if launchctl list | grep -q "$SERVICE_LABEL"; then
    PID=$(launchctl list | grep "$SERVICE_LABEL" | awk '{print $1}')
    echo "✅ Service is RUNNING (PID: $PID)"
else
    echo "⚠️  Service may not have started yet. Check logs:"
fi

echo ""
echo "==========================================="
echo "📋 USEFUL COMMANDS:"
echo ""
echo "  View live logs:"
echo "    tail -f /tmp/telegrambot.log"
echo ""
echo "  View error logs:"
echo "    tail -f /tmp/telegrambot_err.log"
echo ""
echo "  Check service status:"
echo "    launchctl list | grep telegrambot"
echo ""
echo "  Restart service:"
echo "    launchctl kickstart -k gui/\$(id -u)/com.vassiliy.telegrambot"
echo ""
echo "  Stop service:"
echo "    launchctl unload ~/Library/LaunchAgents/com.vassiliy.telegrambot.plist"
echo ""
echo "  Start service again:"
echo "    launchctl load ~/Library/LaunchAgents/com.vassiliy.telegrambot.plist"
echo ""
echo "✅ Done! The bot will now start automatically on login and restart on crash."
