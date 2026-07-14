#!/usr/bin/env bash
# Install the QTSYS terminal as a macOS system LaunchDaemon so it survives
# reboot and the ~/Documents TCC restriction (same mechanism as Tradesys).
# Run:  sudo bash deploy/install-daemon.sh
set -euo pipefail

REPO="/Users/me/Documents/quantsys"
PLIST_SRC="$REPO/deploy/com.qtsys.daemon.plist"
PLIST_DST="/Library/LaunchDaemons/com.qtsys.terminal.plist"
LABEL="com.qtsys.terminal"

if [ "$(id -u)" -ne 0 ]; then
  echo "This installer must run with sudo:  sudo bash deploy/install-daemon.sh" >&2
  exit 1
fi

# log dir (daemon runs as user 'me' and writes here)
install -d -o me -g staff "/Users/me/Library/Logs/qtsys"

# free the port: stop any manually-started uvicorn before the daemon binds :8011
pkill -f "uvicorn qtsys.server" 2>/dev/null || true
sleep 1

# (re)install the plist
launchctl bootout "system/$LABEL" 2>/dev/null || true
cp "$PLIST_SRC" "$PLIST_DST"
chown root:wheel "$PLIST_DST"
chmod 644 "$PLIST_DST"
launchctl bootstrap system "$PLIST_DST"
launchctl enable "system/$LABEL"
launchctl kickstart -k "system/$LABEL"

echo "installed + started $LABEL"
sleep 6
if curl -s -m5 "http://127.0.0.1:8011/api/health" >/dev/null 2>&1; then
  echo "OK — terminal is live on 127.0.0.1:8011 and will auto-start at boot."
else
  echo "started, but health check not answering yet — check ~/Library/Logs/qtsys/gui.err.log"
fi
