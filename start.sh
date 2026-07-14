#!/usr/bin/env bash
# Start the QTSYS terminal server. Loads broker keys from .env if present.
# Portable: finds the venv next to the repo (.venv or ./venv) or one level up
# (../venv, the original Linux layout), else falls back to python3.
cd "$(dirname "$0")"
if [ -f .env ]; then
    chmod 600 .env            # broker keys: owner-only, always
    set -a; source .env; set +a
fi
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
elif [ -x "../venv/bin/python" ]; then
    PY="../venv/bin/python"
else
    PY="$(command -v python3)"
fi
# QTSYS_HOST controls the bind interface. Default 127.0.0.1 (localhost-only, the
# safe dev default). Set QTSYS_HOST=0.0.0.0 to reach the terminal from your iPad
# over Tailscale (the daemon sets this) — access is over the private Tailscale
# mesh and every mutating call still requires the per-boot session token.
exec "$PY" -m uvicorn qtsys.server:app \
    --host "${QTSYS_HOST:-127.0.0.1}" --port "${QTSYS_PORT:-8001}"
