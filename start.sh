#!/usr/bin/env bash
# Start the QTSYS terminal server. Loads broker keys from .env if present.
cd "$(dirname "$0")"
if [ -f .env ]; then
    chmod 600 .env            # broker keys: owner-only, always
    set -a; source .env; set +a
fi
exec /home/mt-consult/trading/QuantSYS/venv/bin/python -m uvicorn qtsys.server:app --host 127.0.0.1 --port "${QTSYS_PORT:-8001}"
