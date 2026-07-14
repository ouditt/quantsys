#!/usr/bin/env python3
"""Launcher for the QTSYS terminal — used by the macOS LaunchDaemon.

macOS TCC blocks /bin/bash from reading ~/Documents, so the daemon cannot run
start.sh there. It CAN run the venv's python directly, because that binary
(/Library/Frameworks/.../python3.13) already holds the Documents grant (the
sibling Tradesys daemon uses the same binary). This launcher mirrors Tradesys's
run.py: it loads .env in-process (start.sh's job) and then boots uvicorn — so no
shell is involved.

Run directly for a normal start too:  .venv/bin/python run.py
"""
from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=value lines, # comments and
    blanks ignored, surrounding quotes stripped. Existing env vars win, so the
    daemon's plist EnvironmentVariables (QTSYS_HOST/PORT) take precedence."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, path)
    if not os.path.exists(p):
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                os.environ.setdefault(k, v)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    load_dotenv()
    import uvicorn
    uvicorn.run("qtsys.server:app",
                host=os.environ.get("QTSYS_HOST", "127.0.0.1"),
                port=int(os.environ.get("QTSYS_PORT", "8001")))


if __name__ == "__main__":
    main()
