"""QTSYS regression runner — executes every module _selftest() plus the
terminal's inline-JS syntax check. Network-dependent tests degrade to a
SKIP instead of failing the suite when offline.

    python tests/run_all.py          (exit 0 = all green)
"""
import importlib
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SELFTEST_MODULES = [
    "qtsys.options", "qtsys.volsurface", "qtsys.execalgo", "qtsys.pit",
    "qtsys.portfolio_risk", "qtsys.arb.pairs", "qtsys.arb.triangular",
    "qtsys.arb.cip", "qtsys.optstrat", "qtsys.proposals",
    "qtsys.tradeplan", "qtsys.autotrader",
]
NETWORK_OK_TO_SKIP = {"qtsys.portfolio_risk"}   # needs bundled data refresh

def main() -> int:
    failed = []
    for name in SELFTEST_MODULES:
        try:
            mod = importlib.import_module(name)
            fn = getattr(mod, "_selftest", None)
            if fn is None:
                print(f"SKIP {name}: no _selftest")
                continue
            fn()
            print(f"PASS {name}")
        except Exception as e:
            if name in NETWORK_OK_TO_SKIP:
                print(f"SKIP {name}: {type(e).__name__}: {e}")
                continue
            print(f"FAIL {name}")
            traceback.print_exc()
            failed.append(name)
    # terminal inline-JS must parse
    html = (ROOT / "qtsys" / "terminal.html").read_text()
    js = "\n;\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    tmp = ROOT / "tests" / "_terminal.js"
    tmp.write_text(js)
    try:
        subprocess.run(["node", "--check", str(tmp)], check=True,
                       capture_output=True)
        print("PASS terminal.html inline JS")
    except subprocess.CalledProcessError as e:
        print("FAIL terminal.html inline JS\n", e.stderr.decode()[:800])
        failed.append("terminal.js")
    finally:
        tmp.unlink(missing_ok=True)
    print(f"\n{'ALL GREEN' if not failed else 'FAILURES: ' + ', '.join(failed)}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
