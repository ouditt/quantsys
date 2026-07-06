#!/usr/bin/env bash
# Bundle ALL runtime state for moving QTSYS to a new home machine.
# Run on the OLD machine from the repo root:  bash deploy/migrate-state.sh
# Produces qtsys-state-<date>.tar.gz — copy it to the new machine's repo
# root and:  tar xzf qtsys-state-*.tar.gz
#
# What travels (none of this is in git): broker keys (.env), agent log +
# posture, auto-trader ledger incl. the 60-day paper-days counter, trade
# plans, proposals inbox, L2 experiment, PIT fundamental vintages, trade
# journal, ML selector store, scan results + caches, agent reports.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="qtsys-state-$(date +%Y%m%d).tar.gz"
tar czf "$OUT" \
    --exclude='qtsys/universe_cache/bars_*' \
    .env \
    qtsys/*.db \
    qtsys/universe_selector.pkl \
    qtsys/registry_results*.csv qtsys/registry_summary.csv \
    qtsys/universe_cache qtsys/reports \
    2>/dev/null || true
chmod 600 "$OUT"    # contains broker keys
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) — copy to the new machine's repo root, then: tar xzf $OUT"
