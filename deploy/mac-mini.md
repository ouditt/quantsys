# QTSYS on a Mac mini + iPad access from anywhere

The Mac mini becomes the **system of record** (plans, paper-day counter, PIT
vintages, agent logs all live there, 24/7), and the iPad reaches it from
anywhere through **Tailscale** — an encrypted private network between your own
devices. Nothing is ever exposed to the public internet: the server keeps
binding to 127.0.0.1 and Tailscale's local proxy is the only way in.

## 1 · Install on the Mac mini (one time, ~10 min)

```bash
# Homebrew python (macOS ships an old one)
brew install python@3.12 git

git clone git@github.com:ouditt/quantsys.git ~/quantsys
cd ~/quantsys
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
# optional FinBERT sentiment (Apple Silicon wheel works directly):
venv/bin/pip install -r requirements-ml.txt
```

## 2 · Move the state from the old machine

On the **Linux box**, from the repo root:

```bash
bash deploy/migrate-state.sh          # -> qtsys-state-YYYYMMDD.tar.gz
scp qtsys-state-*.tar.gz youruser@mac-mini.local:~/quantsys/
```

On the **Mac mini**:

```bash
cd ~/quantsys && tar xzf qtsys-state-*.tar.gz && rm qtsys-state-*.tar.gz
```

This carries `.env` (broker keys), the auto-trader ledger (including your
**60-paper-day counter** — don't restart the clock!), plans, proposals, PIT
vintages, the ML selector, scan results and reports. From this moment run only
ONE home machine: stop the server on the Linux box (`pkill -f "uvicorn
qtsys.server"`) so two engines never trade the same account.

Sanity check: `./start.sh` then open http://127.0.0.1:8001 on the Mac —
the PLAN page should show your existing plan and the paper-days counter.

## 3 · Auto-start + stay awake

Use a **system LaunchDaemon**, not a LaunchAgent. A LaunchAgent runs inside your
login session and macOS blocks it from reading `~/Documents` (TCC privacy) and it
dies at logout. A LaunchDaemon starts at BOOT, before login, and — running in the
system domain — reads the repo from `~/Documents` fine. (This is the same pattern
the sibling Tradesys app uses.)

```bash
# installs /Library/LaunchDaemons/com.qtsys.terminal.plist, binds :8011, starts it
sudo bash deploy/install-daemon.sh

# a trading box must not sleep
sudo pmset -a sleep 0 disksleep 0
```

Manage it:

```bash
sudo launchctl print    system/com.qtsys.terminal | head   # status
sudo launchctl kickstart -k system/com.qtsys.terminal      # restart (apply code/.env)
sudo launchctl bootout  system/com.qtsys.terminal          # stop + unload
```

Logs land in `~/Library/Logs/qtsys/gui.{out,err}.log`. The daemon runs on port
**8011** (8001 is often taken by an unrelated app) and binds `0.0.0.0` so the
iPad can reach it over Tailscale (see §4).

## 4 · Tailscale (Mac mini + iPad)

1. **Mac mini**: `brew install --cask tailscale` (or the Mac App Store app),
   open it, sign in (Google/Apple/GitHub — this creates your private
   "tailnet"). In the Tailscale menu enable **MagicDNS + HTTPS certificates**
   (Admin console → DNS → enable HTTPS).
2. The daemon already binds `0.0.0.0:8011` (via `QTSYS_HOST=0.0.0.0` in the
   plist), so the mini is reachable at its Tailscale hostname — no `tailscale
   serve` and no HTTPS-cert enablement needed. Find your URL:

   ```bash
   tailscale status --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))'
   # e.g.  mouhameds-mac-mini.tail49851e.ts.net  ->  http://<that>:8011
   ```

3. **iPad**: install the **Tailscale app** from the App Store, sign in with
   the SAME account, toggle the VPN on.
4. Open Safari on the iPad → `http://<your-mini>.tailXXXX.ts.net:8011`.
   Everything works — session token, live WebSocket quotes, order confirms,
   the PLAN and PERF pages. Use Safari's **Share → Add to Home Screen** for a
   full-screen app icon.

Abroad = exactly the same: the iPad's Tailscale VPN reaches the mini over any
network (hotel Wi-Fi, cellular), end-to-end encrypted via WireGuard. If the
mini is behind CGNAT it still works (Tailscale relays via DERP).

### Security posture
The server binds `0.0.0.0` but access is intended over the private Tailscale
mesh (WireGuard-encrypted, your-devices-only). Every mutating `/api` call
requires the per-boot session token (injected into the served page only), so
even a same-LAN peer cannot arm the engine, place an order, or fire the kill
switch. Read-only endpoints (quotes/account/plan) are open on the LAN — if that
matters on an untrusted network, set `QTSYS_HOST` back to `127.0.0.1` and use
`tailscale serve` instead. Never forward port 8011 on a router.

## 5 · Day-2 notes

- **Update**: `cd ~/quantsys && git pull && launchctl kickstart -k
  gui/$UID/com.qtsys.terminal`
- **Push notifications** work from the mini exactly as before (set
  `QTSYS_NTFY_TOPIC` in `.env`; the ntfy app also exists on iOS — pair it
  with the iPad for fills/kill/arb pings even when the terminal is closed).
- **Time zone**: keep the mini on your home TZ; the market-hours gate for the
  intraday scan computes US Eastern internally.
- The Linux box can keep a checkout for development; just leave its server
  stopped.
