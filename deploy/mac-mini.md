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

```bash
# start at login, restart on crash (edit YOURUSER in the plist first)
sed -i '' "s/YOURUSER/$USER/g" deploy/com.qtsys.terminal.plist
cp deploy/com.qtsys.terminal.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.qtsys.terminal.plist

# a trading box must not sleep
sudo pmset -a sleep 0 disksleep 0
```

Logs land in `/tmp/qtsys.out` / `/tmp/qtsys.err`.

## 4 · Tailscale (Mac mini + iPad)

1. **Mac mini**: `brew install --cask tailscale` (or the Mac App Store app),
   open it, sign in (Google/Apple/GitHub — this creates your private
   "tailnet"). In the Tailscale menu enable **MagicDNS + HTTPS certificates**
   (Admin console → DNS → enable HTTPS).
2. Serve the terminal over the tailnet with a real HTTPS cert:

   ```bash
   tailscale serve --bg https / http://127.0.0.1:8001
   tailscale serve status     # shows your URL, e.g.:
   #  https://mac-mini.tail1234.ts.net
   ```

3. **iPad**: install the **Tailscale app** from the App Store, sign in with
   the SAME account, toggle the VPN on.
4. Open Safari on the iPad → `https://mac-mini.tail1234.ts.net`. Everything
   works — session token, live WebSocket quotes, order confirms, the PLAN
   page — because Tailscale proxies to 127.0.0.1 on the mini. Use Safari's
   **Share → Add to Home Screen** to get a full-screen app icon.

Abroad = exactly the same: the iPad's Tailscale VPN reaches the mini over any
network (hotel Wi-Fi, cellular), end-to-end encrypted via WireGuard. If the
mini is behind CGNAT it still works (Tailscale relays via DERP).

### Why this and not port-forwarding
The server has a session token on mutations but no TLS and no login page.
Tailscale gives you: WireGuard encryption, your-devices-only access, real
HTTPS, and zero router configuration. Never bind the server to 0.0.0.0 and
never forward port 8001 on a router.

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
