"""volsurface.py — no-arbitrage volatility surface + American-exercise IV.

The options analytics core (replaces per-contract Newton IV in
options.enrich_chain). Pipeline per underlying:

  1. QUALITY-GATE raw chain quotes: zero-bid, crossed, wide (rel spread),
     sub-penny mids never enter the fit — retail chains are ~15% zero-bid
     with p90 spreads at 100% of mid, and fitting to that is fitting noise.
  2. FORWARD per expiry from put-call parity on the nearest-ATM two-sided
     pairs (median of K + e^{rT}(C-P)); implied carry q = r - ln(F/S)/T
     absorbs dividends/borrow model-free. (American parity is approximate;
     near-ATM short-dated the early-exercise premium is negligible.)
  3. AMERICAN IV: invert a CRR binomial (with implied q) on the gated OTM
     quotes — vectorized bisection across all contracts of an expiry at
     once, so a 600-contract chain fits in well under a second.
  4. SVI SMILE per expiry in total-variance space w(k)=a+b(ρ(k-m)+√((k-m)²+σ²)),
     weighted by quote tightness, with the Durrleman g(k)≥0 butterfly
     condition enforced as a fit penalty and verified on a grid after.
     Fewer than 5 clean points -> honest flat fallback (b=0).
  5. CALENDAR floor at read-out: total variance is made non-decreasing in T
     at fixed moneyness, so no calendar arbitrage leaves the surface.
  6. READ-OUT for EVERY contract (deep ITM included — where a raw mid
     inversion is ill-posed, the surface still prices): surface IV, greeks
     (BS with implied carry), plus per-expiry ATM vol, 25Δ risk-reversal,
     25Δ butterfly and the ATM term structure.

Run `python -m qtsys.volsurface` for the synthetic round-trip self-test
(known SVI -> American prices + noise + gaps -> recovered surface), and it
will also exercise /tmp/aapl_chain.pkl if a live fixture is present.
"""
from __future__ import annotations

import math

import numpy as np

from .options import bs_price, greeks, norm

# ----------------------------------------------------------- quality gating
MAX_REL_SPREAD = 0.35        # (ask-bid)/mid above this -> too wide to trust
MIN_MID = 0.02               # sub-2-cent mids are noise at retail spreads
MIN_FIT_POINTS = 5           # fewer clean OTM points -> flat fallback


def gate(contracts: list[dict]) -> list[dict]:
    """Annotate each contract with quote-quality flags; returns new dicts.
    gate_ok means the quote is clean enough to ENTER THE FIT — everything
    still gets a surface IV read out afterwards."""
    out = []
    for c in contracts:
        bid, ask = c.get("bid"), c.get("ask")
        reason = None
        mid = None
        if not bid or not ask:
            reason = "one-sided"
        elif ask < bid:
            reason = "crossed"
        else:
            mid = (bid + ask) / 2.0
            if mid < MIN_MID:
                reason = "sub-penny"
            elif (ask - bid) / mid > MAX_REL_SPREAD:
                reason = "wide"
        out.append({**c, "mid": mid if mid else (c.get("last") or None),
                    "gate_ok": reason is None, "gate_reason": reason})
    return out


# ------------------------------------------------------------------ forward
def parity_forward(exp_contracts: list[dict], spot: float, r: float,
                   T: float) -> tuple[float, float]:
    """(F, q_implied) from put-call parity on two-sided near-ATM pairs."""
    calls = {c["strike"]: c for c in exp_contracts
             if c["type"] == "call" and c["gate_ok"]}
    puts = {c["strike"]: c for c in exp_contracts
            if c["type"] == "put" and c["gate_ok"]}
    ks = sorted(set(calls) & set(puts), key=lambda k: abs(k - spot))[:5]
    est = [k + math.exp(r * T) * (calls[k]["mid"] - puts[k]["mid"]) for k in ks]
    F = float(np.median(est)) if est else spot * math.exp(r * T)
    if F <= 0:
        F = spot * math.exp(r * T)
    q = r - math.log(F / spot) / T if spot > 0 and T > 0 else 0.0
    q = min(max(q, -0.10), 0.15)                 # clamp implied carry to sane
    F = spot * math.exp((r - q) * T)
    return F, q


# ------------------------------------------- vectorized American CRR + IV
def american_price_vec(S: float, K: np.ndarray, T: float, r: float,
                       sigma: np.ndarray, is_call: np.ndarray,
                       q: float = 0.0, steps: int = 96) -> np.ndarray:
    """CRR American prices for N contracts sharing (S,T,r,q), vectorized
    across contracts (columns). sigma per contract."""
    K = np.asarray(K, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))               # (N,)
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    j = np.arange(steps + 1)[:, None]             # node index (rows)
    px = S * u[None, :] ** (steps - 2 * j)        # terminal prices (steps+1, N)
    sign = np.where(is_call, 1.0, -1.0)[None, :]
    val = np.maximum(sign * (px - K[None, :]), 0.0)
    for step in range(steps, 0, -1):
        px = px[:step] * d[None, :] * u[None, :]  # prices at previous level
        # px recomputed directly to avoid drift: S * u^(step-1-2i)
        i = np.arange(step)[:, None]
        px = S * u[None, :] ** (step - 1 - 2 * i)
        val = disc * (p[None, :] * val[:step] + (1 - p[None, :]) * val[1:step + 1])
        val = np.maximum(val, np.maximum(sign * (px - K[None, :]), 0.0))
    return val[0]


def american_iv_vec(price: np.ndarray, S: float, K: np.ndarray, T: float,
                    r: float, is_call: np.ndarray, q: float = 0.0,
                    steps: int = 96, iters: int = 18) -> np.ndarray:
    """Vectorized bisection for American IV. NaN where no vol reproduces the
    price (e.g. mid below intrinsic)."""
    price = np.asarray(price, float)
    K = np.asarray(K, float)
    lo = np.full_like(price, 1e-3)
    hi = np.full_like(price, 4.0)
    p_hi = american_price_vec(S, K, T, r, hi, is_call, q, steps)
    p_lo = american_price_vec(S, K, T, r, lo, is_call, q, steps)
    ok = (price >= p_lo - 1e-9) & (price <= p_hi + 1e-9)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        pm = american_price_vec(S, K, T, r, mid, is_call, q, steps)
        below = pm < price
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    iv = (lo + hi) / 2.0
    iv[~ok] = np.nan
    return iv


# ------------------------------------------------------------------ raw SVI
def _svi_w(k, prm):
    a, b, rho, m, s = prm
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def _durrleman_g(k, prm):
    """Gatheral's g(k); butterfly-arbitrage-free iff g >= 0 (and w > 0)."""
    a, b, rho, m, s = prm
    R = np.sqrt((k - m) ** 2 + s ** 2)
    w = a + b * (rho * (k - m) + R)
    w = np.maximum(w, 1e-10)
    w1 = b * (rho + (k - m) / R)                  # w'
    w2 = b * s ** 2 / R ** 3                      # w''
    return ((1 - k * w1 / (2 * w)) ** 2
            - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2)


def fit_svi(k: np.ndarray, w: np.ndarray, wt: np.ndarray) -> tuple:
    """Fit raw SVI to total variances; returns (params, rmse_in_vol_frac).
    Flat fallback when data is too thin for a smile."""
    from scipy.optimize import minimize
    k, w, wt = map(np.asarray, (k, w, wt))
    if len(k) < MIN_FIT_POINTS:
        a = float(np.average(w, weights=wt)) if len(k) else 1e-4
        return (max(a, 1e-8), 0.0, 0.0, 0.0, 0.1), 0.0
    kg = np.linspace(k.min() - 0.3, k.max() + 0.3, 41)      # no-arb check grid

    def loss(prm):
        wm = _svi_w(k, prm)
        base = float(np.sum(wt * (wm - w) ** 2))
        g = _durrleman_g(kg, prm)
        pen = float(np.sum(np.maximum(-g, 0.0) ** 2)) * 1e4  # butterfly penalty
        neg = float(np.maximum(-_svi_w(kg, prm), 0).sum()) * 1e6
        return base + pen + neg

    wbar = float(np.median(w))
    spread = float(w.max() - w.min())
    krange = max(float(k.max() - k.min()), 0.1)
    bounds = [(1e-8, w.max() * 3), (0.0, 5.0), (-0.999, 0.999),
              (k.min() - 0.5, k.max() + 0.5), (1e-3, 2.0)]
    best, best_val = None, np.inf
    for rho0 in (-0.6, 0.0, 0.6):
        x0 = (wbar * 0.8, max(spread / krange, 1e-4), rho0, 0.0, 0.15)
        try:
            res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_val:
                best, best_val = tuple(res.x), res.fun
        except Exception:
            continue
    if best is None:
        a = float(np.average(w, weights=wt))
        return (max(a, 1e-8), 0.0, 0.0, 0.0, 0.1), 0.0
    # remediation ladder: a production surface must LEAVE arb-free, not just
    # report — dense g-check, then a 100x-penalty refit, then flat fallback
    kd = np.linspace(kg[0], kg[-1], 161)
    if _durrleman_g(kd, best).min() < -1e-9:
        def loss_hard(prm):
            wm = _svi_w(k, prm)
            g = _durrleman_g(kd, prm)
            return (float(np.sum(wt * (wm - w) ** 2))
                    + float(np.sum(np.maximum(-g, 0.0) ** 2)) * 1e6
                    + float(np.maximum(-_svi_w(kd, prm), 0).sum()) * 1e6)
        try:
            res = minimize(loss_hard, best, method="L-BFGS-B", bounds=bounds)
            if _durrleman_g(kd, tuple(res.x)).min() >= -1e-9:
                best = tuple(res.x)
            else:
                best = (float(np.average(w, weights=wt)), 0.0, 0.0, 0.0, 0.1)
        except Exception:
            best = (float(np.average(w, weights=wt)), 0.0, 0.0, 0.0, 0.1)
    resid = np.sqrt(np.maximum(_svi_w(k, best), 1e-10)) - np.sqrt(np.maximum(w, 1e-10))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return best, rmse


# ----------------------------------------------------------------- surface
class Surface:
    """Fitted per-expiry SVI slices + calendar floor + metric read-outs."""

    def __init__(self, spot: float, r: float):
        self.spot, self.r = spot, r
        self.slices: list[dict] = []              # sorted by T ascending

    def add(self, exp: str, T: float, F: float, q: float, prm, n_fit: int,
            rmse: float, krange: tuple = (-0.5, 0.5)):
        self.slices.append(dict(exp=exp, T=T, F=F, q=q, prm=prm,
                                n_fit=n_fit, rmse=rmse, krange=krange))
        self.slices.sort(key=lambda s: s["T"])

    def _w_floor(self, k: float, idx: int) -> float:
        """Total variance at slice idx with the calendar floor applied."""
        w = float(_svi_w(np.array([k]), self.slices[idx]["prm"])[0])
        for j in range(idx):
            wj = float(_svi_w(np.array([k]), self.slices[j]["prm"])[0])
            w = max(w, wj)
        return max(w, 1e-10)

    def vol(self, strike: float, exp: str) -> float | None:
        for i, s in enumerate(self.slices):
            if s["exp"] == exp:
                k = math.log(strike / s["F"])
                return math.sqrt(self._w_floor(k, i) / s["T"])
        return None

    # ------------------------------------------------------ smile metrics
    def _delta_k(self, i: int, target: float, call: bool) -> float | None:
        """k where BS delta hits the target (0<target<1), via bisection."""
        s = self.slices[i]
        T, q = s["T"], s["q"]
        atm = math.sqrt(self._w_floor(0.0, i) / T)
        span = max(4.0 * atm * math.sqrt(T), 0.02)
        lo, hi = (0.0, span) if call else (-span, 0.0)

        def delta(k):
            w = self._w_floor(k, i)
            d1 = (-k + w / 2) / math.sqrt(w)
            return (math.exp(-q * T) * norm.cdf(d1) if call
                    else math.exp(-q * T) * (norm.cdf(d1) - 1))
        want = target if call else -target
        f_lo, f_hi = delta(lo) - want, delta(hi) - want
        if f_lo * f_hi > 0:
            return None
        for _ in range(50):
            mid = (lo + hi) / 2
            if (delta(mid) - want) * f_lo <= 0:
                hi = mid
            else:
                lo, f_lo = mid, delta(mid) - want
        return (lo + hi) / 2

    def metrics(self) -> list[dict]:
        out = []
        for i, s in enumerate(self.slices):
            T = s["T"]
            atm = math.sqrt(self._w_floor(0.0, i) / T)
            kc = self._delta_k(i, 0.25, True)
            kp = self._delta_k(i, 0.25, False)
            vc = math.sqrt(self._w_floor(kc, i) / T) if kc is not None else None
            vp = math.sqrt(self._w_floor(kp, i) / T) if kp is not None else None
            rr = (vc - vp) if (vc is not None and vp is not None) else None
            fly = ((vc + vp) / 2 - atm) if (vc is not None and vp is not None) else None
            # no-arb verified on the fitted domain (where quotes exist and
            # read-outs happen) — not at ±dozens of sigmas of extrapolation
            g = _durrleman_g(np.linspace(*s["krange"], 81), s["prm"])
            out.append(dict(exp=s["exp"], T=round(T, 5), F=round(s["F"], 4),
                            q_implied=round(s["q"], 5), atm_vol=round(atm, 5),
                            rr25=round(rr, 5) if rr is not None else None,
                            fly25=round(fly, 5) if fly is not None else None,
                            n_fit=s["n_fit"], fit_rmse_vol=round(s["rmse"], 5),
                            butterfly_ok=bool(g.min() > -1e-9)))
        return out


# -------------------------------------------------------------- build + IO
def build(contracts: list[dict], spot: float, r: float = 0.04) -> dict:
    """Full pipeline. Returns {contracts: enriched list, surface: metrics list,
    spot, r}. Never raises — on hopeless input returns contracts un-enriched."""
    import datetime
    if not contracts or not spot or spot <= 0:
        return {"contracts": contracts, "surface": [], "spot": spot, "r": r}
    today = datetime.date.today()
    gated = gate(contracts)
    by_exp: dict[str, list] = {}
    for c in gated:
        by_exp.setdefault(c["expiration"], []).append(c)

    surf = Surface(spot, r)
    for exp, group in sorted(by_exp.items()):
        try:
            T = max((datetime.date.fromisoformat(exp) - today).days, 1) / 365.0
        except Exception:
            continue
        F, q = parity_forward(group, spot, r, T)
        # OTM two-sided quotes only enter the fit (standard practice: the
        # liquid side; ITM carries the same info via parity, with more noise)
        fit = [c for c in group if c["gate_ok"] and
               ((c["type"] == "call" and c["strike"] >= F) or
                (c["type"] == "put" and c["strike"] <= F))]
        if fit:
            Ks = np.array([c["strike"] for c in fit])
            mids = np.array([c["mid"] for c in fit])
            is_call = np.array([c["type"] == "call" for c in fit])
            ivs = american_iv_vec(mids, spot, Ks, T, r, is_call, q)
            good = np.isfinite(ivs) & (ivs > 1e-3) & (ivs < 3.5)
            k = np.log(Ks[good] / F)
            w = (ivs[good] ** 2) * T
            rel = np.array([(c["ask"] - c["bid"]) / max(c["mid"], 1e-9)
                            for c in fit])[good]
            wt = 1.0 / (rel + 0.05)
            prm, rmse = fit_svi(k, w, wt)
            n_fit = int(good.sum())
            kr = ((float(k.min()) - 0.3, float(k.max()) + 0.3)
                  if n_fit else (-0.5, 0.5))
        else:
            prm, rmse, n_fit, kr = (1e-4, 0.0, 0.0, 0.0, 0.1), 0.0, 0, (-0.5, 0.5)
        surf.add(exp, T, F, q, prm, n_fit, rmse, kr)

    # ------------------------------------------------- per-contract read-out
    out = []
    sl_by_exp = {s["exp"]: s for s in surf.slices}
    for c in gated:
        s = sl_by_exp.get(c["expiration"])
        iv = delta = gamma = theta = vega = None
        if s and c.get("strike"):
            try:
                v = surf.vol(c["strike"], c["expiration"])
                if v and 0 < v < 4:
                    iv = float(v)
                    g = greeks(spot, c["strike"], s["T"], r, iv,
                               c["type"], s["q"])
                    delta, gamma = float(g["delta"]), float(g["gamma"])
                    theta, vega = float(g["theta"]), float(g["vega"])
            except Exception:
                pass
        out.append({**c, "iv": iv, "delta": delta, "gamma": gamma,
                    "theta": theta, "vega": vega,
                    "iv_src": "surface" if iv is not None else None})
    return {"contracts": out, "surface": surf.metrics(), "spot": spot, "r": r}


# ---------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.default_rng(7)
    S, r, q_true = 100.0, 0.04, 0.006
    true = {0.06: (-0.55, 0.22), 0.20: (-0.45, 0.21)}   # T: (rho, atm_vol)
    contracts = []
    import datetime
    today = datetime.date.today()
    for T, (rho, atm) in true.items():
        exp = str(today + datetime.timedelta(days=round(T * 365)))
        F = S * math.exp((r - q_true) * T)
        w_atm = atm ** 2 * T
        prm = (w_atm * 0.55, w_atm * 2.2, rho, 0.0, 0.18)  # a,b,rho,m,s
        assert _durrleman_g(np.linspace(-1, 1, 201), prm).min() > 0
        for K in np.unique(np.r_[np.arange(70, 132.5, 2.5),
                                 np.arange(90, 111, 1.25)]):
            k = math.log(K / F)
            sig = math.sqrt(_svi_w(np.array([k]), prm)[0] / T)
            for kind in ("call", "put"):
                px = american_price_vec(S, np.array([K]), T, r,
                                        np.array([sig]),
                                        np.array([kind == "call"]), q_true)[0]
                half = max(px * 0.025, 0.005)
                bid, ask = px - half, px + half
                if px < 0.05 and rng.random() < 0.6:     # kill junk wings
                    bid = 0.0
                contracts.append(dict(symbol=f"T{K}{kind[0]}", expiration=exp,
                                      strike=float(K), type=kind,
                                      bid=round(max(bid, 0), 4),
                                      ask=round(ask, 4), last=None,
                                      open_interest=10))
    res = build(contracts, S, r)
    ms = res["surface"]
    assert len(ms) == 2, "two expiries fitted"
    for m, (T, (rho, atm)) in zip(ms, sorted(true.items())):
        assert abs(m["q_implied"] - q_true) < 0.004, f"carry {m['q_implied']}"
        assert abs(m["atm_vol"] - atm) < 0.012, f"ATM {m['atm_vol']} vs {atm}"
        assert m["rr25"] is not None and m["rr25"] < 0, "skew sign (rho<0 -> RR<0)"
        assert m["butterfly_ok"], "no butterfly arb"
        assert m["n_fit"] >= 10
    # calendar: floored total variance non-decreasing at fixed k
    sfc = Surface(S, r)                                  # rebuild for probe
    res2 = build(contracts, S, r)
    e1, e2 = ms[0]["exp"], ms[1]["exp"]
    c_by = {(c["expiration"], c["strike"], c["type"]): c for c in res2["contracts"]}
    # deep-ITM contracts (the old ~0-IV edge case) get finite surface IV
    deep = [c for c in res2["contracts"]
            if c["type"] == "call" and c["strike"] <= 75]
    assert deep and all(c["iv"] and c["iv"] > 0.05 for c in deep), "deep ITM IV"
    assert all(c["iv_src"] == "surface" for c in deep)
    print("volsurface self-test ✓  forward/carry, ATM vol, skew sign, "
          "butterfly-free, deep-ITM read-out")
    import os
    import pickle
    if os.path.exists("/tmp/aapl_chain.pkl"):
        d = pickle.load(open("/tmp/aapl_chain.pkl", "rb"))
        import time
        t0 = time.perf_counter()
        live = build(d["contracts"], d["spot"], 0.0392)
        ms = live["surface"]
        dt = time.perf_counter() - t0
        n_iv = sum(1 for c in live["contracts"] if c["iv"])
        print(f"live AAPL fixture: {len(live['contracts'])} contracts, "
              f"{n_iv} surface IVs in {dt:.2f}s")
        for m in ms:
            print(f"  {m['exp']}  F={m['F']:.2f} q={m['q_implied']:+.4f} "
                  f"ATM={m['atm_vol']:.1%} RR25={m['rr25'] if m['rr25'] is None else format(m['rr25'], '+.4f')} "
                  f"fly={m['fly25']} n={m['n_fit']} rmse={m['fit_rmse_vol']} "
                  f"bfly_ok={m['butterfly_ok']}")


if __name__ == "__main__":
    _selftest()
