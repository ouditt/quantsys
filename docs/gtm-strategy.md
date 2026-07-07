# QTSYS Broker — Go-to-Market Strategy
### The world's first agentic brokerage · Africa + Europe · built on Alpaca Broker API
*Version 1.0 — July 2026*

---

## 0 · The one-line thesis

**Every competitor sells access. We sell a desk.** Bamboo, Hisa, Trading 212 and Trade Republic give customers a buy button; QTSYS gives every customer — from a Lagos student with $10 to a Frankfurt professional with €100k — a Bloomberg-grade team of AI agents: a research analyst, a risk officer, a portfolio manager that drafts a daily plan, a voice copilot that answers "what's my risk?" in plain language, and a verification gate (DSR) that refuses to trade unproven strategies. Nobody in either region offers this. The product already exists — this document is how we sell it.

---

## 1 · Why now, and why Alpaca

- **Infrastructure is solved.** Alpaca Broker API is proven, end-to-end brokerage infrastructure powering **7M+ accounts across 40+ countries** (partners include SBI Securities in Japan, Sav in the GCC, ZAD in Kuwait), with fractional shares, options, crypto, and a 2026 Nasdaq data partnership. We don't build custody, clearing, or execution — we build the experience layer, which is exactly where our advantage lives.
- **The omnibus model fits emerging markets.** Alpaca's model lets a locally-licensed partner onboard customers under its own brand while assets sit at a US broker-dealer (SIPC-protected) — the exact structure Bamboo/Hisa already use. We can match their regulatory posture and beat them on product and price.
- **The AI moment.** Retail investors everywhere are asking ChatGPT what to buy. No broker in Africa or Europe has shipped a *grounded, account-aware, verification-gated* AI desk. We have. First-mover window: 12–24 months before incumbents copy credibly.
- **Europe just banned payment-for-order-flow** (MiFIR, fully effective 2026). Neobroker revenue models across the EU are being forced onto FX spreads, cash interest, and subscriptions — a reset moment when switching costs are low and a differentiated entrant can take share.

---

## 2 · Product: what the customer actually gets

| Layer | What it is | Who else has it |
|---|---|---|
| **Core brokerage** | US stocks/ETFs (fractional from $1), options, crypto — via Alpaca | Everyone (table stakes) |
| **The Agent Desk** | Research Analyst, Risk Officer, Fundamental Analyst, Arb Strategist, Microstructure Analyst working 24/7 on *your* account | **No one** |
| **Daily Trade Plan** | PM agent drafts a plan each morning; specialist agents debate it (one bounded round); human-readable plan + deliberation transcript | **No one** |
| **Verification gate** | Strategies must pass a deflated-Sharpe (DSR) statistical gate before the machine may trade them; the user sets the threshold | **No one** — this is our trust story |
| **Auto-trader (opt-in)** | Guarded autonomous execution: paper-proven-60-days lock, per-symbol caps, daily-loss breaker, kill switch | Copy-trading exists (eToro); *verified, guard-railed* autonomy does not |
| **Voice Copilot** | Ask anything about your account by voice; answers grounded in live data; morning brief + day wrap read aloud; remote confirm via Telegram | **No one** |
| **Pro analytics** | Options vol surface, strategy builder, L2 depth, factor/VaR attribution, SEC-filing AI summaries | IBKR (hostile UX), Bloomberg ($24k/yr) |

**Tiering the product (see Pricing):** the buy button is free; the desk is the subscription.

---

## 3 · Regulatory architecture (the honest, load-bearing section)

This is the hardest part of the plan and it determines sequencing. Three postures:

**A. Introducing / sub-broker model (Africa fast path).**
Local entity holds the local license; customer accounts open at Alpaca Securities (US). Precedents per market:
- **Nigeria** — SEC Nigeria digital sub-broker / "digital investment platform" registration (the Bamboo/Chaka path; Chaka was first licensed in 2021). Time: 6–9 months.
- **Kenya** — CMA Regulatory Sandbox entry, then full investment-adviser/broker license (the Hisa path). Time: sandbox in ~3 months.
- **South Africa** — FSCA FSP license (Cat I) + partnership with a local Cat II if we add discretionary features. EasyEquities' home turf; enter last in Africa.
- **Ghana / Egypt** — SEC Ghana and FRA Egypt respectively; Egypt has Thndr as a strong local incumbent — partner or delay.

**B. MiFID passport (Europe).**
One EU investment-firm license passports to all 27+EEA states. Realistic routes, in order of speed:
1. **Lithuania (Bank of Lithuania)** or **Cyprus (CySEC)** MiFID license — 9–15 months, €150k–€750k initial capital depending on permissions. Alpaca already documents EU integration.
2. **Faster interim:** operate as a **tied agent** of an existing EU MiFID firm (several firms rent this), launching in ~3–4 months while our own license processes.
3. **UK separate:** FCA authorization or an FCA-authorized principal (Appointed Representative regime) — treat the UK as its own phase-2 market.

**C. What we never do:** onboard EU/African customers on a pure US-only basis with no local wrapper. That's the compliance corner-cutting that kills brokers. Every market gets a legal opinion before a single ad runs.

**Marketing rule:** the AI desk is decision-*support* plus an execution tool with explicit user-set guardrails — we market verification and risk controls, never "returns." This is both the compliant posture and, genuinely, the brand.

---

## 4 · Geography: sequencing and rationale

**Phase 1 (months 0–9): Nigeria + Kenya.**
- Why first: largest African retail-investing demand (naira devaluation makes **USD assets an inflation hedge** — the actual #1 purchase driver), proven regulatory paths, English-speaking, mobile-money rails, and competitors are beatable on both price and product (Bamboo: 1.5%/trade; Hisa: 1% flat).
- Beachhead goal: 100k funded accounts, prove the agentic wedge in a market where nobody else can copy it quickly.

**Phase 2 (months 6–18, overlapping): EU via tied agent → own MiFID license.**
- Entry markets: **Poland, Ireland, Netherlands, Nordics/Baltics** — high English proficiency, underserved relative to Germany/France, lighter incumbent gravity than Trade Republic's home market. Germany and France once localized (language models for the voice copilot are the gate — see Product-localization).
- Goal: 150k funded accounts by month 18.

**Phase 3 (months 12–24): Ghana, South Africa, Egypt (partner-led), UK, and the African diaspora in Europe** — a deliberately targeted segment: EU-resident Africans investing for family back home; one product, both licenses, remittance-to-investment flows.

**Explicitly not now:** US retail (Alpaca's own partners saturate it), MENA (Sav/ZAD already own the Alpaca lane), Francophone Africa (BRVM/CEMAC regulatory complexity — revisit at scale).

---

## 5 · Customer segments (ICP by region)

**Africa**
1. **The Dollar-Hedger (primary).** 24–40, urban Lagos/Nairobi, salaried or freelance, already uses fintech (OPay, M-Pesa), buys USD assets to escape currency erosion. Needs: low FX cost, fractional shares, trust. *Message: "Own dollars that work. Your AI desk watches them."*
2. **The Aspiring Trader.** 18–30, follows FX/crypto Telegram groups, burned by scams and prop-firm schemes. Needs: a *legitimate* platform that teaches while it protects — the DSR gate and risk officer are the product. *Message: "The first broker that tells you when your strategy is statistically fake."*
3. **The Diaspora Investor** (activated Phase 3). Sends money home; wants to invest for family. Cross-border account pairing.

**Europe**
1. **The Disillusioned Neobroker User (primary).** 25–45, has Trading 212/Trade Republic, holds ETFs, wants more intelligence without becoming a full-time trader. Needs: the daily plan, briefings, and copilot layered on commission-free basics. *Message: "Your broker gives you a button. We give you a desk."*
2. **The Quant-Curious Professional.** 30–55, tech/finance job, IBKR feels hostile, Bloomberg is $24k/yr. Needs: vol surfaces, factor attribution, options builder, API/data-out. This segment pays the top tier happily. 
3. **The Algo Hobbyist.** Runs scripts, wants verified backtesting infrastructure + guarded automation rather than raw API rope to hang themselves with.

---

## 6 · Pricing: beat every competitor where their customers actually bleed

The insight from the competitive audit: headline "zero commission" is universal; the **real** costs are FX conversion, cash drag, and (in Africa) per-trade commissions that make small accounts unviable.

### Africa (vs Bamboo 1.5%/trade + FX spread; Hisa 1% flat, $2 min)

| | **QTSYS** | Bamboo | Hisa |
|---|---|---|---|
| US stock/ETF commission | **$0** | 1.5% ($1 min) | 1% ($2 min <$200) |
| FX conversion (NGN/KES→USD) | **0.5%, transparent** | ~1.5–2% embedded spread | ~1%+ |
| Account minimum | **$1** | ~$10 | ~$10 |
| Fractional shares | Yes | Yes | Yes |
| **AI desk (Copilot, briefings, plan)** | **Basic tier free** | — | — |
| Auto-trader + full desk | $4.99/mo (₦-priced local equiv.) | — | — |

A Nigerian investing $100/month pays Bamboo ≈ $3–3.5 in commission+FX per cycle; pays us ≈ $0.50. **We are 6–7× cheaper on the all-in cost that matters** — and give them an AI desk. Revenue: FX spread (0.5% honest and stated), subscriptions, interest share on idle USD cash, securities lending.

### Europe (vs Trading 212 0.15% FX / T. Republic €1/trade / eToro up to 1.5% FX)

| | **QTSYS** | Trading 212 | Trade Republic | eToro |
|---|---|---|---|---|
| Stock/ETF commission | **€0** | €0 | €1/trade | $1–2/trade |
| FX fee | **0.15% free tier / 0% on Pro** | 0.15% | ~0.1–0.2% | up to 1.5% |
| Interest on cash | **Full pass-through minus 0.25%** | high | 3.5% | low |
| **AI desk** | **Free: briefings+copilot Q&A · Pro €9.99/mo: plan+auto-trader+options/analytics · Desk €29.99/mo: everything incl. L2, factor attribution, API** | — | — | copy-trading only |

We match the cheapest on every commodity line (so no reason *not* to switch) and monetize the thing only we have. Post-PFOF-ban, incumbents must quietly raise FX/cash margins — we publish ours and win the trust news cycle.

**Pricing principles:** (1) never monetize opacity — every fee on one page; (2) the free tier must be genuinely the best free broker in the market (that's the growth engine); (3) subscriptions price the desk, not the pipes.

---

## 7 · Distribution: how we acquire

**Africa**
- **Mobile-money rails as channel:** M-Pesa (KE) and Nigerian agency-banking integrations for instant funding — funding friction is the #1 churn point for competitors; make deposit-to-first-trade under 3 minutes.
- **Creator economy, but accountable:** finance YouTubers/TikTokers in Lagos & Nairobi paid on *funded accounts*, with a compliance-reviewed script library. The voice copilot demo ("ask your account a question, it answers out loud") is natively viral content.
- **Campus + developer motion:** hackathons on our data-out API; ambassador programs at UNILAG, UI, UoN, Strathmore.
- **Employer/payroll partnerships:** salary-deduction USD investing with tech employers and remittance firms.

**Europe**
- **Comparison-site + finance-community seeding:** the personal-finance forums (r/eupersonalfinance, Finanzfluss-adjacent communities, Bogleheads-EU) decide neobroker flows; win the spreadsheet comparisons (we're built to).
- **Product-led growth:** the daily voice briefing is shareable ("my broker talks to me every morning"); referral = one month of Pro per funded referral, both sides.
- **The "PFOF is dead" campaign:** transparent-pricing positioning timed to the MiFIR transition coverage.
- **B2B2C second engine:** our terminal white-labeled to EU independent advisers and asset managers (they get the agent desk + their clients get the brokerage) — one sale = hundreds of funded accounts.

**Both regions:** publish the verification methodology (DSR gating) openly — the "we're the broker that refuses to let the machine trade unproven strategies" story earns regulator goodwill and press that money can't buy.

---

## 8 · Moats (in order of durability)

1. **The verification culture + data flywheel.** Every trade across every customer feeds anonymized strategy-performance data → our DSR gates and scan rankings improve → better plans → better retention. Copyable in form, not in accumulated data.
2. **Voice + local-language copilot.** Swahili, Yoruba, Pidgin, Polish, German voice briefings — deep localization incumbents won't prioritize.
3. **Regulatory footprint.** Each license is 6–15 months of moat.
4. **Cost structure.** No branches, no legacy stack, Alpaca-variable costs — we can sustain free tiers that force incumbents to bleed.
5. **Brand = trust in markets burned by scams.** In Nigeria especially, "the broker whose AI tells you NO" is a category-defining trust position.

---

## 9 · Unit economics (targets, honest ranges)

| Metric | Africa target | Europe target |
|---|---|---|
| CAC (blended, funded account) | $4–8 | €25–45 |
| Year-1 ARPU | $12–20 (FX 0.5% on ~$1.2–2k flow + 6–10% sub attach) | €40–80 (interest share + 12–18% sub attach + FX) |
| Contribution margin | 60%+ (Alpaca per-account costs are low single $) | 65%+ |
| Payback | < 6 months | < 12 months |
| North-star | Monthly funded active accounts × net deposit growth | Same + sub attach rate |

Break-even sketch: ~120–150k funded accounts blended across both regions covers a lean 25-person team + licenses + infra. Aggressive but within the range Bamboo/Hisa demonstrated (Bamboo crossed ~500k registered users years ago on a worse product).

---

## 10 · Roadmap & KPIs

| Phase | Timeline | Milestones | Kill/adjust criteria |
|---|---|---|---|
| **0 — Foundations** | M0–M3 | Nigeria sub-broker filing; Kenya sandbox entry; EU tied-agent LOI; Alpaca Broker API partner agreement; product localization (₦/KSh funding, Pidgin+Swahili copilot alpha) | No Alpaca agreement or no viable NG path → rescope to Kenya-first |
| **1 — Africa beachhead** | M3–M9 | Lagos launch → Nairobi; 25k funded accounts by M6, 100k by M9; deposit-to-trade < 3 min; sub attach ≥ 5% | CAC > $12 sustained or funding-rail failure |
| **2 — Europe entry** | M6–M18 | Tied-agent launch (PL/IE/NL/Baltics) M8; own MiFID filed M6, granted ~M18; 150k EU funded accounts M18; Pro attach ≥ 12% | Attach < 6% at M12 → reprice tiers before scaling spend |
| **3 — Depth** | M12–M24 | UK, Ghana, SA partner, diaspora product, B2B2C adviser white-label; options rollout EU; 500k total funded accounts M24 | — |

---

## 11 · Risks (the ones that actually kill this)

1. **Regulatory reversal in Nigeria/Kenya** (rules on cross-border investing tighten). *Mitigation:* multi-market from day one; local counsel on retainer; never > 50% of accounts in one jurisdiction after M12.
2. **FX/repatriation controls** (naira liquidity). *Mitigation:* partner with licensed FX providers; USD-stablecoin funding rail where legal; conservative treasury.
3. **AI-advice regulatory scrutiny** (EU AI Act + MiFID suitability). *Mitigation:* the desk is architected for this — deterministic gateway, human confirm, DSR verification, full audit logs are literally already built; publish the framework proactively.
4. **Alpaca concentration risk.** *Mitigation:* clean abstraction layer over the broker API (already in the codebase); second-provider due diligence at M12.
5. **Incumbent response** (Trade Republic ships "AI assistant"). *Mitigation:* speed + depth — a chat wrapper is not a verified agent desk; our moat is the gate + data flywheel, and we say so loudly.
6. **Trust incident** (an auto-trader loss goes viral). *Mitigation:* paper-first defaults, explicit user-set guardrails, loss-boundary marketing ("your max loss is prepaid" for options; caps everywhere), and never marketing returns.

---

## 12 · The pitch, in three sentences

Retail investors in Africa are overpaying 1–1.5% a trade for a buy button; retail investors in Europe just watched the PFOF model die and are choosing new brokers. We are launching the first **agentic brokerage** — commission-free access on Alpaca's proven infrastructure, plus an AI desk that plans, verifies, protects, and *talks* — priced free where competitors charge, and subscription-priced where we're the only offer on earth. Access is a commodity; intelligence is the product; we're the only broker selling intelligence.
