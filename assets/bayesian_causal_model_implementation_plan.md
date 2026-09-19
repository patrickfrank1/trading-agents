# Implementation Plan — Quarterly Causal Bayesian Model for the Macro Analyst

Companion to [`bayesian_causal_model.md`](bayesian_causal_model.md). This plan operationalizes
that design under the following constraints agreed for this project:

- **Quarterly granularity** for the model panel; higher-frequency data only as inputs for
  aggregation or (later) shock identification.
- **Free data only.** Paid sources are dropped and replaced by free surrogates (§3.3).
- **Keep complexity low.** Staged V1 → V3 only. Latent-state expansion (V4), regime
  switching (V5) and identified shocks (V6) are explicitly deferred.
- **DAG discipline.** Feedback loops are allowed *across time slices only*, never within a
  slice, and the cross-slice dynamics must be stable (§2.2).
- **Latent vs. structural separation.** Indicators of a latent state are measurement
  equations, not structural parents of other nodes (§2.4).

---

## 1. Scope

Model asset nodes: **Gold, S&P 500, Treasury bonds, REITs** (REITs are the tradable proxy
for real estate — see §2.5). Driver blocks: growth, inflation, monetary policy / rates,
fiscal, credit conditions, external (USD).

Output consumed by the macro analyst: a quarterly forward scenario table —
posterior predictive distribution of asset returns over 1–8 quarters, directional
probabilities, and responses to a small set of scenario interventions (rates path,
inflation shock, growth/productivity shock, risk-off shock).

Not in scope (deferred): physical CRE appraisal data, mortgage-rate-distribution /
lock-in modeling, 6-regime Markov switching, full identified-shock impulse-response
system, paid consensus-forecast data.

---

## 2. Architecture decisions

### 2.1 Model form

A **dynamic Bayesian network** in state-space/VAR(1) form:

```
X_t = A · X_{t-1} + B · U_t + ε_t,      Y_t = C · X_t + D · U_t + η_t
```

- `X`: endogenous state (macro blocks + asset returns + a small number of latent states).
- `U`: observed exogenous/identified drivers (shock proxies, policy path, oil, USD).
- `Y`: observed indicators (measurement layer for latents).

Estimated in **PyMC** with Student-t innovations, shrinkage priors (regularized
horseshoe or horseshoe) on off-diagonal coefficients, and sign-informed priors where the
economics is unambiguous (per §23 of the design doc). V1 and V2 are small enough to be
estimated as Bayesian regressions with AR(1) terms; the full joint system arrives only
in V3.

### 2.2 Feedback loops: allowed, but only across time slices

**Rule (best practice for DBNs / dynamic causal models):**

1. **Within a quarter: strictly acyclic.** Every time slice is a DAG with a fixed
   topological order. Cycles within a slice make the joint factorization undefined and
   are the classic mistake when porting a "system" diagram to a Bayesian network.
2. **Feedback across quarters: yes — this is the standard way to represent loops.**
   `SPX → wealth → consumption` is not an arrow inside quarter *t*; it is
   `SPX_{t-1} → Consumption_t`. All apparent loops in the design doc
   (§1, §12, §13, §22) are implemented as **lagged edges only**.
3. **Stability is mandatory.** Because lagged feedback closes loops through time, the
   transition matrix `A` must have all eigenvalues inside the unit circle
   (spectral radius ρ(A) < 1), otherwise posterior simulations explode and scenario
   analysis is meaningless. Enforcement, in order of preference:
   - **Shrinkage + rejection:** Normal/horseshoe priors with rejection (resample) of
     posterior draws violating ρ(A) < 0.99. Cheap and sufficient for small `A`
     (≤ ~10 variables), i.e., V1–V3.
   - **Stationarity-enforcing parameterization** (VAR(1) via the partial-autocorrelation
     matrix, à la Heaps 2023): only if V3's `A` turns out too large for rejection to be
     efficient.
   - **Posterior check regardless:** report max |eigenvalue| and verify impulse responses
     decay; treat violation as a failed validation, not a tuning knob.
4. **Within-quarter simultaneity is reserved for V6.** When identified shocks arrive
   (§5), simultaneity is resolved by shock ordering (identification), still without
   within-slice cycles.

### 2.3 Fixed topological order within a quarter

```
Exogenous shocks (oil, USD, policy path, identified shocks)
  → Macro fundamentals (growth, inflation, output gap, fiscal)
    → Policy & rates (expected policy path, real yield, term premium)
      → Credit & financial conditions (latents)
        → Asset returns (bonds, SPX, REITs, gold)
```

Assets affect macro fundamentals **only through lagged edges** (wealth, collateral,
financial-accelerator channels, §13/§16 of the design doc).

### 2.4 Latent states: measurement ≠ structure

The design doc lists latent nodes whose "drivers" are market observables (e.g., ERP
driven by VIX and credit spreads). Those observables are **outcomes** — using them as
structural parents leaks reverse causality (flag from the review). Implementation rule:

- Each latent gets **structural parents** (fundamentals/shocks only) and a
  **measurement equation** onto its indicators.
- Indicators never appear as parents of other structural nodes; they inform the latent,
  and the latent propagates.
- V1–V3 latent budget (small, per "keep complexity low"):
  | Latent | Structural parents (quarterly) | Measurement indicators |
  |---|---|---|
  | Expected earnings growth | real growth, inflation, unit labor costs | corporate profits, SPX earnings yield |
  | Credit conditions | growth, policy path, banking delinquencies | IG/HY OAS, SLOOS, loan growth |
  | Financial conditions | real yield, USD, credit conditions, lagged SPX | NFCI (+ subindexes) |
  | Structural gold demand | AR(1) persistence, (V2+: CB purchases) | residual gold-demand signal, GLD-based proxy |
  | Fiscal credibility (V3) | debt/GDP trajectory, primary balance | term premium proxy, inflation expectations |

  ERP in V1–V3 is **not** a free latent: it is the residual priced into the SPX
  measurement equation (earnings yield − real yield), monitored but not modeled
  separately until V3.

### 2.5 Real estate: REIT node, corrected cap-rate equation

- Asset node = **REITs** (VNQ total return, quarterly). Physical-market variables
  (mortgage rates, Case-Shiller, permits, starts) are **drivers**, not asset nodes.
- The design doc's cap-rate equation omitted expected growth. Use the Gordon-consistent
  form:
  `cap rate ≈ risk-free + real-estate risk premium − E[NOI growth]`
  so the growth channel is not double-counted and CRE/REIT rate sensitivity is not
  overstated. REIT valuation node: price-to-FFO proxy via dividend/earnings yield spread
  vs. 10Y real yield.
- Mortgage-lock-in is **not modeled explicitly** (no free distributional data); its
  effect is absorbed by distributed lags on the mortgage-rate term.

### 2.6 Fiscal system: chosen direction (easier to model)

- Structural arrows: **fiscal fundamentals → fiscal credibility (latent) → term premium
  / inflation expectations**.
- The reverse link (market yields → government financing cost → debt dynamics) enters
  **only through the lagged average effective interest cost** `r_eff_{t-1}` in the debt
  accumulation equation:
  `Debt_t = ((1 + r_eff_{t-1}) / (1 + g_t)) · Debt_{t-1} − PrimaryBalance_t + SFA_t`
  where `SFA` is an explicit stock-flow-adjustment term (not white noise; it is large in
  practice). `r_eff` is computed from interest expense / prior-period debt stock —
  *not* the marginal market yield.
- This keeps the within-quarter slice acyclic (§2.2) and requires only free data
  (debt, receipts, outlays, interest expense).

### 2.7 Regimes: deferred

No regime switching in V1–V3. If added later: **2 regimes max** (normal / stress) with
switching in **innovation volatility only** (coefficients pooled). The 6-regime design
in §24 of the doc is not estimable at quarterly frequency (crisis regimes have ~2–4
quarters of data in 40 years).

---

## 3. Data plan

### 3.1 New dataflow: `tradingagents/dataflows/macro_timeseries.py`

One module that builds and caches the quarterly model panel. Responsibilities:

1. **Full-history fetch** (unlike current snapshot tools):
   - FRED via `fredapi` — full observation history per series (FRED_API_KEY required;
     reuse the existing `macro_vendors/cache.py` + rate-limit helpers).
   - yfinance — full history (max period) for asset tickers, resampled to quarterly.
   - Free non-API downloads (CSV/Excel over HTTPS) for: NY Fed ACM term premium,
     GPR index, WGC central-bank gold demand, CFTC COT (optional, V3).
2. **Quarterly aggregation rules:**
   | Series type | Rule |
   |---|---|
   | Asset prices (gold, SPY, VNQ, TLT/IEF, DXY, oil) | end-of-quarter price → **quarterly total return** |
   | Yields / rates (10Y, TIPS, mortgage, fed funds) | quarterly **average**; ΔYield for return equations uses end-vs-end |
   | Macro levels (CPI index, debt stock, Fed balance sheet) | end-of-quarter level |
   | Macro flows (payrolls, receipts, outlays, profits) | quarterly sum (or FRED's native quarterly series) |
   | Sentiment/standards surveys (UMich, SLOOS) | quarterly average of monthly readings |
3. **Alignment & cleaning:** union-of-quarters index, explicit missing-mask (the model
   handles missingness natively in PyMC), outlier clamp at ±8σ for spurious prints,
   unit sanity assertions (percent vs. index level — the class of bug already hit by
   the OECD vendor).
4. **Look-ahead guard (critical for a trading tool):** align each series by
   **publication lag**, not observation date. Pragmatic per-series lag table
   (CPI/PCE +1 month, payroll +1 month, GDP advance +1 month, corporate profits +2
   months, SLOOS +~3 weeks, Case-Shiller +2 months, debt/fiscal +~1 quarter …).
   Real-time ALFRED vintages are the gold standard — optional later, the lag table is
   the V1 compromise.
5. **Storage:** versioned **Parquet** panel under the existing `data_cache_dir`
   (`data_cache_dir/macro_panel/v<date>_<git_sha or config hash>.parquet` + a small
   JSON manifest: series IDs, vintage, row counts, last observation dates). Rebuild is
   explicit (script/CLI), not implicit per analyst call.

### 3.2 Series inventory (all free)

**FRED (API, full history)** — IDs marked ⚠︎ to be confirmed via FRED search at
implementation time; all others are well-known IDs.

| Block | Series (ID) |
|---|---|
| Policy & rates | FEDFUNDS, DFEDTARU, DGS3MO, DGS2, DGS10, DGS30, T10Y2Y, **DFII10** (10Y TIPS real), **T10YIE**, **T5YIE**, MORTGAGE30US, WALCL (Fed balance sheet / QE-QT) |
| Inflation | CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, MICH, UMCSENT, CES0500000003 (wages), **ULCNFB** (unit labor costs), PPIACO |
| Growth | GDPC1, GDPPOT (→ output gap = GDPC1−GDPPOT), **OPHNFB** (productivity), CIVPART, UNRATE, PAYEMS, INDPRO, GPDIC1 (real fixed investment ⚠︎), **CP** (corporate profits after tax) |
| Fiscal | **GFDEGDQ188S** (debt % GDP), GFDEBTN (gross debt level), **FGRECPT** (receipts), **FGEXPND** (expenditures), federal interest payments (BEA 3.1 ⚠︎ confirm ID) |
| Housing | HOUST, HOUST1F, PERMIT, **CSUSHPINSA** (Case-Shiller), MSPUS, rent (CPI rent component ⚠︎), HHMSDODNS (household mortgage debt ⚠︎) |
| Credit & financial conditions | **NFCI** (+ NFCICREDIT, NFCILEVERAGE, NFCIRISK), **BAMLC0A0CM** (IG OAS), **BAMLH0A0HYM2** (HY OAS), **DRTSCILM** (SLOOS tightening ⚠︎), TOTLL / TOTBKCR (bank loans), DRALACBS / DRBLACBS / CRE delinquency (⚠︎ IDs), TOTALSL |
| Banking (V3) | bank failures series (⚠︎), delinquencies above |
| External | DTWEXBGS (broad USD, daily → quarterly), current-account (World Bank annual, existing vendor) |
| Household (V3) | TNWBSHNO (household net worth ⚠︎) |

**yfinance (tickers, full history):** GC=F (gold), SPY (equities), ^TNX/^TYX/^IRX
(cross-check yields), **TLT or IEF** (Treasury bond return proxy), VNQ (REITs),
XHB/ITB (housing equity cross-check), KRE (bank equity, V3), HYG/LQD (credit ETF
cross-check for OAS), DX-Y.NYB (USD), CL=F (oil), ^VIX (daily → quarterly vol).

**Free non-API downloads (plain HTTPS, no key):**
| Source | Content | Use |
|---|---|---|
| NY Fed ACM (CSV) | 10Y term premium | term-premium node (preferred over T10Y2Y proxy) |
| Caldara–Iacoviello GPR (CSV) | Geopolitical Risk index | gold risk channel |
| World Gold Council (quarterly file) | central-bank net purchases | structural gold demand (V2+) |
| CFTC COT (CSV) | managed-money gold positioning | optional, V3 |
| Zillow/OFR free pages | optional rent / CRE cross-checks | not required for V1–V3 |

**Existing vendors reused:** OECD (extend `OECD_SERIES` to EZ/JP/CN/GB quarterly GDP +
CPI — currently US-only), World Bank (annual external/debt context), ECB (Eurozone
context). These stay snapshot-style report tools for the analyst; the model uses only
the panel.

### 3.3 Paid sources → omitted or surrogated

| Omitted (paid/unavailable) | Decision |
|---|---|
| Consensus forecasts (CPI/FOMC surprises) | Surrogate: **market-implied surprises** — ΔT5YIE on CPI release days, ΔDGS2 on FOMC days (needs daily data; V6 only) |
| Sell-side forward EPS estimates | Surrogate: **FRED corporate profits (CP)** + SPY trailing earnings yield (yfinance) |
| NCREIF cap rates / NOI | Surrogate: corrected cap-rate identity (§2.5) + REIT yields |
| Fund-flow / allocation databases | Omit; fold into structural gold demand residual (V1) / asset-allocation latent (deferred) |
| SGE–LBMA Asian premium | Omit; folded into structural gold demand latent |
| Mortgage-rate distribution / NMDB | Omit explicit node; distributed lags on MORTGAGE30US (§2.5) |
| FRA-OIS / TED funding spreads | Surrogate: **NFCI + NFCILEVERAGE** |
| Import price index (granular) | Surrogate: PPIACO + WTI oil |

### 3.4 Derived series (computed in the panel builder)

- Real yield: `DFII10` (fallback: `DGS10 − T10YIE`).
- Breakeven / inflation expectations: `T10YIE`, `MICH`.
- Output gap: `GDPC1` vs `GDPPOT` (log gap), or CBO-based detrend.
- Earnings yield (SPX): 1 / trailing P/E from SPY; earnings-growth proxy from `CP`.
- ERP proxy: earnings yield − real yield (monitored, see §2.4).
- Treasury bond return: from IEF/TLT total return (preferred over duration-approximation;
  duration formula kept as cross-check).
- Effective interest cost `r_eff`: interest expense / prior debt stock.
- Primary balance: `(FGRECPT − FGEXPND) / GDP`.
- Fiscal impulse: Δ primary balance, GDP-scaled.
- Inflation (q/q annualized) for CPI and core PCE; wage growth q/q.

---

## 4. Model versions (V1–V3; V4–V6 deferred)

### V1 — Gold baseline (quarterly)

```
gold_q_return_t = α + β1·Δreal_yield_t + β2·ΔUSD_t + β3·infl_surprise_t
                  + β4·NFCI_t + β5·GPR_t + ρ·gold_q_return_{t-1} + ε_t,   ε ~ Student-t
```

- Sign-informed priors: `β1 < 0`, `β2 < 0`, `β3 > 0`, `β4 > 0` (NFCI: higher = tighter),
  `β5 > 0`; `ρ ∈ (−1, 1)` with shrinkage.
- `infl_surprise_t`: realized CPI/PCE minus trailing AR forecast (free surrogate for
  consensus surprise).
- Structural gold demand enters as an AR(1) residual latent only if diagnostics demand
  it — otherwise V1 ships without latents (keep complexity low).
- **Feasible immediately** after M1/M2 (§6): every input is in the panel.

### V2 — Cross-asset layer

Adds joint nodes: Treasury bond return (IEF total return + Δyield cross-check),
SPX quarterly return, REIT quarterly return, plus the **credit conditions** and
**financial conditions** latents (§2.4). Cross-slice feedback: lagged SPX/REIT into
financial conditions and growth (wealth channel) — stability constraints active
(§2.2). Shrinkage priors throughout; ≤ ~8 endogenous variables.

### V3 — Cash-flow / discount-rate decomposition + housing & fiscal blocks

- SPX return split into: cash-flow channel (expected earnings growth latent ← real
  growth, inflation, ULC), discount-rate channel (real yield), financing channel
  (interest expense proxy), residual (monitored ERP proxy).
- Housing block: REIT/HPI return ← mortgage rate (distributed lags), income growth,
  supply (permits/starts), credit conditions.
- Fiscal block: debt dynamics with `r_eff` and SFA (§2.6), fiscal credibility latent →
  term premium (ACM) / inflation expectations.
- ERP becomes an explicit latent with measurement onto the ERP proxy.
- Estimated jointly; stability-enforced; full posterior validation suite (§7).

### Deferred (documented, not built now)

- **V4:** remaining Tier-1/2 latents (banking stress, funding liquidity, household rate
  exposure, asset-allocation state).
- **V5:** 2-regime volatility switching (§2.7).
- **V6:** identified shocks — FOMC-day ΔDGS2, CPI-day ΔT5YIE (market-implied surprise
  surrogates, §3.3); local impulse responses; only then upgrade the "causal" label.

---

## 5. Analyst tool integration

- **Fitting is offline.** A script (`scripts/fit_macro_model.py`, CLI-able) refits on the
  latest panel and stores the posterior (NetCDF) next to the panel manifest. Not run
  inside the agent loop.
- **New tool for the macro analyst:** `get_macro_causal_forecast(horizon_quarters=4,
  scenario="baseline|hawkish|inflation_shock|risk_off|productivity_boom")` in
  `agents/utils/macro_data_tools.py`:
  1. loads latest panel + cached posterior,
  2. runs posterior-predictive forward simulation (8–12 quarters, scenario intervention
     applied where the scenario pins an exogenous path),
  3. renders a markdown scenario table: per asset (gold, SPX, treasuries, REITs) —
     median cumulative return, 25–75% interval, P(positive), plus channel attribution
     (cash-flow vs discount-rate contribution for SPX).
- Register the tool in `create_macro_analyst` (macro_analyst.py:43) with prompt
  guidance: use for cross-asset scenario framing; conditional-proposal scope rules
  unchanged.
- Caching consistent with existing 7-day snapshot tools; refit cadence decoupled
  (quarterly panel → refit after each FRED quarterly publication or on demand).

---

## 6. Work breakdown / milestones

| Milestone | Deliverables | Tests |
|---|---|---|
| **M1 — Panel builder** | `dataflows/macro_timeseries.py`: FRED full-history fetch, yfinance quarterly resampling, aggregation rules (§3.1), Parquet+manifest cache | unit: aggregation rules, alignment, missing-mask, unit assertions; smoke: fetch with key |
| **M2 — Derived series + publication-lag guard** | §3.4 derivations, lag table (§3.1.4), vintage manifest | unit: r_eff, primary balance, output gap; look-ahead regression test (model inputs at t contain nothing published after t) |
| **M3 — V1 gold model** | `tradingagents/models/macro_bayes/v1_gold.py` (PyMC), fitting script, priors config | unit: prior sign constraints, posterior-sampling smoke (mock/fake data); integration (optional, real fit, marked) |
| **M4 — Validation harness** | rolling-origin OOS: log predictive density vs random walk / AR / `gold ~ USD + real yield` baselines; directional accuracy; interval coverage; parameter stability | unit: metric implementations; the harness doubles as V1 acceptance gate |
| **M5 — V2 cross-asset** | joint model + stability constraint (rejection on ρ(A) < 0.99) + wealth-channel lagged edges | unit: spectral-radius check; simulation stability test (12-quarter forward sim stays bounded) |
| **M6 — V3 decomposition + fiscal/housing blocks** | full V3, scenario interventions | integration: fit on panel, impulse responses decay, sign-consistency checks (§28 doc) |
| **M7 — Analyst tool + docs** | `get_macro_causal_forecast`, registration, prompt text, README/AGENTS.md updates | unit: report rendering from fake posterior; tool-health check entry |
| **M8 — (later) V4–V6** | regimes, identified shocks | — |

New dependencies: `pymc`, `arviz`, `pyarrow` (pandas Parquet). `fredapi` already used.
Add as extras (e.g., `[model]` extra) so the base install stays light.

Config additions (`default_config.py`): `macro_panel_cache_dir`, panel versioning flag,
refit cadence, V1 prior file path.

---

## 7. Validation & acceptance gates

Per §28 of the design doc, judged **quarterly** out-of-sample (rolling origin, last
~8 years as successive holdouts):

- Log predictive density (primary), RMSE/MAE, directional accuracy, 50%/80% interval
  coverage & calibration, crisis-quarter performance.
- **Acceptance gates:**
  - V1: beats random walk *and* `gold ~ USD + Δreal yield` on log predictive density
    out-of-sample; 80% intervals achieve ≈80% coverage.
  - V2/V3: additionally — impulse responses decay (stability), sign-consistency of
    sign-prior'd coefficients in ≥95% of posterior mass, no look-ahead (M2 test).
- Scenario consistency review (the design doc's §29 scenarios) is a qualitative gate:
  outputs must reproduce the doc's sign logic (e.g., productivity shock → real yield ↑
  but SPX/property can rise; funding-stress shock → gold can fall initially).

---

## 8. Risks & open questions

1. **Quarterly n ≈ 160** (40 years). V2/V3 must stay ≤ ~10 endogenous variables;
   shrinkage priors are not optional.
2. **Revision noise:** GDP, profits, fiscal flows are revised; the publication-lag table
   mitigates look-ahead but not revisions. ALFRED vintages are the later upgrade.
3. **Surrogate fidelity:** market-implied surprises (V6) and corporate-profits-as-earnings
   are approximations; document them in the tool's output so the analyst LLM doesn't
   over-read precision.
4. **Non-API scrapes** (NY Fed, WGC, GPR) can break silently — pin URLs, add schema
   assertions, and make these series optional-with-warning rather than hard failures.
5. **FRED key:** panel build requires FRED_API_KEY (free); the analyst tool must degrade
   gracefully (return "model not available" message), mirroring existing vendor tooling.
6. **Open question:** bond-return proxy — IEF total return (simple, liquid) vs
   duration-approximation from DGS10 (cleaner link to the yield node). Default: IEF with
   duration cross-check; revisit if the two diverge materially in the panel.
