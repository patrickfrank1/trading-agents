"""Sector Specialist: a dynamically-allocated analyst that applies sector-specific
domain knowledge (value drivers, industry structure, regulation, cyclicality, KPIs,
valuation norms) on top of the generic analyst team. The playbook is selected at
runtime from the instrument's actual sector / quote type, so the same agent covers
pharma, semiconductors, utilities, consumer staples, ETFs, and everything else
without sector-specific code paths elsewhere in the graph.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_report_hygiene_instruction,
    resolve_instrument_profile,
    get_company_profile,
    get_sector_performance,
    get_peer_comparison,
    get_10k_filing,
    get_20f_filing,
    get_earnings_call_transcripts,
    web_search,
    WEB_SEARCH_INSTRUCTION,
)


# Sector playbooks. Keys are normalized yfinance GICS sector names (lowercase),
# plus the special keys "etf" and "general". Each playbook tells the analyst
# what actually drives value and risk in that sector so the report is not a
# generic template with the sector name swapped in. Each playbook covers six
# layers of expert sector analysis:
#   1. Industry structure — where profit pools sit, who holds bargaining power,
#      what the real barriers to entry are (Porter-style, but sector-applied).
#   2. Value drivers — what actually compounds per-share value here.
#   3. KPI checklist — the metrics specialist investors verify in filings/calls.
#   4. Cycle & rotation profile — how the sector trades through the business
#      cycle and rate cycle, so timing context is never confused with quality.
#   5. Red flags — the classic ways this sector permanently impairs capital.
#   6. Valuation norms — which methodology fits the cash-flow profile and what
#      "normal" looks like across a full cycle.
SECTOR_PLAYBOOKS = {
    "healthcare": """Sector: HEALTHCARE (pharma / biotech / medtech / tools & services).
Industry structure: the "customer" is rarely the patient — payers (insurers, PBMs, national health systems) control pricing and access, so pricing power sits with whoever controls formularies, reimbursement codes, or hard-to-replicate biology. Patent exclusivity is the moat: after expiry, generic/biosimilar entry typically erodes 70-90% of branded volume within 1-2 years. Distribution is concentrated (three wholesalers move >90% of US drug volume; a few PBMs steer most formularies).
Value drivers: patent-cliff runway per top product (small-molecule expiry dates are public; biologics face biosimilar timing instead); pipeline probability-weighted value (historical base rates: a molecule entering Phase 1 has roughly 7-10% approval odds, Phase 3 roughly 30-50% depending on indication — discount accordingly, never at full value); revenue/profit concentration in the top 1-3 products; gross-to-net dynamics (rebates inflate gross sales); medtech: procedure volumes, adoption curves, reimbursement codes; services/tools: bookings, backlog, capital-equipment cyclicality.
KPIs to check: product-level revenue splits; % of revenue from products losing exclusivity within 5 years; R&D productivity (approvals per $B spent); gross-to-net trend; prescription volume vs price realization; for pre-revenue biotech: cash runway vs burn rate; for medtech: procedure growth and share.
Red flags: single-molecule concentration with expiry inside the holding horizon; pipeline failure on the lead asset; FDA/EMA complete-response letters or clinical holds; pricing legislation (IRA-style Medicare negotiation) aimed at the flagship product; IP litigation (ANDA paragraph IV challenges); wholesaler inventory build channel-stuffing (check wholesaler days); serial restructuring that cuts R&D to prop up EPS.
Cycle: pharma/medtech are defensive; biotech is a funding-cycle business — pre-revenue companies depend on capital-markets access, so the group is rate-sensitive. Healthcare typically rotates into favor late-cycle and early in bear markets.
Valuation norms: mature pharma trades 10-15x earnings; value the pipeline with risk-adjusted rNPV (probability-weighted), not full DCF credit; pre-profit biotech on EV vs cash minus burn, cross-checked against probability-weighted pipeline value; medtech earns a premium P/E only if procedure volumes are recurring. Judge cheapness against the patent-protected cash-flow runway, never trailing P/E alone.""",
    "technology": """Sector: TECHNOLOGY (software / internet / hardware / IT services).
Industry structure: value concentrates at the top — software moats are switching costs and workflow lock-in; platform moats are network effects and ecosystems; hardware moats are scale, supply-chain position, and product cycles; IT services is largely a low-moat labor business (judge it that way, not as a software multiple). Near-zero marginal distribution cost creates winner-take-most dynamics, but low switching costs let inferior rivals destroy pricing.
Value drivers: net revenue retention (>110% = the product sells itself into the base; <100% = every year must be re-earned); gross-margin structure (SaaS 70-80%+, infrastructure/hardware lower); FCF conversion — how much accrual profit becomes cash; SBC dilution vs buyback offset; TAM penetration and expansion (land-and-expand); product-leadership runway; for hardware: component-cost curves and product-cycle timing.
KPIs to check: revenue growth and its deceleration rate; NRR/GRR; customer concentration; cash conversion (FCF vs net income); SBC as % of revenue and FCF/share vs EPS growth; Rule of 40 (growth % + FCF margin %) for software; remaining performance obligations/backlog; days sales outstanding.
Red flags: growth bought by stretching payment terms or channel stuffing; SBC-heavy compensation masking weak per-share economics; dependence on one platform or channel (app stores, one hyperscaler, one ad channel); an unproven AI/product pivot priced as certain; deferred-revenue or NRR declines preceding revenue declines; serial guidance misses with "one-time" explanations.
Cycle: partly cyclical (enterprise IT budgets, device refresh cycles); as long-duration assets, growth tech is rate-sensitive — it rotates with liquidity. AI infrastructure capex is a live buildout whose payback at the application layer is unproven; distinguish suppliers capturing spend from buyers destroying returns.
Valuation norms: judge on EV/FCF and growth-adjusted multiples, not headline P/E; decompose the multiple into assumed growth duration + margin expansion + terminal share and identify which assumption is most fragile. Deep drawdowns in quality names are usually multiple normalization, not business impairment — state explicitly which one is happening.""",
    "semiconductors": """Sector: SEMICONDUCTORS / SEMICONDUCTOR EQUIPMENT.
Industry structure: extreme specialization with chokepoints — design (fabless), manufacturing (foundries; only ~3 credible leading-edge players), equipment (near-monopolies in lithography), memory (commodity-like, brutal cycles), EDA/IP (oligopoly). Customer concentration is structural: a handful of foundries, OSATs, and hyperscalers drive most revenue. Export controls (BIS rules on China) are a recurring, live policy lever.
Value drivers: process-node leadership and design wins (each socket win compounds for years); utilization and pricing in memory; equipment backlog and order visibility; AI/datacenter demand vs mature smartphone/PC markets; geographic exposure (Taiwan concentration = geopolitical tail risk; China revenue = policy risk).
KPIs to check: book-to-bill; utilization rates; inventory days at every chain level (chipmaker, distributor, customer — the bullwhip effect turns small demand changes into large inventory signals); customer capex guidance (hyperscaler announcements lead equipment orders by quarters); design-win pipeline; gross margin vs model; China revenue %.
Red flags: peak-cycle multiples on peak earnings (the sector's classic error); customer capex digestion after a boom; inventory build while demand flattens; export-control expansion hitting revenue-critical products; design losses at anchor customers; memory price collapse.
Cycle: the deepest cyclicality in equities — 2-4 year silicon cycles ride multi-year secular waves. Never extrapolate a boom quarter; never capitulate at a trough.
Valuation norms: trailing P/E is most misleading exactly at cycle extremes (lowest P/E at peak earnings, highest at trough). Normalize earnings: mid-cycle P/E on through-cycle EPS; EV/sales with positive EBITDA for trough memory names; replacement-value reasoning for equipment leaders with monopoly-like positions.""",
    "utilities": """Sector: UTILITIES (regulated / IPP / water / grid infrastructure).
Industry structure: a government-sanctioned franchise — returns are set by regulators, so the real analysis is political economy: how constructive is the jurisdiction, how lagged are rate cases, what ROE is allowed vs earned. The live structural debate is data-center/AI and electrification load growth justifying rate-base expansion at unprecedented scale.
Value drivers: rate-base growth (the compounding engine) × allowed ROE; load growth from data centers and electrification (signed electric-supply agreements/PPAs are credible evidence; MOUs are not); fuel mix and hedging; constructive vs adversarial regulatory relationships and rate-case track record; capital plan self-funding vs dilution.
KPIs to check: rate-base growth guidance vs historical delivery; allowed vs earned ROE spread; regulatory lag; equity-issuance needs (ATM programs = dilution in disguise); FFO/debt credit trajectory; pension/OPEB status; % of capex backed by signed contracts vs speculation.
Red flags: capex plans predicated on hoped-for friendly regulators; merchant exposure dressed as regulated; dividend payout + heavy capex = structural equity dilution; wildfire, stranded-asset (nuclear decommissioning, coal), and storm liabilities; data-center demand built on unsigned letters of intent.
Cycle: bond proxies — the entire sector trades off interest rates; watch the 10-year Treasury and the utility dividend-yield spread over it. Defensives lag early-cycle risk-on rallies and rotate back in late-cycle/late-recovery.
Valuation norms: dividend yield spread vs the 10-year; P/E vs its own historical premium/discount to the sector; rate-base-growth-adjusted P/E (a 6-8% grower deserves more than a 3% grower — but only if the capex is accretive, not dilutive). Separate rate-driven derating (sector-wide, mean-reverting) from regulatory deterioration (company-specific and lasting).""",
    "energy": """Sector: ENERGY (E&P / integrated / oilfield services / midstream / refining).
Industry structure: a price-taker commodity business; the only durable levers are cost-curve position, reserve quality, capital discipline, and balance sheet. Demand is global and inelastic short-term; supply responds with multi-year lags, which is what creates the cycle. OPEC+ spare capacity sets the effective price floor/ceiling; US shale is the swing supplier.
Value drivers: corporate breakeven vs strip (which WTI/Brent makes FCF positive after the dividend?); FCF per share at strip prices; reserve replacement ratio and reserve life; decline rates (shale base declines ~30%/yr — a perpetual drilling treadmill is not "cheap"); hedging book; credibility of the capital-return framework (mechanical buybacks beat discretionary promises).
KPIs to check: FCF at strip prices; production per share (dilution can consume all growth); net debt/EBITDAX through the cycle; decline rates; cost per boe vs peers; hedging %; reserve-writedown history.
Red flags: reserve writedowns (core assets over-promised); acquisitions at cycle peaks (the sector's classic capital destroyer); payout promises that break below mid-cycle prices; well-cost inflation; RBL covenant pressure; dividends funded by debt at trough prices.
Cycle: historically a late-cycle leader and early-recovery laggard; refining runs its own mini-cycle on crack spreads; oilfield services ride upstream capex with a lag; midstream is contract/fee-based and less commodity-levered.
Valuation norms: value is normalized FCF yield across the cycle (e.g., FCF at $60-70 Brent), not spot earnings; P/NAV with a discount for execution and jurisdiction risk; midstream judged on distribution coverage and leverage, not oil-price beta. The assumed commodity price deck is where all valuation error lives — state your deck explicitly and test the payout against it.""",
    "financial services": """Sector: FINANCIALS (banks / insurers / asset managers / exchanges & market infrastructure).
Industry structure: leverage IS the business model, so equity is a thin call option on asset quality — underwriting and credit discipline through cycles is everything. The sub-industries are entirely different businesses; do not average them.
- Banks: value = deposit franchise (how cheap and sticky are the deposits?) × asset discipline. Watch NIM and deposit betas across rate shifts, loan mix (CRE/office is the classic landmine), CET1 trajectory, allowance vs charge-offs, HTM unrealized losses (duration risk — the SVB lesson), and whether earnings come from reserve releases.
- Insurers: value = underwriting discipline (combined ratio <100% through the cycle) + float cost (negative-cost float is a compounding machine). Watch prior-year reserve development (adverse development = chronic under-reserving), catastrophe exposure and reinsurance structure, and pricing-cycle position (hard vs soft market).
- Asset managers: value = organic net flows + fee mix. Watch fee compression, the passive mix shift, performance-fee dependence, and whether growth is bought with M&A rather than earned.
- Exchanges/processors: value = secular volume growth on near-monopoly rails; watch interchange regulation and take-rate drift.
KPIs to check: ROE vs cost of equity; NIM trend; credit-loss provisions and charge-offs; reserve ratios; book-value-per-share trajectory; payout sustainability.
Red flags: earnings propped by reserve releases; rapid loan growth late-cycle; duration mismatch in securities books; adverse insurance reserve development; conglomerate complexity hiding losses (demand segment-level data); deposit-flight sensitivity.
Cycle: banks lead early recoveries (steep curve + credit normalization) and break late-cycle; insurers are pricing-cycle businesses partly independent of the economy; asset managers track markets with flow beta.
Valuation norms: P/TBV vs ROE is the primary lens (justified P/B ≈ (ROE − g)/(COE − g)); dividend-discount/residual-income models fit far better than DCF-on-FCF; life insurers on price/embedded value; exchanges on mid-cycle P/E. Never call a bank earning below its cost of equity "cheap" without arguing the specific ROE-recovery path.""",
    "consumer defensive": """Sector: CONSUMER STAPLES.
Industry structure: mature oligopolistic categories where growth comes from share gains, pricing, and premiumization — not category expansion. The central power struggle is brand pricing power vs retailer power: as retail consolidated (mass, club, hard discounters, Amazon), bargaining power shifted to the shelf. Private label is the permanent competitive threat at every price tier.
Value drivers: brand pricing power (can it take price without permanent volume loss?); volume/price/mix decomposition of growth (price-led growth with negative volumes is a franchise in decay — often a multi-year value trap); distribution and shelf-space moats; EM route-to-market; productivity programs funding reinvestment rather than just margin.
KPIs to check: organic volume growth (the single most important line); price/mix vs volume split; household penetration and repeat rates; market-share trends; gross-margin recovery after input inflation; retailer concentration; private-label share in the category (trade-down indicator).
Red flags: volume declines masked by pricing; brand-equity erosion (consumers trade down and never return); structural demand shifts (GLP-1-type effects, category disruption — energy drinks vs soda); EM FX and expropriation; serial restructuring programs that never restore growth; gross-margin "recovery" funded by cutting brand investment.
Cycle: defensive/contrarian — outperforms in bear markets, lags early-cycle risk-on recoveries. Within the sector, commodity input cycles (coffee, cocoa, packaging, freight) squeeze or release gross margin with a pricing lag.
Valuation norms: low-growth, high-visibility cash flows — P/E vs its own 5-10-year history, FCF yield, dividend coverage, EV/EBITDA. The persistent quality premium evaporates when volume growth stays negative for years — verify the premium is still earned, not inherited.""",
    "consumer cyclical": """Sector: CONSUMER DISCRETIONARY.
Industry structure: highly cyclical demand plus extreme operator heterogeneity — the gap between great operators (unit-level ROI, culture, supply chain) and mediocre ones widens violently in downturns. E-commerce and off-price structurally took share from mall retail; housing-linked durables are interest-rate businesses; autos combine cyclicality, huge fixed costs, and EV-transition capital risk.
Value drivers: four-wall unit economics and new-unit payback (growth must be ROI-positive at the store level, not just revenue-additive); same-store sales vs new-unit growth; brand loyalty and pricing power; scale advantages in sourcing/logistics; digital mix and loyalty economics.
KPIs to check: SSS growth; unit growth and its cash-on-cash returns/payback; inventory vs sales (the earliest demand-crack indicator); gross-margin trajectory; promotional intensity; customer credit exposure (private-label cards, BNPL); digital penetration; order books for housing-linked durables.
Red flags: growth bought with declining unit economics; inventory build ahead of demand cracks; reliance on consumer-credit expansion; luxury/aspirational brand stretching (discounting permanently damages equity); founder/key-man departures; EV-transition capex outrunning returns (autos).
Cycle: the archetypal early-cycle sector — leads out of recessions; trailing P/E bottoms exactly at the earnings peak, so low multiples at cycle tops are a trap. Housing-linked names are rate-sensitive; watch the savings rate and credit delinquencies as leading indicators.
Valuation norms: mid-cycle EBIT margins and mid-cycle EPS, never trailing; EV/EBITDA and P/E on normalized earnings; balance-sheet resilience matters more than the multiple at cycle peaks. Do not anchor on the low P/E that appears precisely at the top.""",
    "industrials": """Sector: INDUSTRIALS (aerospace & defense / machinery / transport / electrical equipment / multi-industry).
Industry structure: long-cycle backlog businesses with oligopoly niches — aerospace OEM and aftermarket are near-duopolies at the top, defense primes are tied to government budgets, machinery rides global capex. Aftermarket/services (spares, maintenance, long-term agreements) is the hidden gem: recurring, high-margin, less cyclical. Multi-industry conglomerates are sum-of-the-parts stories — judge each piece, not the blend.
Value drivers: backlog quality and conversion timing; aftermarket/services revenue share and margin; contract structure and pricing (fixed-price vs cost-plus); program execution; industrial capex cycle exposure; policy tailwinds (infrastructure, reshoring, defense budgets).
KPIs to check: book-to-bill; backlog coverage (years of revenue); program margins on fixed-price contracts; organic vs acquired growth; working-capital swings (aerospace inventory build can swallow a decade of margin gains); FCF conversion (lumpy — normalize over the cycle); segment margins vs best-in-class peers.
Red flags: fixed-price development-program losses (the aerospace classic); backlog inflated by low-margin or cancellable orders; cyclical capex peak misread as structural demand; pension deficits; serial-acquirer integration failures; guidance premised on backlog that can slip right.
Cycle: mid-to-late-cycle; machinery turns with global/commodity capex; aerospace recovers over multi-year horizons after shocks (supply chains, program failures); defense follows budget cycles with a lag.
Valuation norms: through-cycle margins and FCF conversion; EV/EBITDA vs peers and own history; sum-of-the-parts for conglomerates; never pay a market multiple for backlog whose conversion margin is unproven.""",
    "basic materials": """Sector: MATERIALS / MINING / CHEMICALS / PACKAGING.
Industry structure: commodity price-takers where the durable advantage is occupying the low end of the cost curve — in downturns the high-cost quartile preserves nothing while the low-cost quartile compounds through the cycle. Cost-curve position, reserve/resource quality, and jurisdiction risk are the entire franchise. Chemicals split into commodity (feedstock-advantaged, e.g. US ethane) vs specialty (application/mix moats, steadier margins).
Value drivers: all-in sustaining cost vs the peer curve; reserve life and replacement; project pipeline and execution (mines are decade-long, over-budget builds); commodity-specific demand drivers (copper = electrification/grid; lithium = EVs; potash = agriculture); contract vs spot mix (chemicals); downstream integration.
KPIs to check: AISC/cash cost vs peers; reserve replacement ratio; mine life; capex discipline and project-overrun history; volume vs price contribution; jurisdiction concentration; maintenance-capex honesty.
Red flags: cost inflation eroding cost-curve position; large project overruns; M&A executed at commodity peaks (the sector's classic mistake — buying the top); resource nationalism and regulatory interference; hedges that cap upturns; understated maintenance capex.
Cycle: late-cycle — materials inflect with global industrial demand and often lead energy; chemicals turn first in a manufacturing recovery. Never treat a trough year's earnings as permanent (or a peak year's).
Valuation norms: P/NAV for miners (with jurisdiction/execution discounts); mid-cycle EV/EBITDA (trough multiples look alarming at trough earnings — that is the point of normalizing); replacement cost as a floor for quality assets; specialty chemicals earn modest premium multiples for mix stability.""",
    "communication services": """Sector: COMMUNICATION SERVICES (telecom / media & streaming / interactive platforms).
Industry structure: three genuinely different businesses under one GICS label — analyze the actual sub-industry, never the blend.
- Telecom: capital-intensive near-utilities; value = subscriber economics and FCF after capex. Fiber/mobile convergence and bundling fight churn; tower economics beat owning towers. Leverage and dividend coverage dominate the equity story.
- Media/streaming: content is a decaying asset (amortized over ~4 years) — the moat is franchise IP, engagement, and distribution scale; linear TV is in secular decline (quantify the decline rate and the bridge math to streaming profitability).
- Platforms: network-effect ad monopolies — value = engagement durability, ad take-rates, and regulatory/AI disruption risk. Antitrust/DMA actions are a real overhang; AI answer engines vs search is a live structural question.
KPIs to check: subscriber adds and churn; ARPU trends; FCF after capex (telcos); net debt/EBITDA and dividend coverage; content amortization policy (aggressive amortization flatters earnings); ad pricing vs volume; engagement (DAU) trends; spectrum payments.
Red flags: subscriber losses at pricing-power names; telco capex arms races with no price response; engagement declines masked by rising ad load; content spend growing faster than revenue; antitrust remedies aimed at the core monetization engine.
Cycle: ad revenue is macro-cyclical (media/platform budgets cut first); telecom is defensive yield; streaming economics are a transition story largely independent of the cycle.
Valuation norms: telcos = yield proxies (dividend coverage, leverage, FCF after capex); platforms = FCF and user-value durability (state explicitly how long the growth run-rate lasts); media = content-library ROI and the linear-to-streaming bridge. Pick the right lens per sub-industry.""",
    "real estate": """Sector: REAL ESTATE (REITs / landlords / developers).
Industry structure: levered property ownership — the analysis is asset-level (NOI, lease terms, occupancy) plus capital-structure (LTV, maturity ladder, covenants). Subsectors are disconnected businesses: office is in a structural demand shock (hybrid work); industrial benefits from e-commerce; data centers ride AI/cloud capex; senior housing faces a favorable supply/demand setup; retail bifurcates (Class A strong, B/C dying). GICS hides this — identify the actual subsector before anything else.
Value drivers: same-property NOI growth; lease terms and rollover (WALT — short WALT is re-pricing risk or opportunity depending on market rents); occupancy and leasing spreads; development-pipeline yields vs cost of capital; balance sheet (LTV, fixed vs floating, hedges rolling off, maturity wall).
KPIs to check: same-property NOI; occupancy; leasing spreads (new vs expiring rents); interest-expense trajectory as hedges roll off; AFFO payout ratio; price/NAV; implied cap rate vs debt cost; external capital access (is equity issuance accretive or dilutive vs NAV?).
Red flags: AFFO payout >100% (distributing capital, not income); floating-rate exposure repricing upward; tenant concentration and tenant credit; development capex funded with distressed equity; office rollover cliff; covenant pressure.
Cycle: interest-rate dominated — REITs derate wholesale when rates rise; separate the rate effect (sector-wide, mean-reverting) from asset-specific impairment (secular demand shift — permanent). Supply cycles are subsector-specific (industrial boomed 2015-2022; office demand broke permanently).
Valuation norms: NAV per share is the anchor — price/NAV premium/discount, implied cap rate vs debt cost (a negative spread means leverage is destroying value), dividend yield vs bonds, AFFO multiple vs own history. The honest test: would you buy this property portfolio at today's price with today's financing?""",
    "etf": """Instrument type: ETF / INDEX FUND. This is NOT an operating company — never apply company analysis and never fabricate company financials.
Value drivers: index construction (how are constituents selected and weighted — cap-weight, equal-weight, factor screens — and what is the implicit bet?); cost (expense ratio, but more importantly tracking difference over 5-10 years); holdings concentration (top-10 weight and effective number of holdings — cap-weighted products are frequently mega-cap concentrated funds in disguise); fund flows and AUM trajectory (closures happen); structural design (physical vs synthetic, securities lending, sampling); tax efficiency (in-kind redemptions).
KPIs to check: tracking error/difference vs the stated index; premium/discount to NAV (critical for thin or international funds); bid/ask spread; AUM trend; overlap with what the rest of a typical portfolio already owns; rebalance frequency and its turnover/cost drag.
Red flags: yield-chasing products distributing principal (check return-of-capital in distributions); concentration dressed as diversification; leveraged/inverse products whose daily-reset math decays in volatile sideways markets; methodology changes that quietly alter exposure; synthetic replication counterparty risk; buying an index whose constituents have already run.
Cycle: an ETF is a "which exposure and when" question, not an "is this cheap" question — the macro and market analysts carry the timing weight. Still, evaluate the underlying index's valuation percentile and the equity risk premium vs bonds, plus the current stage of the sector-rotation cycle for sector funds.
Valuation: index-level P/E, earnings yield, and historical percentile for equity funds; duration, credit quality, and yield-to-maturity vs the curve for bond funds; roll-yield drag vs spot exposure for commodity funds. State explicitly what exposure/factor the investor is buying and what would make it underperform.""",
    "general": """Sector: GENERAL / UNCLASSIFIED. Apply a disciplined first-principles sector analysis and derive every answer from the filings, transcripts, and peer data — never assert them without evidence:
1. Industry structure (Porter-style): who holds pricing power and why; what the real barriers to entry are (scale, switching costs, network effects, licenses, patents); how concentrated suppliers and customers are; how credible substitution is; whether rivalry expresses itself in price, features, or service.
2. Business-model mechanics: what are the unit economics (revenue per unit × margin per unit × retention)? Where in the value chain does profit actually pool (it is rarely where revenue is largest)?
3. Structural demand: which secular trends drive or erode demand, and which ones is this company actually levered to?
4. KPIs: what do sophisticated investors in this industry actually watch? Mine the filings (segment data, disclosed KPIs) and earnings calls (management's own steering metrics) for them.
5. Capital impairment: what are the classic ways this industry permanently destroys capital (over-building at cycle peaks, regulatory change, technology substitution, serial M&A)? Which apply here, with evidence?
6. Cyclicality: is this an early-cycle, late-cycle, defensive, or rate-sensitive business, and where in the cycle are we now?
7. Valuation: which methodology fits this cash-flow profile (DCF, EV/EBITDA, P/TBV+ROE, NAV, FCF yield), and what have normal multiples been across a full cycle?""",
}


# yfinance GICS sector string -> playbook key
_SECTOR_KEY_MAP = {
    "healthcare": "healthcare",
    "technology": "technology",
    "utilities": "utilities",
    "energy": "energy",
    "financial services": "financial services",
    "consumer defensive": "consumer defensive",
    "consumer cyclical": "consumer cyclical",
    "industrials": "industrials",
    "basic materials": "basic materials",
    "communication services": "communication services",
    "real estate": "real estate",
}


def select_sector_playbook(profile: dict) -> str:
    """Pick the playbook key for an instrument profile dict."""
    quote_type = (profile.get("quote_type") or "").strip().lower()
    if quote_type in ("etf", "mutual fund", "index"):
        return "etf"
    sector = (profile.get("sector") or "").strip().lower()
    return _SECTOR_KEY_MAP.get(sector, "general")


def create_sector_analyst(llm, enable_web_search=True):
    def sector_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        profile = resolve_instrument_profile(ticker)
        playbook_key = select_sector_playbook(profile)
        playbook = SECTOR_PLAYBOOKS[playbook_key]
        sector_label = (
            profile.get("sector", "Unknown")
            if playbook_key != "etf"
            else "ETF / Fund"
        )

        tools = [
            get_company_profile,
            get_sector_performance,
            get_peer_comparison,
            get_10k_filing,
            get_20f_filing,
            get_earnings_call_transcripts,
        ]
        if enable_web_search:
            tools.append(web_search)

        system_message = (
            f"You are a SECTOR SPECIALIST — an analyst who has spent their career "
            f"covering this single industry, the way a top-tier sell-side specialist "
            f"or a sector-focused portfolio manager would. You supply the "
            f"sector-specific judgment the generalist analysts cannot: which value "
            f"drivers and KPIs actually matter here, how industry structure shapes "
            f"who captures the profit pool, where the sector sits in its business "
            f"cycle, what the regulatory/cyclical exposure is, and what the classic "
            f"permanent capital-impairment mistakes in this sector look like. "
            f"The resolved sector for {ticker} is: {sector_label}.\n\n"
            "Apply this master framework, then specialize it with the sector "
            "playbook below:\n"
            "0. **Identify the true sub-industry first.** GICS sector labels hide "
            "wildly different businesses (e.g., IT services vs software; office REIT "
            "vs data-center REIT; commodity chemicals vs specialty). Name the actual "
            "sub-industry and, where the label is misleading, say so — this is often "
            "the single most important sentence in the report.\n"
            "1. **Industry structure.** Where does the profit pool sit in the value "
            "chain, who holds bargaining power (suppliers, buyers, distributors), "
            "what are the genuine barriers to entry, and how intense/expressed is "
            "rivalry? Rank the company on these dimensions, not just on size.\n"
            "2. **Value drivers and KPIs.** Quantify the sector's key metrics for "
            "this company with the checklist below, using data from the tools.\n"
            "3. **Cycle & rotation position.** Is this an early-cycle, late-cycle, "
            "defensive, or rate-sensitive sector, where does it currently sit, and "
            "are current sector flows consistent with fundamentals? Never let a "
            "favorable cycle mask a deteriorating franchise, or vice versa.\n"
            "4. **Regulatory / policy / geopolitical exposure** — and whether the "
            "market is currently pricing it correctly.\n"
            "5. **Red flags.** Which of the sector's classic permanent-impairment "
            "patterns (if any) are present here, with evidence.\n"
            "6. **Valuation method fit.** State which methodology fits this "
            "cash-flow profile, what 'normal' looks like across a full cycle, and "
            "where the company trades vs that norm.\n\n"
            "Use the available tools first: `get_company_profile` for the business "
            "description, `get_sector_performance` for sector benchmark performance "
            "and the company's alpha vs SPY, `get_peer_comparison` for how the "
            "company stacks up against direct competitors on the sector's key "
            "metrics, `get_10k_filing`/`get_20f_filing` for industry-structure and "
            "regulatory disclosures, and `get_earnings_call_transcripts` for how "
            "management talks about industry conditions. If the instrument is an "
            "ETF, company filings and transcripts will return nothing — use "
            "`web_search` for index methodology, expense ratio, and holdings "
            "concentration instead, and never fabricate company financials.\n\n"
            f"{playbook}\n\n"
            "Evidence rules: quantify every important claim with a number from the "
            "tool outputs (filings, peer comparisons, transcripts); prefer specifics "
            "over adjectives; when a key sector KPI cannot be verified with the "
            "available data, say so explicitly instead of inventing it. A sector "
            "specialist who cannot tell you whether the company is gaining or losing "
            "share has failed at the core job.\n\n"
            "Write a comprehensive sector-specialist report covering:\n"
            "1. **Sub-industry & industry structure** — the true sub-industry, "
            "competitive landscape, where pricing power sits, and how the company "
            "ranks on the sector's key metrics vs peers.\n"
            "2. **Sector-specific value drivers and KPIs** — quantify them for this "
            "company with the checklist above.\n"
            "3. **Cycle & rotation position** — the sector's cycle profile and where "
            "we are in it, plus any market-pricing inconsistency you see.\n"
            "4. **Regulatory / policy / cyclicality exposure** — and whether the "
            "market is currently pricing it correctly.\n"
            "5. **Sector red flags** — which of the classic impairment patterns "
            "(if any) are present here, with evidence.\n"
            "6. **Valuation norms** — what multiple/methodology fits this sector, "
            "what 'normal' looks like across a full cycle, where this company "
            "trades vs those norms, and whether the premium or discount is "
            "justified by sector-specific fundamentals.\n\n"
            "End with a Markdown table organizing the key points, including an "
            "explicit row stating whether the company is a LEADER or LAGGARD on "
            "the sector's core value drivers."
            + (WEB_SEARCH_INSTRUCTION if enable_web_search else "")
            + get_language_instruction()
            + get_report_hygiene_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sector_report": report,
        }

    return sector_analyst_node
