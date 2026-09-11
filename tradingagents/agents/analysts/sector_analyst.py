"""Sector Specialist: a dynamically-allocated analyst that applies sector-specific
domain knowledge (value drivers, regulation, cyclicality, KPIs, valuation norms)
on top of the generic analyst team. The playbook is selected at runtime from the
instrument's actual sector / quote type, so the same agent covers pharma,
semiconductors, utilities, consumer staples, ETFs, and everything else without
sector-specific code paths elsewhere in the graph.
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
# generic template with the sector name swapped in.
SECTOR_PLAYBOOKS = {
    "healthcare": """Sector: HEALTHCARE (pharma / biotech / medtech).
Value drivers: patent exclusivity runway and patent-cliff dates per product; pipeline probability-weighting (phase-transition base rates); concentration of revenue and profit in the top 1-3 products; payer/pricing dynamics (rebates, IRA-style negotiation, national health systems); regulatory approvals and label expansions.
KPIs to check: product-level revenue splits, gross-to-net trend, R&D productivity (approvals per $ spent), prescription/volume share vs competitors where disclosed.
Red flags: single-molecule concentration with expiry inside the holding horizon; pipeline failures on lead assets; FDA/EMA complete-response letters; pricing legislation aimed at the flagship product; litigation on core IP.
Valuation norms: mature pharma trades 10-15x earnings; pipeline value should be discounted heavily until Phase 3. Judge cheapness vs the patent-protected cash-flow runway, not vs the growth multiple the company had at peak.""",
    "technology": """Sector: TECHNOLOGY (software / internet / hardware).
Value drivers: durable competitive moat (switching costs, network effects, ecosystem lock-in); recurring-revenue mix and net revenue retention; gross-margin structure; R&D converting to product leadership; platform monetization runway.
KPIs to check: revenue growth deceleration, SBC dilution vs buybacks, customer/concentration metrics, cash conversion (FCF vs net income).
Red flags: multiple compression from decelerating growth; heavy stock-based compensation masking weak per-share economics; reliance on one platform/channel; unproven AI/product pivots priced as certainties.
Valuation norms: judge vs FCF yield and growth-adjusted multiples, not headline P/E; deep drawdowns in quality names often reflect multiple normalization, not business impairment — distinguish the two.""",
    "semiconductors": """Sector: SEMICONDUCTORS / SEMICONDUCTOR EQUIPMENT.
Value drivers: process-node leadership and design wins; customer concentration (foundries/OSATs/hyperscalers); capital-cycle position; equipment backlog and utilization; export-control exposure.
KPIs to check: utilization rates, book-to-bill, inventory days across the chain, capex announcements by customers, geographic revenue (Taiwan/China/US).
Red flags: peak-cycle multiples on peak earnings; customer capex digestion after a boom; export restrictions on revenue-critical geographies; design losses at large customers.
Valuation norms: the sector is deeply cyclical — trailing multiples mislead at cycle extremes. Normalize earnings across the cycle before judging cheapness; mid-cycle P/E on normalized EPS is the honest yardstick.""",
    "utilities": """Sector: UTILITIES (regulated / IPP / water / grid).
Value drivers: regulated rate base growth and allowed ROE; regulatory relationships and track record; fuel mix and hedging; rate-case outcomes; load growth (data centers/electrification is a live driver).
KPIs to check: rate-base growth guidance, allowed vs earned ROE, regulatory lag, capital plan funding (equity issuance dilution), pension/OPEB status.
Red flags: aggressive capex plans predicated on friendly regulators; merchant exposure dressed as regulated; high payout + heavy capex = structural equity dilution; nuclear/coal stranded-asset liabilities.
Valuation norms: bond proxies — judge vs dividend yield spread over the 10-year Treasury, P/E vs historical premium/discount to the sector, and rate-base growth. Rising rates compress the whole sector; distinguish rate-driven derating from company-specific problems.""",
    "energy": """Sector: ENERGY (E&P / integrated / oilfield services).
Value drivers: reserve quality and replacement cost; production per share; breakeven price vs strip; capital discipline (returns of capital vs growth capex); cost curve position.
KPIs to check: FCF at strip prices, decline rates, hedging book, reserve life index, net debt/EBITDAX across the cycle.
Red flags: reserve writedowns; acquisitions at cycle peaks; payout promises that break below mid-cycle prices; cost inflation eating margin.
Valuation norms: value is in normalized FCF yield across the cycle (e.g. FCF at $60-70 Brent), not spot earnings. Assets in the ground deserve a discount to PV-10; integrateds deserve a conglomerate-quality judgment.""",
    "financial services": """Sector: FINANCIALS (banks / insurers / asset managers / exchanges).
Value drivers: underwriting or credit discipline through cycles; deposit franchise cost; float/insurance margin; asset-management fee streams; capital return capacity (CET1, excess capital).
KPIs to check: ROE vs cost of equity, NIM trend, credit-loss provisions and charge-offs, reserve ratios, book value trajectory, payout sustainability.
Red flags: earnings propped by reserve releases; rapid loan growth in late cycle; duration risk in securities books; conglomerate complexity hiding losses (check segments).
Valuation norms: P/TBV vs ROE is the primary lens (residual income / dividend discount models fit here better than DCF-on-FCF). A bank earning below its cost of equity deserves P/TBV << 1 — check which regime you are in before calling it cheap.""",
    "consumer defensive": """Sector: CONSUMER STAPLES.
Value drivers: brand pricing power vs private label; shelf-space and distribution moats; input-cost pass-through; emerging-market exposure and route-to-market.
KPIs to check: volume vs price/mix split of growth (price-led growth without volume is a decaying franchise), gross-margin recovery after inflation, market-share trends, retailer concentration.
Red flags: volume declines masked by pricing; brand equity erosion (trade-down); GLP-1-type structural demand shifts; emerging-market FX and expropriation.
Valuation norms: low-growth, high-visibility cash flows — judge on FCF yield, dividend coverage, and P/E vs its own history; a staples premium evaporates when volume growth turns negative.""",
    "consumer cyclical": """Sector: CONSUMER DISCRETIONARY.
Value drivers: brand loyalty and unit economics at the store/model level; same-store sales vs new-unit returns; scale advantages in sourcing/logistics; channel shift management.
KPIs to check: SSS growth, unit growth ROI, inventory vs sales, credit exposure of customers (BNPL/subprime financing), promotional intensity.
Red flags: growth bought with declining unit economics; inventory build ahead of demand cracks; reliance on consumer credit expansion; key-man/founder departures.
Valuation norms: highly cyclical — trailing P/E is at its lowest exactly at cycle peaks. Judge on mid-cycle margins and balance-sheet resilience; avoid catching the multiple before the earnings cut.""",
    "industrials": """Sector: INDUSTRIALS (aerospace / machinery / transport / defense).
Value drivers: backlog quality and conversion; aftermarket/services revenue share (recurring, high-margin); pricing on long-term contracts; capex cycle exposure.
KPIs to check: book-to-bill, backlog coverage, program margins (especially fixed-price defense contracts), working-capital swings, order cancellations.
Red flags: fixed-price program losses; backlog inflated by low-margin orders; cyclical capex peak read as structural demand; pension deficits.
Valuation norms: judge on through-cycle margins and FCF conversion; backlog-based P/E can mislead when backlog conversion is uncertain.""",
    "basic materials": """Sector: MATERIALS / MINING / CHEMICALS.
Value drivers: position on the industry cost curve; reserve life / resource quality; commodity price sensitivity and hedging; project pipeline and execution risk.
KPIs to check: all-in sustaining costs vs peers, reserve replacement, capex discipline, downstream integration, contract vs spot mix (chemicals).
Red flags: cost inflation eroding cost-curve position; big project overruns; chasing commodity highs with M&A; state/regulatory interference in key jurisdictions.
Valuation norms: same cyclical caveat as energy/semis — normalized mid-cycle earnings and replacement value (P/NAV for miners), not spot multiples.""",
    "communication services": """Sector: COMMUNICATION SERVICES (telecom / media / platforms).
Value drivers: subscriber economics (ARPU, churn); content/platform moats; spectrum/network asset position; ad-market cyclicality for media; capital intensity for telcos.
KPIs to check: subscriber adds and churn, ARPU trends, FCF after capex (telcos), ad pricing vs volume, content amortization policies.
Red flags: subscriber losses at pricing power names; telco capex arms races; engagement decline masked by ad-load increases; regulatory/antitrust action on platform economics.
Valuation norms: telcos are yield proxies (judge vs dividend coverage and leverage); platforms on FCF and user-value durability; media on content library economics.""",
    "real estate": """Sector: REAL ESTATE (REITs / landlords / developers).
Value drivers: property-level NOI growth; lease terms and rollover (WALT); occupancy; development pipeline returns; balance sheet (LTV, debt maturity, covenants).
KPIs to check: same-property NOI, occupancy and leasing spreads, interest expense trajectory vs hedges rolling off, payout ratio vs AFFO, external capital access.
Red flags: AFFO payout >100%; floating-rate exposure repricing upward; tenant concentration; development capex funded at distressed equity costs; office-sector structural demand shift.
Valuation norms: NAV per share is the anchor — compare price/NAV, implied cap rate vs debt cost, and dividend yield vs bond yields. Rate moves dominate; separate rate effects from asset-specific issues.""",
    "etf": """Instrument type: ETF / INDEX FUND. This is NOT an operating company — do not apply company analysis.
Value drivers: benchmark/index construction (how are constituents selected and weighted?); cost (expense ratio, tracking difference over 5-10 years); holdings concentration (top-10 weight, effective number of holdings); fund flows and AUM trend; structural design (physical vs synthetic, securities lending); distribution and tax efficiency.
KPIs to check: tracking error/difference vs the stated index, premium/discount to NAV (especially for thinner funds), bid/ask spread, AUM trajectory (closures happen), overlap with what the investor already owns.
Red flags: yield-chasing products that distribute principal; concentration risk dressed as diversification (cap-weighted mega-cap tilt); leveraged/inverse decay if applicable; methodology changes that alter exposure; synthetic replication counterparty risk.
Valuation: the underlying index's valuation is what matters — compare index-level P/E, earnings yield, and historical percentile. An ETF is almost always a "which exposure and when" question, not a "is this cheap" question; the entry-timing and macro analysts carry that weight.""",
    "general": """Sector: GENERAL / UNCLASSIFIED. Apply a disciplined first-principles sector analysis:
1. What actually drives value creation in this industry (unit economics, regulation, technology, cyclicality)?
2. Where does pricing power come from, and who has it?
3. What are the industry's structural demand trends and structural threats?
4. What KPIs do sophisticated investors in this industry actually watch?
5. What are the classic ways companies in this industry permanently impair capital?
6. What valuation methodology fits this industry's cash-flow profile, and what have normal multiples been across a full cycle?
Derive the answers from the filings, transcripts, and peer data rather than asserting them.""",
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
            f"You are a SECTOR SPECIALIST analyst with deep domain expertise in this "
            f"instrument's industry. Your job is to supply the sector-specific judgment "
            f"the generalist analysts cannot: which value drivers and KPIs actually "
            f"matter here, how the industry structure shapes returns, what the "
            f"regulatory/cyclical exposure is, and what the classic permanent "
            f"capital-impairment mistakes in this sector look like. "
            f"The resolved sector for {ticker} is: {sector_label}.\n\n"
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
            "Write a comprehensive sector-specialist report covering:\n"
            "1. **Industry structure** — competitive landscape, where pricing power "
            "sits, and how the company ranks on the sector's key metrics vs peers.\n"
            "2. **Sector-specific value drivers and KPIs** — quantify them for this "
            "company with the checklist above.\n"
            "3. **Regulatory / policy / cyclicality exposure** — and whether the "
            "market is currently pricing it correctly.\n"
            "4. **Sector red flags** — which of the classic impairment patterns "
            "(if any) are present here, with evidence.\n"
            "5. **Valuation norms** — what multiple/methodology fits this sector, "
            "where this company trades vs those norms, and whether the premium or "
            "discount is justified by sector-specific fundamentals.\n\n"
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
