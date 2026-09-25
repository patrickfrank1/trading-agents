"""SEC-filing signal extractors — one tool per high-value disclosure.

Each tool pulls a specific signal out of a 10-K (or, where appropriate, a
DEF 14A / 8-K / S-3 / 13D / 13F) and returns the relevant passages verbatim
so the analyst reads the actual filing language rather than a paraphrase.
All calls degrade gracefully to "not found" messages; none can crash the
pipeline.

Grouped:
  Tier 1 (cross-sector): debt maturities, off-balance-sheet, segment/geographic,
    RPO disaggregation, risk-factor diff, legal proceedings, critical
    accounting estimates, internal controls (material weakness).
  Tier 2 (sector-conditional): stock-based comp, goodwill/intangibles, pension/
    OPEB, uncertain tax positions, VIEs, regulatory capital, proved reserves/
    mine safety, cybersecurity, properties/capacity, commitments & contingencies.
  Tier 3 (other filings): proxy governance (DEF 14A), activist filings (13D/13G),
    institutional 13F filings, 8-K event classification, insider Form 4 codes,
    prospectus (S-3/424B).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

import yfinance as yf
from langchain_core.tools import tool

from tradingagents.dataflows.sec_edgar import (
    _fetch_filings_text,
    _get_item_section,
    _extract_keyword_passages,
    _format_passage_report,
    _edgar_fulltext_search,
    _company_name_for_search,
    _sec_request,
    _ingest_complete_filing_text,
)
from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.agents.utils.tool_errors import safe_tool


def _latest_filing(ticker, form_types, curr_date=None, max_filings=1):
    fs = _fetch_filings_text(ticker, set(form_types), before_date=curr_date, max_filings=max_filings)
    return fs


def _C(pattern):
    return re.compile(pattern, re.IGNORECASE)


# ===========================================================================
# Tier 1 — cross-sector
# ===========================================================================


@tool
@safe_tool
def get_debt_maturity_schedule(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract the contractual-obligations / debt-maturity table and surrounding
    disclosure from the latest annual filing (year-by-year debt principal
    maturities, plus operating leases and purchase obligations).

    Use this to resolve refinancing-risk claims (e.g. 'debt maturities out to
    2066') with exact per-year figures instead of assertions.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"contractual\s+obligations"),
        _C(r"maturit(?:y|ies)\s+of\s+(?:long[- ]term\s+)?debt"),
        _C(r"debt\s+maturities"),
        _C(r"principal\s+(?:payments|repayments)\s+due"),
        _C(r"long[- ]term\s+debt.{0,30}(?:due|maturing|maturities)"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=1000, max_passages=6)
    return _format_passage_report(
        f"Debt Maturity / Contractual Obligations for {ticker.upper()}",
        fs[0]["filing"], passages,
        "No contractual-obligations or debt-maturity disclosure matched. "
        "Look for a 'Long-term Debt' footnote with a maturities table.",
    )


@tool
@safe_tool
def get_off_balance_sheet_arrangements(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract off-balance-sheet arrangements, guarantees, and standby
    commitments (operating leases, financial guarantees, letters of credit,
    VIE exposures) from the latest annual filing.

    Use this to see the real leverage vs. what's on the balance sheet.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"off[- ]balance[- ]sheet\s+arrangements?"),
        _C(r"off\s+balance\s+sheet"),
        _C(r"financial\s+guarantees?"),
        _C(r"standby\s+letters?\s+of\s+credit"),
        _C(r"variable\s+interest\s+entities?"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=900, max_passages=6)
    return _format_passage_report(
        f"Off-Balance-Sheet Arrangements for {ticker.upper()}",
        fs[0]["filing"], passages,
    )


@tool
@safe_tool
def get_segment_geographic_reporting(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract the segment-information and revenue-disaggregation footnote
    (revenue, operating income, assets, capex by segment and geography).

    Use this for SOTP analysis and to isolate which segment drives value, plus
    geographic concentration (e.g. China/Russia exposure).
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"segment\s+(?:information|reporting|data)"),
        _C(r"reportable\s+segments?"),
        _C(r"disaggregation\s+of\s+revenue"),
        _C(r"revenue\s+by\s+geograph"),
        _C(r"geographic\s+(?:information|areas?)"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=1000, max_passages=8)
    return _format_passage_report(
        f"Segment & Geographic Reporting for {ticker.upper()}",
        fs[0]["filing"], passages,
    )


@tool
@safe_tool
def get_rpo_disaggregation(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract the revenue-recognition footnote, focusing on remaining
    performance obligations (RPO) disaggregated by time-to-recognition
    (within 12 months vs. thereafter) and by type/segment.

    Use this to resolve the order-book debate: short-term RPO = near-term
    contracted revenue; long-term RPO may be legacy support.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"remaining\s+performance\s+obligations?"),
        _C(r"performance\s+obligations?\s+(?:recognized|satisfied|expected)"),
        _C(r"revenue\s+recognition"),
        _C(r"deferred\s+(?:revenue|performance)"),
        _C(r"recognized\s+(?:in|as)\s+revenue"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=900, max_passages=8)
    return _format_passage_report(
        f"RPO / Revenue Recognition Disaggregation for {ticker.upper()}",
        fs[0]["filing"], passages,
    )


def _sentence_set(text, max_chars=150000):
    if not text:
        return set()
    text = text[:max_chars]
    parts = re.split(r"(?<=[.;])\s+", text)
    return {re.sub(r"\s+", " ", p).strip().lower() for p in parts if len(p.strip()) > 40}


@tool
@safe_tool
def get_risk_factor_changes(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Compare Item 1A (Risk Factors) between the two most recent annual
    filings and report newly-added and removed risk factors (sentence-level).

    Newly-disclosed risks are a leading indicator — a company adding 'we
    depend on a single supplier' or a cybersecurity risk is signalling
    something changed. This is the only genuinely leading qualitative signal
    in the 10-K.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date, max_filings=2)
    if len(fs) < 2:
        return f"Need two annual filings for {ticker} to diff risk factors; found {len(fs)}."
    cur_1a = _get_item_section(fs[0]["text"], "1a", max_chars=150000)
    prior_1a = _get_item_section(fs[1]["text"], "1a", max_chars=150000)
    if not cur_1a or not prior_1a:
        return "Could not isolate Item 1A (Risk Factors) in one or both filings."
    cur_s = _sentence_set(cur_1a)
    prior_s = _sentence_set(prior_1a)
    added = sorted(cur_s - prior_s)
    removed = sorted(prior_s - cur_s)

    lines = [
        f"# Risk-Factor Changes (YoY) for {ticker.upper()}",
        f"Current filing: {fs[0]['filing']['form_type']} filed {fs[0]['filing']['filing_date']}",
        f"Prior filing:   {fs[1]['filing']['form_type']} filed {fs[1]['filing']['filing_date']}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"## Newly-added risk-factor statements ({len(added)})",
    ]
    lines.extend(added[:30] if added else ["(none detected)"])
    lines.append(f"\n## Removed risk-factor statements ({len(removed)})")
    lines.extend(removed[:20] if removed else ["(none detected)"])
    lines.append(
        "\nNote: sentence-level diff with normalization; minor rewording may "
        "appear as add/remove. Focus on substantively new risk language."
    )
    return "\n".join(lines)


@tool
@safe_tool
def get_legal_proceedings(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract Item 3 (Legal Proceedings) — material litigation, environmental,
    and regulatory matters with quantified accruals and reasonably-possible
    losses.

    Use this for pharma, banks, industrials, and energy. The gap between
    accrual and disclosed exposure is itself a signal.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    section = _get_item_section(fs[0]["text"], "3", max_chars=20000)
    patterns = [_C(r"legal\s+proceedings"), _C(r"material\s+litigation"),
                _C(r"environmental\s+(?:matters|proceedings)")]
    passages = []
    if section:
        passages = [re.sub(r"\s+", " ", section).strip()]
    else:
        passages = _extract_keyword_passages(fs[0]["text"], patterns, context=900, max_passages=5)
    return _format_passage_report(
        f"Legal Proceedings for {ticker.upper()}", fs[0]["filing"], passages,
    )


@tool
@safe_tool
def get_critical_accounting_estimates(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract the 'Critical Accounting Estimates' discussion from MD&A —
    management's own list of where its numbers are most uncertain (credit-loss
    allowances, goodwill impairment, valuation allowances, reserves).

    Use this to flag earnings-quality soft spots management itself acknowledges.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"critical\s+accounting\s+estimates?"),
        _C(r"critical\s+accounting\s+policies?"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=1100, max_passages=6)
    return _format_passage_report(
        f"Critical Accounting Estimates for {ticker.upper()}",
        fs[0]["filing"], passages,
        "No 'Critical Accounting Estimates' section matched in MD&A.",
    )


@tool
@safe_tool
def get_internal_controls(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract Item 9A (Controls and Procedures) — internal-control
    deficiencies, material weaknesses, and restatements.

    A material weakness is a strong negative signal and a restatement-risk
    leading indicator.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    section = _get_item_section(fs[0]["text"], "9a", max_chars=15000)
    passages = []
    if section:
        passages = [re.sub(r"\s+", " ", section).strip()]
    else:
        patterns = [_C(r"material\s+weakness"), _C(r"significant\s+deficiency"),
                    _C(r"internal\s+control\s+over\s+financial\s+reporting")]
        passages = _extract_keyword_passages(fs[0]["text"], patterns, context=900, max_passages=5)
    return _format_passage_report(
        f"Internal Controls (Item 9A) for {ticker.upper()}", fs[0]["filing"], passages,
    )


# ===========================================================================
# Tier 2 — sector-conditional
# ===========================================================================


def _passage_tool(name, title_label, patterns, not_found_note=""):
    """Factory building a named 10-K keyword-passage tool."""

    def _tool(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
    ) -> str:
        fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
        if not fs:
            return f"No annual filing found for {ticker} on SEC EDGAR."
        compiled = [_C(p) if isinstance(p, str) else p for p in patterns]
        passages = _extract_keyword_passages(fs[0]["text"], compiled, context=900, max_passages=6)
        return _format_passage_report(
            f"{title_label} for {ticker.upper()}", fs[0]["filing"], passages, not_found_note
        )

    _tool.__name__ = name
    _tool.__doc__ = (
        f"Extract {title_label.lower()} disclosures from the latest annual "
        "filing. Returns the relevant passage(s) verbatim."
    )
    return tool(safe_tool(_tool))


get_stock_based_compensation = _passage_tool(
    "get_stock_based_compensation",
    "Stock-Based Compensation",
    [
        r"stock[- ]based\s+compensation",
        r"share[- ]based\s+compensation",
        r"restricted\s+stock\s+units?",
        r"\bRSUs?\b",
        r"unvested\s+awards?",
    ],
    "No stock-based-compensation footnote matched. The company may not issue equity awards.",
)

get_goodwill_intangibles = _passage_tool(
    "get_goodwill_intangibles",
    "Goodwill & Intangibles",
    [
        r"goodwill\s+and\s+intangible",
        r"impairment\s+(?:of\s+)?(?:goodwill|intangible)",
        r"intangible\s+assets",
        r"goodwill\s+impairment",
    ],
)

get_pension_opeb = _passage_tool(
    "get_pension_opeb",
    "Pension & Postretirement Obligations",
    [
        r"pension\s+(?:benefits?|plan|obligations?)",
        r"postretirement",
        r"defined\s+benefit",
        r"\bOPEB\b",
        r"retirement\s+benefits?",
    ],
    "No pension/postretirement footnote matched. The company may not sponsor a defined-benefit plan.",
)

get_uncertain_tax_positions = _passage_tool(
    "get_uncertain_tax_positions",
    "Uncertain Tax Positions",
    [
        r"uncertain\s+tax\s+positions?",
        r"unrecognized\s+tax\s+benefits?",
        r"FIN\s?48",
        r"income\s+taxes?\s+(?:note|disclosure|uncertain)",
    ],
)

get_variable_interest_entities = _passage_tool(
    "get_variable_interest_entities",
    "Variable Interest Entities (VIE)",
    [
        r"variable\s+interest\s+entit",
        r"\bVIEs?\b",
        r"special[- ]purpose\s+entit",
        r"primary\s+beneficiary",
    ],
)

get_regulatory_capital = _passage_tool(
    "get_regulatory_capital",
    "Regulatory Capital",
    [
        r"regulatory\s+capital",
        r"Basel\s+III",
        r"risk[- ]weighted\s+assets?",
        r"tier\s+1\s+capital",
        r"capital\s+adequacy",
        r"common\s+equity\s+tier",
    ],
    "No regulatory-capital disclosure matched. The company is likely not a bank/insurer.",
)

get_commitments_contingencies = _passage_tool(
    "get_commitments_contingencies",
    "Commitments & Contingencies",
    [
        r"commitments?\s+and\s+contingenc",
        r"purchase\s+obligations?",
        r"long[- ]term\s+supply\s+(?:agreements?|commitments?)",
        r"operating\s+leases?",
    ],
)


@tool
@safe_tool
def get_proved_reserves_mine_safety(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract proved-reserves / reserve-replacement disclosures (oil & gas)
    and mine-safety disclosures (miners) from Item 4 and the reserves
    footnote.

    Reserve replacement ratio is the core depletion signal for E&P.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"proved\s+reserves?"),
        _C(r"reserve\s+replacement"),
        _C(r"probable\s+reserves?"),
        _C(r"mine\s+safety"),
        _C(r"oil\s+and\s+gas\s+reserves"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=1000, max_passages=6)
    return _format_passage_report(
        f"Proved Reserves / Mine Safety for {ticker.upper()}", fs[0]["filing"], passages,
        "No reserves or mine-safety disclosure matched. The company is likely not in E&P or mining.",
    )


@tool
@safe_tool
def get_cybersecurity_disclosure(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract Item 1C (Cybersecurity) — material incident disclosures and
    risk-management process (mandatory under the 2023 SEC rule).

    A leading signal now that cyber disclosure is required.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    section = _get_item_section(fs[0]["text"], "1c", max_chars=12000)
    passages = []
    if section:
        passages = [re.sub(r"\s+", " ", section).strip()]
    else:
        patterns = [_C(r"cybersecurity"), _C(r"cyber\s+(?:security|incident|risk)")]
        passages = _extract_keyword_passages(fs[0]["text"], patterns, context=800, max_passages=5)
    return _format_passage_report(
        f"Cybersecurity Disclosure (Item 1C) for {ticker.upper()}", fs[0]["filing"], passages,
        "No Item 1C cybersecurity disclosure matched (the filing may predate the 2023 rule).",
    )


@tool
@safe_tool
def get_properties_capacity(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract Item 2 (Properties) — owned vs. leased footprint, data-center
    capacity, plant utilization.

    For capex-heavy stories (e.g. cloud/AI capacity), this is the supply-side
    limit on growth.
    """
    fs = _latest_filing(ticker, {"10-K", "20-F"}, curr_date)
    if not fs:
        return f"No annual filing found for {ticker} on SEC EDGAR."
    section = _get_item_section(fs[0]["text"], "2", max_chars=12000)
    passages = []
    if section:
        passages = [re.sub(r"\s+", " ", section).strip()]
    else:
        patterns = [_C(r"properties"), _C(r"data\s+centers?"),
                    _C(r"manufacturing\s+facilities"), _C(r"production\s+capacity")]
        passages = _extract_keyword_passages(fs[0]["text"], patterns, context=800, max_passages=5)
    return _format_passage_report(
        f"Properties / Capacity (Item 2) for {ticker.upper()}", fs[0]["filing"], passages,
    )


# ===========================================================================
# Tier 3 — other filings
# ===========================================================================


@tool
@safe_tool
def get_proxy_governance(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract executive compensation, related-party transactions, director
    independence, and say-on-pay disclosures from the latest DEF 14A proxy
    statement.

    Use this for governance signals (executive pay-vs-performance, founder
    compensation, board independence).
    """
    fs = _latest_filing(ticker, {"DEF 14A"}, curr_date, max_filings=1)
    if not fs:
        return f"No DEF 14A proxy filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"executive\s+compensation"),
        _C(r"related\s+(?:party|parties)\s+transactions?"),
        _C(r"director\s+independence"),
        _C(r"say[- ]on[- ]pay"),
        _C(r"pay\s+versus\s+performance"),
        _C(r"compensation\s+committee"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=1000, max_passages=8)
    return _format_passage_report(
        f"Proxy / Governance (DEF 14A) for {ticker.upper()}", fs[0]["filing"], passages,
    )


@tool
@safe_tool
def get_activist_filings(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """List recent Schedule 13D/13G activist filings mentioning the company,
    and extract the 'Purpose of Transaction' (Item 4) from the most recent
    13D.

    An activist appearing is a catalyst; the 13D purpose clause states the
    thesis.
    """
    query = _company_name_for_search(ticker)
    hits = _edgar_fulltext_search(
        query, ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"], limit=10
    )
    if not hits:
        return f"No recent 13D/13G filings found for {ticker} ({query})."
    lines = [
        f"# Activist / 13D-13G Filings for {ticker.upper()}",
        f"Search query: {query}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Recent filings",
    ]
    for h in hits:
        lines.append(
            f"- {h['date']} | {h['form']} | filer: {h['filer']} | {h['url']}"
        )
    # Extract Item 4 purpose from the most recent 13D (not 13G, which has none).
    target = next((h for h in hits if "13D" in h["form"] and "/A" not in h["form"]), None)
    if target and target["url"]:
        try:
            listing = _sec_request(target["url"]).decode("utf-8", errors="replace")
            # Find the primary 13D document in the filing index.
            doc_match = re.search(r'href="([^"]*\.txt)"', listing)
            if doc_match:
                doc_url = target["url"] + doc_match.group(1)
                raw = _sec_request(doc_url).decode("utf-8", errors="replace")
                text = _ingest_complete_filing_text(raw)
                purpose = _extract_keyword_passages(
                    text, [_C(r"purpose\s+of\s+transaction"), _C(r"item\s+4")],
                    context=1200, max_passages=2,
                )
                if purpose:
                    lines.append("\n## Purpose of Transaction (most recent 13D, Item 4)")
                    lines.append(f"...{purpose[0]}...")
        except Exception:
            pass
    return "\n".join(lines)


@tool
@safe_tool
def get_institutional_13f_filings(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """List recent 13F-HR institutional filings that mention the company.

    Complements the yfinance top-holders view with the raw 13F source filings
    (full holdings including options positions) filed by institutions.
    """
    query = _company_name_for_search(ticker)
    hits = _edgar_fulltext_search(query, ["13F-HR", "13F-HR/A"], limit=12)
    if not hits:
        return f"No recent 13F-HR filings found mentioning {ticker} ({query})."
    lines = [
        f"# Institutional 13F Filings mentioning {ticker.upper()}",
        f"Search query: {query}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Recent 13F filers",
    ]
    for h in hits:
        lines.append(
            f"- {h['date']} | {h['form']} | filer: {h['filer']} | {h['url']}"
        )
    lines.append(
        "\nNote: EFTS matches by issuer name; verify each filer's holding for "
        "this ticker in the full 13F holding table at the filing URL."
    )
    return "\n".join(lines)


_8K_ITEM_DESCRIPTIONS = {
    "1.01": "Entry into Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "2.01": "Completion of Acquisition/Disposition",
    "2.02": "Results of Operations (earnings)",
    "2.03": "Direct Financial Obligation Created",
    "2.04": "Triggering Event Accelerating Debt",
    "2.05": "Exit/Disposal Costs",
    "2.06": "Material Impairment",
    "5.02": "Departure/Election of Directors or Officers",
    "5.03": "Amendments to Articles/Bylaws",
    "7.01": "Regulation FD Disclosure (guidance)",
    "8.01": "Other Events",
    "9.01": "Financial Statements/Exhibits",
}


@tool
@safe_tool
def get_form_8k_events(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Classify the most recent 8-K current reports by event type (earnings,
    executive change, M&A, debt default, guidance, impairment).

    Near-term catalyst and event-risk signal.
    """
    fs = _latest_filing(ticker, {"8-K"}, curr_date, max_filings=8)
    if not fs:
        return f"No recent 8-K filings found for {ticker} on SEC EDGAR."
    lines = [
        f"# Recent 8-K Events for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    for entry in fs:
        filing = entry["filing"]
        items_found = sorted(set(re.findall(r"item\s+(\d+\.\d+)", entry["text"], re.IGNORECASE)))
        descs = [f"{i} ({_8K_ITEM_DESCRIPTIONS.get(i, 'other')})" for i in items_found[:6]]
        lines.append(f"- {filing['filing_date']}: {', '.join(descs) or '(items not parsed)'}")
    lines.append(
        "\nNote: item codes parsed from the filing text; see the full 8-K for detail."
    )
    return "\n".join(lines)


_FORM4_CODES = {
    "P": "Open-market purchase",
    "S": "Open-market sale",
    "A": "Award (grant)",
    "M": "Option exercise",
    "X": "Option exercise",
    "G": "Gift",
    "F": "Tax withholding",
    "J": "Other acquisition/disposition",
}


@tool
@safe_tool
def get_insider_form4_activity(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Classify recent insider Form 4 transactions by transaction code
    (P purchase, S sale, A award, M/X option exercise, F tax withholding).

    Distinguishes discretionary open-market buying/selling from routine
    grants and exercises — a sharper insider signal than raw transaction
    counts.
    """
    t = yf.Ticker(ticker.upper())
    df = yf_retry(lambda: t.insider_transactions)
    if df is None or getattr(df, "empty", True):
        return f"No insider transaction data available for {ticker}."
    lines = [
        f"# Insider Form 4 Activity for {ticker.upper()}",
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    # Identify a code column if present.
    code_col = None
    for c in df.columns:
        if str(c).lower() in ("trans_code", "transactioncode", "transaction_type", "transactiontype", "type"):
            code_col = c
            break
    shown = df.head(20)
    if code_col is not None:
        lines.append("Date | Insider | Code | Meaning | Shares | Value")
        for _, row in shown.iterrows():
            code = str(row[code_col]).strip().upper()[:1]
            lines.append(
                f"{row.get('Transaction Date', row.get('StartDate', ''))} | "
                f"{row.get('Filer Name', row.get('filerName', 'N/A'))} | "
                f"{code} | {_FORM4_CODES.get(code, 'other')} | "
                f"{row.get('Shares Traded', row.get('sharesTraded', 'N/A'))} | "
                f"{row.get('Value', row.get('value', 'N/A'))}"
            )
    else:
        lines.append("(Transaction code column not available; raw recent transactions below.)")
        lines.append(shown.to_csv())
    return "\n".join(lines)


@tool
@safe_tool
def get_prospectus_disclosure(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Extract 'Use of Proceeds' and 'Dilution' sections from the latest
    S-3 / 424B prospectus (follow-on offerings).

    Use this when the company has recently issued or registered equity/debt to
    assess dilution and use of capital.
    """
    fs = _latest_filing(
        ticker, {"S-3", "S-3/A", "424B3", "424B4", "424B5"}, curr_date, max_filings=1
    )
    if not fs:
        return f"No recent prospectus (S-3/424B) filing found for {ticker} on SEC EDGAR."
    patterns = [
        _C(r"use\s+of\s+proceeds"),
        _C(r"\bdilution\b"),
        _C(r"underwriting"),
        _C(r"principal\s+(?:amount|holders)\s+(?:of|selling)"),
    ]
    passages = _extract_keyword_passages(fs[0]["text"], patterns, context=1000, max_passages=6)
    return _format_passage_report(
        f"Prospectus Disclosure (S-3/424B) for {ticker.upper()}", fs[0]["filing"], passages,
    )
