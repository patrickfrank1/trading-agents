"""Facts Snapshot node: compile one canonical facts block from the analyst reports.

Without this, every downstream agent re-derives (and slightly mis-copies) the
same key figures from the prose reports — which is how a single analysis ends
up citing four different "current prices" across sections. This node runs once,
right after the analysts finish, and produces a short reconciled snapshot
(price, 52-week range, key multiples, balance-sheet / cashflow headlines, RPO,
growth, etc.) that every subsequent agent reads as the single source of truth.
"""

from __future__ import annotations


def create_facts_snapshot(llm, enabled=True):
    def facts_snapshot_node(state) -> dict:
        if not enabled:
            return {"facts_snapshot": ""}
        reports = _collect_reports(state)
        if not reports.strip():
            return {"facts_snapshot": ""}

        prompt = f"""You are a fact-checking analyst. Below are several analyst reports for {state.get("company_of_interest", "the company")} on {state.get("trade_date", "")}. Extract a SINGLE canonical facts block: the key quantitative figures every downstream agent must agree on.

Resolve any conflicts between reports (e.g. if two reports cite different current prices, pick the most recent / authoritative and note the reconciliation). Output a compact, well-structured block covering at least:
- Current price, 52-week range, and recent drawdown/return
- Key valuation multiples (P/E trailing & forward, PEG, EV/EBITDA if available)
- Revenue growth rate and margin trend
- Balance-sheet headlines: total debt, net debt, D/E, interest coverage, tangible book value
- Cashflow headlines: capex, free cash flow
- Order-book / RPO if mentioned
- Any other load-bearing figure (off-balance-sheet commitments, founder ownership, etc.)

Rules:
- State ONLY figures that appear in the reports below. Do not invent or estimate.
- If a figure is missing or disputed across reports, write "DISPUTED: <what conflicts>" so downstream agents know it is unresolved.
- Keep it under ~250 words. No recommendations, no narrative — just the agreed numbers.

Analyst reports:
{reports}"""

        response = llm.invoke(prompt)
        snapshot = response.content if hasattr(response, "content") else str(response)
        return {"facts_snapshot": snapshot.strip()}

    return facts_snapshot_node


def _collect_reports(state: dict) -> str:
    parts = []
    for key, label in (
        ("market_report", "Market Analyst"),
        ("fundamentals_report", "Fundamentals Analyst"),
        ("macro_report", "Macro Analyst"),
        ("business_report", "Business Analyst"),
        ("news_report", "News Analyst"),
        ("sentiment_report", "Sentiment Analyst"),
    ):
        text = state.get(key, "")
        if text:
            parts.append(f"--- {label} ---\n{text}")
    return "\n\n".join(parts)
