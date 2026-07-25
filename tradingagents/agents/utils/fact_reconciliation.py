"""Fact Reconciliation node: resolve flagged contradictions by re-querying raw data.

When the fact-check flags a material contradiction that can be resolved by
re-querying fundamentals data (e.g. "senior notes mature out to 2066" vs. the
Fundamentals report only saying "2029 to 2066"), this node fetches the raw
balance-sheet / cashflow / fundamentals data directly and asks the LLM to
answer the specific flagged questions against it. The resolved facts are
appended to the canonical facts snapshot so every downstream agent sees the
reconciliation.

This is the pipeline's targeted feedback loop: instead of freezing the
analyst reports at the top and letting a contradiction linger unresolved
through 1,400 lines of debate, the contradiction is answered with raw data
before any decision is made.
"""

from __future__ import annotations

import logging

from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)

logger = logging.getLogger(__name__)


def create_fact_reconciliation(llm):
    def reconciliation_node(state) -> dict:
        claim_audit = state.get("claim_audit", "")
        ticker = state.get("company_of_interest", "")
        trade_date = str(state.get("trade_date", ""))
        facts_snapshot = state.get("facts_snapshot", "")

        questions = _extract_questions(claim_audit)
        if not questions or not ticker:
            return {"facts_snapshot": facts_snapshot}

        raw_data = _fetch_raw(ticker, trade_date)
        if not raw_data.strip():
            logger.warning("Fact Reconciliation: no raw data returned for %s", ticker)
            return {"facts_snapshot": facts_snapshot}

        prompt = f"""You are reconciling factual contradictions flagged in an investment debate. Below are the specific unresolved questions and the RAW fundamentals data fetched directly from the data vendor. Answer each question using ONLY the raw data, then state the resolved fact with the source figure.

**Unresolved questions (from the claim audit):**
{questions}

**Raw fundamentals data (freshly fetched for {ticker}):**
{raw_data}

For each question, output:
- Q: <question>
- A: <Resolved: <fact> | Still unverifiable> — SOURCE: <the raw figure you used, quoted>

Be precise. Quote the exact numbers from the raw data. If the raw data does not contain the answer, say "Still unverifiable" — do not guess."""

        response = llm.invoke(prompt)
        reconciled = response.content if hasattr(response, "content") else str(response)
        reconciled = reconciled.strip()

        addition = (
            "\n\n--- Reconciled facts (from raw data re-query) ---\n" + reconciled
            if reconciled
            else ""
        )
        return {"facts_snapshot": (facts_snapshot + addition).strip()}

    return reconciliation_node


def _extract_questions(claim_audit: str) -> str:
    """Pull the Contradicted/Unverifiable claim lines out of the audit."""
    if not claim_audit:
        return ""
    lines = []
    for line in claim_audit.splitlines():
        low = line.lower()
        if "contradicted" in low or "unverifiable" in low:
            lines.append(line.strip("- ").strip())
    return "\n".join(lines)


def _fetch_raw(ticker: str, trade_date: str) -> str:
    """Fetch raw fundamentals data directly (no tool-calling loop needed)."""
    parts = []
    for label, tool in (
        ("Fundamentals", get_fundamentals),
        ("Balance Sheet", get_balance_sheet),
        ("Cash Flow", get_cashflow),
        ("Income Statement", get_income_statement),
    ):
        try:
            if tool is get_fundamentals:
                data = tool.invoke({"ticker": ticker, "curr_date": trade_date})
            else:
                data = tool.invoke(
                    {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}
                )
            if data:
                parts.append(f"--- {label} ---\n{data}")
        except Exception as exc:
            logger.warning("Fact Reconciliation: %s fetch failed for %s: %s", label, ticker, exc)
    return "\n\n".join(parts)
