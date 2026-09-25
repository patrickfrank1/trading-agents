"""Fact Check node: audit the debate's claims against the source analyst reports.

After the bull/bear debate ends, this node walks through the load-bearing
factual claims each side made and checks each one against the original analyst
reports (and the canonical facts snapshot). Claims that are unsupported,
exaggerated, or directly contradicted get flagged in ``claim_audit``, which is
injected into the Research Manager and Portfolio Manager prompts so they can
discount rhetorical hot air rather than treating every debater assertion as
fact.

It also emits a RECONCILIATION_NEEDED marker so the graph can route to the
Fact Reconciliation node when a flagged contradiction could be resolved by
re-querying raw fundamentals data (e.g. "debt maturities out to 2066",
"interest coverage 4.5x").
"""

from __future__ import annotations


def create_fact_check(llm, enabled=True):
    def fact_check_node(state) -> dict:
        if not enabled:
            return {"claim_audit": ""}
        history = state["investment_debate_state"].get("history", "")
        facts_snapshot = state.get("facts_snapshot", "")
        reports = _collect_reports(state)

        prompt = f"""You are a fact-checker auditing an investment debate. Below are (a) the debate transcript and (b) the source analyst reports plus a canonical facts snapshot. Your job: find every load-bearing quantitative or factual claim made in the debate and verify it against the sources.

**Debate transcript:**
{history}

**Canonical facts snapshot:**
{facts_snapshot}

**Source analyst reports:**
{reports}

For each load-bearing claim, output a line in exactly this form:
- CLAIM: "<the claim>" — VERDICT: <Supported | Unsupported | Contradicted | Unverifiable> — NOTE: <one sentence; if Contradicted, quote the conflicting source figure>

Focus on claims that would change the decision if wrong: valuations, debt levels, cash flow, growth rates, RPO composition, interest coverage, maturity profiles, analogies to other companies, macro statistics. Ignore pure rhetoric.

After the claim lines, add a final line:
RECONCILIATION_NEEDED: <yes|no>
Set "yes" ONLY if at least one Contradicted/Unverifiable claim could be resolved by re-querying raw fundamentals/balance-sheet/cashflow data (e.g. debt maturities, interest coverage, off-balance-sheet commitments). Otherwise "no".

If there are no load-bearing claims to audit, write "No load-bearing claims flagged." and "RECONCILIATION_NEEDED: no"."""

        response = llm.invoke(prompt)
        audit = response.content if hasattr(response, "content") else str(response)
        return {"claim_audit": audit.strip()}

    return fact_check_node


def _collect_reports(state: dict) -> str:
    parts = []
    for key, label in (
        ("fundamentals_report", "Fundamentals Analyst"),
        ("business_report", "Business Analyst"),
        ("market_report", "Market Analyst"),
        ("macro_report", "Macro Analyst"),
    ):
        text = state.get(key, "")
        if text:
            parts.append(f"--- {label} ---\n{text}")
    return "\n\n".join(parts) if parts else "(no source reports available)"
