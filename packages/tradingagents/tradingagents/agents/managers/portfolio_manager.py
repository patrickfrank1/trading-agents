"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.

Before deciding, the PM always queries the Jev decision tool
(``agents.utils.jev``) with the collected facts and analyst reports. Jev
returns an intrinsic-value estimate with a low/high interval, a target
allocation, and a 5-tier rating. Both judgments are recorded in the decision.
Reconciliation is confidence-gated: when Jev's calibrated confidence exceeds
``JEV_PRECEDENCE_CONFIDENCE`` (0.95) Jev's rating and allocation take
precedence; otherwise the portfolio manager reconciles the two views and its
own rating stands. When Jev is unavailable the existing LLM-only path runs
unchanged.
"""

from __future__ import annotations

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    render_pm_decision,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_claim_audit_block,
    get_facts_block,
    get_language_instruction,
    get_report_hygiene_instruction,
    get_reports_digest,
)
from tradingagents.agents.utils.jev import (
    JEV_PRECEDENCE_CONFIDENCE,
    assess_with_jev,
    jev_takes_precedence,
    render_jev_assessment,
)
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def _render_reconciliation(assessment, pm_rating, precedence: bool) -> str:
    """Record both judgments and how they were reconciled in the decision."""
    jev_rating = assessment.rating.value
    confidence = (
        f"{assessment.confidence:.0%}" if assessment.confidence is not None else "unknown"
    )
    threshold = f"{JEV_PRECEDENCE_CONFIDENCE:.0%}"
    pm = pm_rating.value if pm_rating is not None else "n/a"
    if precedence and pm_rating is not None:
        outcome = (
            f"Jev's confidence ({confidence}) exceeds the {threshold} threshold, so "
            f"**Jev's rating takes precedence** over the portfolio manager's own **{pm}**."
        )
    elif precedence:
        outcome = (
            f"Jev's confidence ({confidence}) exceeds the {threshold} threshold, so "
            "**Jev's rating takes precedence** over the portfolio manager's own view."
        )
    elif pm_rating is None:
        outcome = (
            f"Jev's confidence ({confidence}) is at or below the {threshold} threshold, "
            "so Jev is advisory only and the portfolio manager's own rating could not "
            "be determined from its output."
        )
    else:
        outcome = (
            f"Jev's confidence ({confidence}) is at or below the {threshold} threshold, "
            f"so **the portfolio manager reconciled the two views** and its rating "
            f"**{pm}** stands."
        )
    return (
        "**Decision Reconciliation**\n"
        f"- Portfolio Manager's own rating: **{pm}**\n"
        f"- Jev rating: **{jev_rating}** (confidence {confidence}; "
        f"precedence threshold {threshold})\n"
        f"- {outcome}"
    )


def create_portfolio_manager(
    llm,
    jev_enabled: bool = True,
    jev_model: str = "jev-latest",
    jev_max_state_chars: int = 24000,
):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        facts_block = get_facts_block(state)
        reports_digest = get_reports_digest(state)
        claim_audit_block = get_claim_audit_block(state)

        # Mandatory Jev decision tool: hand Jev the collected facts and analyst
        # reports, then use its intrinsic-value / allocation / rating output to
        # drive the final decision. Degrades to the LLM-only path on failure.
        jev_assessment = None
        if jev_enabled:
            jev_assessment = assess_with_jev(
                state,
                model=jev_model,
                max_state_chars=jev_max_state_chars,
            )
        jev_block = render_jev_assessment(jev_assessment) if jev_assessment else ""
        jev_precedence = jev_takes_precedence(jev_assessment)

        if jev_block and jev_precedence:
            jev_instruction = (
                f"Jev answered the collected facts with {jev_assessment.confidence:.0%} "
                f"confidence, above the {JEV_PRECEDENCE_CONFIDENCE:.0%} precedence "
                f"threshold, so Jev's rating **{jev_assessment.rating.value}** takes "
                "precedence over the portfolio manager's own view: your final rating "
                "MUST be Jev's rating and your trade ticket MUST use Jev's target "
                "allocation. Explain Jev's intrinsic value and confidence interval, "
                "and note where your own weighted-score analysis differs.\n\n"
            )
        elif jev_block:
            confidence = (
                f"{jev_assessment.confidence:.0%}"
                if jev_assessment.confidence is not None
                else "unknown"
            )
            jev_instruction = (
                f"Jev answered the collected facts with {confidence} confidence, below "
                f"the {JEV_PRECEDENCE_CONFIDENCE:.0%} precedence threshold, so Jev is "
                "advisory only. You must reconcile Jev's view with your own analysis "
                "and decide the final rating yourself. In your investment_thesis you "
                "MUST explicitly address Jev's rating, intrinsic value, and confidence, "
                "and explain how you reconciled them (agree or disagree, and why).\n\n"
            )
        else:
            jev_instruction = ""

        jev_context = (
            "**Jev Decision Tool output (an independent judgment — reconcile it with your own):**\n"
            f"{jev_block}\n"
            if jev_block
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

{jev_context}{jev_instruction}**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Analyst Weighting Priority** — used to assign impact ratings (same priority as the research team):
- Business Analyst — Highest priority: competitive moat, management execution, product strategy, and long-term business value.
- Fundamentals Analyst — Core financial analysis: profitability, valuation, balance sheet strength, and financial health.
- Sector Specialist — Sector-specific dynamics: industry structure, regulation, cyclicality, and the sector's key value drivers.
- Macro Analyst — Macroeconomic context: Fed policy, inflation, labor markets, and geopolitical factors.
- Market Analyst — Technical indicators and price action as an ENTRY-TIMING GATE only — NOT for the directional thesis itself, and NOT a source of exit levels. Exits must be justified by fundamentals or thesis invalidation.
- News Analyst — Recent news flow, material events, and catalysts, but do not let news override fundamentals.

**How to use the inputs below:**
- The raw Business and Fundamentals reports are primary evidence alongside the risk debate. The debate is a *filter*, not a substitute.
- The Canonical Facts Snapshot is the single source of truth for numbers; do not re-derive them.
- The Claim Audit lists debate claims flagged as unsupported/contradicted by the source reports — discount them when assigning impact.

**Step 1 — Arguments Table:**
First, compile a markdown table of the key BUY and SELL arguments extracted from the risk debate and supporting context below. Each row must include: the argument, its source analyst type, an impact rating (High / Medium / Low), and whether it supports BUY or SELL. Arguments sourced from Business Analyst, Fundamentals Analyst, or Sector Specialist should generally carry High impact; arguments from Macro, Market, or News Analysts should generally carry Medium or Low impact. This table anchors your final decision in transparent, weighted evidence.

**Step 2 — Weighted Score:**
Compute a single net score from -100 to +100 by weighing the arguments table (High = ±20, Medium = ±10, Low = ±5; positive for BUY, negative for SELL; clamp to [-100, +100]). The score must be auditable: a reader tallying the table should arrive at the same number. Map score to rating band (>=+40 Buy, +15..+39 Overweight, -14..+14 Hold, -39..-15 Underweight, <=-40 Sell). If your final rating falls outside the band implied by the score, you MUST explain the override in the investment thesis — silent overrides are not allowed.

**Step 3 — Probability-Weighted Scenario Table:**
Produce a three-row (Bull / Base / Bear) markdown table with explicit probability weights (summing to 100%), a price target per scenario, and the one-sentence driver. Compute and show the probability-weighted expected price. Scenario price targets MUST be derived from fundamentals (earnings trajectories, margin scenarios, justified multiples on normalized earnings) — do NOT anchor them on sell-side price targets, options open-interest strikes, or 52-week highs/lows. This forces an explicit view on any binary / "show-me" outcomes instead of leaving them implicit in prose.

**Step 4 — Trade Ticket:**
Resolve the risk team's conflicting sizing and hedge proposals into ONE executable plan. Specify: action (consistent with the rating), position size as % of portfolio, entry/exit levels, hedge structure with concrete strikes and approximate premium (do NOT write "consider a collar" — write "buy 6-mo $110 put / sell $150 call, ~$0 net debit"), and named exit triggers. For a Hold with no new capital, state size = 0% for new capital and give the maintenance plan for existing holders.

**Step 5 — Investment Thesis & Final Decision:**
Provide a concise summary of the key drivers behind your final decision, then deliver the final rating and supporting details. The rating must be consistent with the weighted score band unless an override is explicitly justified.

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{facts_block}
{reports_digest}
{claim_audit_block}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}{get_report_hygiene_instruction()}"""

        pm_own_rating: list = [None]

        def _apply_jev(decision: PortfolioDecision) -> PortfolioDecision:
            # Keep the portfolio manager's own call for the reconciliation
            # record, then apply Jev only when its confidence clears the
            # precedence threshold; otherwise the PM's rating stands.
            pm_own_rating[0] = decision.rating
            if jev_precedence and jev_assessment is not None:
                decision.rating = jev_assessment.rating
            return decision

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
            mutate=_apply_jev,
        )
        if pm_own_rating[0] is None and not jev_precedence:
            # The structured path was unavailable or failed, so ``mutate`` never
            # ran and we have no typed rating. Recover the PM's own call from its
            # prose deterministically so the reconciliation does not report "n/a".
            # Skip this when Jev takes precedence: the free-text PM was told to
            # adopt Jev's rating, so its prose is not an independent judgment.
            parsed = parse_rating(final_trade_decision, default="")
            if parsed in {r.value for r in PortfolioRating}:
                pm_own_rating[0] = PortfolioRating(parsed)
        if jev_block:
            final_trade_decision = (
                f"{final_trade_decision}\n\n{jev_block}\n\n"
                f"{_render_reconciliation(jev_assessment, pm_own_rating[0], jev_precedence)}"
            )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
            "referee_notes": risk_debate_state.get("referee_notes", ""),
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
