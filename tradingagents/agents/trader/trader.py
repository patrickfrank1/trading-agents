"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        # The trader previously saw ONLY the Research Manager's plan, which made
        # it a rubber stamp. Give it the debate history, the canonical facts, and
        # the core analyst reports so it can sanity-check the plan against the
        # underlying evidence before proposing a transaction.
        debate_history = state.get("investment_debate_state", {}).get("history", "")
        from tradingagents.agents.utils.agent_utils import (
            get_facts_block,
            get_reports_digest,
        )
        facts_block = get_facts_block(state)
        reports_digest = get_reports_digest(state)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports, the research plan, and the "
                    "canonical facts snapshot. If the research plan's recommendation is not supported "
                    "by the underlying evidence, say so and propose what the evidence actually justifies."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision — but verify it against the debate and source reports below.\n\n"
                    f"Proposed Investment Plan: {investment_plan}\n\n"
                    f"{facts_block}"
                    f"{reports_digest}"
                    f"**Bull/Bear debate history (for context):**\n{debate_history}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
