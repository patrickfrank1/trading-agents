"""Debate Referee: scores each bull/bear round for convergence and can end the debate early.

The original pipeline ran a fixed number of back-and-forth turns and then
handed straight to the Research Manager. Because the prompts told each side
only to "refute" and never to concede, extra rounds just produced louder
restatement. This referee runs after every complete Bull+Bear round and:

  1. Identifies which opponent points each side actually conceded vs. dodged.
  2. Detects when both sides are only recycling their prior arguments
     (no new evidence, no new concessions) — the convergence signal.
  3. Appends a short verdict to the debate history so debaters can see what
     was scored against them and adjust.

When the referee signals convergence, ``should_continue_debate`` short-circuits
to the fact-check / Research Manager instead of running the remaining rounds.
"""

from __future__ import annotations


def create_debate_referee(llm, enabled=True):
    def referee_node(state) -> dict:
        if not enabled:
            # Referee disabled: no LLM, no convergence signal. The round-cap
            # in ConditionalLogic._debate_done still bounds the debate.
            return {}
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        referee_notes = investment_debate_state.get("referee_notes", "")
        count = investment_debate_state["count"]

        prompt = f"""You are the Debate Referee. You just watched the latest round of a Bull vs. Bear investment debate. Your job is NOT to pick a side — it is to score the QUALITY of the dialectic.

Evaluate the most recent round only, against the full history:

**Full debate history:**
{history}

**Bull's prior arguments (so you can detect repetition):**
{bull_history}

**Bear's prior arguments (so you can detect repetition):**
{bear_history}

Answer in two short parts:

1. CONCESSIONS: List which opponent points each side explicitly conceded this round (if any), and which load-bearing points each side dodged or restated instead of answering.

2. CONVERGENCE: Has the debate converged? Convergence means BOTH:
   - Neither side introduced new evidence or a new line of argument this round, AND
   - Neither side conceded anything new; they are only restating prior points louder.
   If both are true, write "CONVERGED: yes". Otherwise write "CONVERGED: no".

Be terse. No more than ~120 words."""

        response = llm.invoke(prompt)
        verdict = response.content if hasattr(response, "content") else str(response)
        verdict = verdict.strip()

        converged = "converged: yes" in verdict.lower()

        note = f"\n[Referee — round {count // 2}]\n{verdict}"
        new_referee_notes = referee_notes + note
        new_history = history + note

        new_investment_debate_state = {
            "history": new_history,
            "bull_history": bull_history,
            "bear_history": bear_history,
            "current_response": investment_debate_state.get("current_response", ""),
            "judge_decision": investment_debate_state.get("judge_decision", ""),
            "count": count,
            "referee_notes": new_referee_notes,
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "debate_converged": converged,
        }

    return referee_node
