"""Risk Debate Referee: scores each Aggressive/Conservative/Neutral round for
convergence and can end the risk debate early.

The bull/bear debate already had a mid-debate referee; the risk debate did
not, so it ran a fixed number of back-and-forth turns even after all three
analysts were only restating their prior positions (the failure mode visible
in long ORCL reports where Aggressive/Conservative/Neutral each re-proposed
nearly identical collar structures 4-5 times).

This referee runs once per complete round (after Neutral speaks) and:

  1. Identifies which opponent points each analyst conceded vs. dodged.
  2. Detects when all three are recycling prior arguments with no new
     evidence and no new concessions — the convergence signal.
  3. Appends a short verdict to the debate history so the next round can
     adjust.

When the referee signals convergence, ``should_continue_after_risk_referee``
short-circuits to the Portfolio Manager instead of running the remaining
rounds.
"""

from __future__ import annotations


def create_risk_debate_referee(llm, enabled=True):
    def referee_node(state) -> dict:
        if not enabled:
            return {}
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")
        referee_notes = risk_debate_state.get("referee_notes", "")
        count = risk_debate_state["count"]

        prompt = f"""You are the Risk Debate Referee. You just watched the latest round of a three-way Aggressive vs. Conservative vs. Neutral risk debate. Your job is NOT to pick a winner — it is to score the QUALITY of the dialectic.

Evaluate the most recent round only, against the full history:

**Full risk-debate history:**
{history}

**Aggressive's prior arguments (so you can detect repetition):**
{aggressive_history}

**Conservative's prior arguments:**
{conservative_history}

**Neutral's prior arguments:**
{neutral_history}

Answer in two short parts:

1. CONCESSIONS: List which opponent points each analyst explicitly conceded this round (if any), and which load-bearing points each dodged or restated instead of answering.

2. CONVERGENCE: Has the risk debate converged? Convergence means ALL THREE:
   - No analyst introduced new evidence or a new line of argument this round, AND
   - No analyst conceded anything new; they are only restating prior positions (e.g. re-proposing nearly identical position structures with tweaked strikes).
   If all three are true, write "CONVERGED: yes". Otherwise write "CONVERGED: no".

Be terse. No more than ~120 words."""

        response = llm.invoke(prompt)
        verdict = response.content if hasattr(response, "content") else str(response)
        verdict = verdict.strip()

        converged = "converged: yes" in verdict.lower()

        round_num = count // 3 if count else 1
        note = f"\n[Risk Referee — round {round_num}]\n{verdict}"
        new_referee_notes = referee_notes + note
        new_history = history + note

        new_risk_debate_state = {
            "history": new_history,
            "aggressive_history": aggressive_history,
            "conservative_history": conservative_history,
            "neutral_history": neutral_history,
            "latest_speaker": risk_debate_state.get("latest_speaker", ""),
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "judge_decision": risk_debate_state.get("judge_decision", ""),
            "count": count,
            "referee_notes": new_referee_notes,
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "risk_debate_converged": converged,
        }

    return referee_node
