"""Tests for the Jev (TypeSafe System One) Portfolio Manager decision tool."""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.utils.jev import (
    JevAssessment,
    assess_with_jev,
    extract_current_price,
    render_jev_assessment,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Answer:
    def __init__(self, choice=None, confidence=None):
        self.choice = choice
        self.confidence = confidence


class _Response:
    def __init__(self, answers, model="jev-test"):
        self.answers = answers
        self.model = model


class _FakeClient:
    def __init__(self, answers=None, exc=None):
        self._answers = answers or {}
        self._exc = exc
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self._exc is not None:
            raise self._exc
        return _Response(self._answers)


def _answers(rating="Overweight", allocation="2-5% of NAV", value=None, low=None, high=None):
    payload = {
        "rating": _Answer(choice=rating, confidence=0.8),
        "allocation": _Answer(choice=allocation),
    }
    if value is not None:
        payload["intrinsic_value"] = _Answer(choice=value)
        payload["intrinsic_value_low"] = _Answer(choice=low)
        payload["intrinsic_value_high"] = _Answer(choice=high)
    return payload


_PRICED_STATE = {
    "company_of_interest": "TEST",
    "trade_date": "2026-01-01",
    "facts_snapshot": "Current price: $100.00. P/E 20x.",
}


# ---------------------------------------------------------------------------
# Price extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractCurrentPrice:
    def test_explicit_current_price(self):
        assert extract_current_price(_PRICED_STATE) == 100.0

    def test_generic_dollar_amount(self):
        assert extract_current_price({"market_report": "Shares trade at $42.50 today."}) == 42.5

    def test_returns_none_when_absent(self):
        assert extract_current_price({"facts_snapshot": "No numbers here."}) is None


# ---------------------------------------------------------------------------
# Assessment mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssessWithJev:
    def test_maps_bands_allocation_and_rating(self):
        client = _FakeClient(
            _answers(
                rating="Overweight",
                allocation="2-5% of NAV",
                value="1.10x-1.20x of current price",
                low="0.90x-1.00x of current price",
                high="1.30x-1.40x of current price",
            )
        )
        assessment = assess_with_jev(_PRICED_STATE, client=client)

        assert assessment is not None
        assert assessment.rating == PortfolioRating.OVERWEIGHT
        assert assessment.allocation_pct == 3.5
        assert assessment.intrinsic_value == 115.0
        assert assessment.intrinsic_value_low == 95.0
        assert assessment.intrinsic_value_high == 135.0
        assert assessment.confidence == 0.8
        assert assessment.model == "jev-test"

    def test_interval_widens_to_contain_point_estimate(self):
        client = _FakeClient(
            _answers(
                value="1.10x-1.20x of current price",
                low="1.30x-1.40x of current price",
                high="0.60x-0.70x of current price",
            )
        )
        assessment = assess_with_jev(_PRICED_STATE, client=client)
        assert assessment.intrinsic_value == 115.0
        assert assessment.intrinsic_value_low == 115.0
        assert assessment.intrinsic_value_high == 115.0

    def test_sell_forces_zero_allocation(self):
        client = _FakeClient(_answers(rating="Sell", allocation="10-15% of NAV"))
        assessment = assess_with_jev(_PRICED_STATE, client=client)
        assert assessment.rating == PortfolioRating.SELL
        assert assessment.allocation_pct == 0.0

    def test_underweight_caps_allocation(self):
        client = _FakeClient(_answers(rating="Underweight", allocation="10-15% of NAV"))
        assessment = assess_with_jev(_PRICED_STATE, client=client)
        assert assessment.allocation_pct == 2.5

    def test_value_questions_skipped_without_price(self):
        client = _FakeClient(_answers())
        state = {"company_of_interest": "TEST", "facts_snapshot": "No price given."}
        assessment = assess_with_jev(state, client=client)

        questions = client.calls[0][1]
        assert "rating" in questions
        assert "allocation" in questions
        assert "intrinsic_value" not in questions
        assert assessment.intrinsic_value is None

    def test_failure_returns_none(self):
        client = _FakeClient(exc=RuntimeError("boom"))
        assert assess_with_jev(_PRICED_STATE, client=client) is None

    def test_missing_api_key_skips_without_client(self):
        assert assess_with_jev(_PRICED_STATE) is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderJevAssessment:
    def test_renders_all_fields(self):
        assessment = JevAssessment(
            rating=PortfolioRating.OVERWEIGHT,
            allocation_pct=3.5,
            intrinsic_value=115.0,
            intrinsic_value_low=95.0,
            intrinsic_value_high=135.0,
            confidence=0.8,
            model="jev-test",
        )
        text = render_jev_assessment(assessment)
        assert "**Overweight**" in text
        assert "3.5% of NAV" in text
        assert "115.00" in text and "95.00" in text and "135.00" in text
        assert "confidence 80%" in text

    def test_renders_unavailable_value(self):
        assessment = JevAssessment(rating=PortfolioRating.HOLD, allocation_pct=0.0)
        text = render_jev_assessment(assessment)
        assert "unavailable" in text


# ---------------------------------------------------------------------------
# Portfolio Manager integration
# ---------------------------------------------------------------------------


def _make_pm_state():
    return {
        "company_of_interest": "TEST",
        "past_context": "",
        "risk_debate_state": {
            "history": "Risk debate history.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "facts_snapshot": "Current price: $100.00.",
        "investment_plan": "Research plan.",
        "trader_investment_plan": "Trader plan.",
    }


def _structured_pm_llm(captured, decision):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestPortfolioManagerJevIntegration:
    def _run(self, monkeypatch, jev_assessment, decision):
        from tradingagents.agents.managers import portfolio_manager as pm_module

        monkeypatch.setattr(pm_module, "assess_with_jev", lambda state, **kw: jev_assessment)
        monkeypatch.setattr(
            pm_module, "build_instrument_context", lambda ticker: f"Instrument {ticker}"
        )
        captured = {}
        llm = _structured_pm_llm(captured, decision)
        node = create_portfolio_manager(llm)
        return node(_make_pm_state()), captured

    @pytest.mark.unit
    def test_jev_rating_overrides_llm_and_block_is_appended(self, monkeypatch):
        jev = JevAssessment(
            rating=PortfolioRating.OVERWEIGHT,
            allocation_pct=3.5,
            intrinsic_value=115.0,
            intrinsic_value_low=95.0,
            intrinsic_value_high=135.0,
            confidence=0.8,
            model="jev-test",
        )
        decision = PortfolioDecision(
            arguments_table="| a | b | c | d |",
            weighted_score=-50.0,
            scenario_table="| s | p | t | d |",
            trade_ticket="Exit now.",
            rating=PortfolioRating.SELL,
            executive_summary="Sell.",
            investment_thesis="Bearish.",
        )
        result, captured = self._run(monkeypatch, jev, decision)
        final = result["final_trade_decision"]

        assert "**Rating**: Overweight" in final
        assert "Jev Decision Tool" in final
        assert "3.5% of NAV" in final
        assert "authoritative" in captured["prompt"]

    @pytest.mark.unit
    def test_no_jev_leaves_llm_rating_untouched(self, monkeypatch):
        decision = PortfolioDecision(
            arguments_table="| a | b | c | d |",
            weighted_score=-50.0,
            scenario_table="| s | p | t | d |",
            trade_ticket="Exit now.",
            rating=PortfolioRating.SELL,
            executive_summary="Sell.",
            investment_thesis="Bearish.",
        )
        result, captured = self._run(monkeypatch, None, decision)
        final = result["final_trade_decision"]

        assert "**Rating**: Sell" in final
        assert "Jev Decision Tool" not in final
        assert "authoritative" not in captured["prompt"]
