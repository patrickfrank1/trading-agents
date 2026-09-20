"""Tests for the Jev (TypeSafe System One) Portfolio Manager decision tool."""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.utils.jev import (
    JEV_PRECEDENCE_CONFIDENCE,
    JevAssessment,
    assess_with_jev,
    extract_current_price,
    jev_takes_precedence,
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

    def test_facts_snapshot_wins_over_reports(self):
        state = {
            "facts_snapshot": "Price: $43.07 (close 2026-09-11).",
            "fundamentals_report": "Current Price: $44.00.",
        }
        assert extract_current_price(state) == 43.07

    def test_does_not_mistake_last_close_date_for_price(self):
        state = {"facts_snapshot": "Price: $495.63 (last close 2026-09-11)."}
        assert extract_current_price(state) == 495.63

    def test_hong_kong_and_euro_prices(self):
        assert extract_current_price(
            {"facts_snapshot": "Price: HK$75.10 (last close 2026-09-11; used 2026-09-12)."}
        ) == 75.1
        assert extract_current_price(
            {"facts_snapshot": "Price: €48.70 (2026-09-10 close)."}
        ) == 48.7

    def test_currency_codes_and_no_space(self):
        assert extract_current_price(
            {"facts_snapshot": "Price: NOK 179.40. USD conversion $19.34."}
        ) == 179.4
        assert extract_current_price(
            {"facts_snapshot": "Current price: DKK157.00 (12-Sep reports)."}
        ) == 157.0

    def test_pence_suffix(self):
        assert extract_current_price(
            {"facts_snapshot": "GKP.L, 2026-09-12 (last close 194.00p, 2026-09-11)"}
        ) == 194.0

    def test_bare_number_when_currency_is_in_header(self):
        state = {"facts_snapshot": "Roche, CHF\nCurrent: 344.20 (2026-09-11 close)."}
        assert extract_current_price(state) == 344.2

    def test_price_line_beats_softer_current_hint(self):
        state = {
            "facts_snapshot": (
                "Price: 571.60 SEK last close 2026-09-11. Stale: 578.00.\n"
                "Balance sheet: Current 1.23; quick 0.76; BVPS vs ~SEK 93 derived."
            )
        }
        assert extract_current_price(state) == 571.6

    def test_excludes_market_cap_magnitudes(self):
        assert extract_current_price({"facts_snapshot": "Market cap $190.3B."}) is None

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

    def _run_freetext(self, monkeypatch, jev_assessment, content):
        from tradingagents.agents.managers import portfolio_manager as pm_module

        monkeypatch.setattr(pm_module, "assess_with_jev", lambda state, **kw: jev_assessment)
        monkeypatch.setattr(
            pm_module, "build_instrument_context", lambda ticker: f"Instrument {ticker}"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("no structured output")
        llm.invoke.return_value = MagicMock(content=content)
        node = create_portfolio_manager(llm)
        return node(_make_pm_state())["final_trade_decision"]

    @staticmethod
    def _decision():
        return PortfolioDecision(
            arguments_table="| a | b | c | d |",
            weighted_score=-50.0,
            scenario_table="| s | p | t | d |",
            trade_ticket="Exit now.",
            rating=PortfolioRating.SELL,
            executive_summary="Sell.",
            investment_thesis="Bearish.",
        )

    @staticmethod
    def _jev(confidence):
        return JevAssessment(
            rating=PortfolioRating.OVERWEIGHT,
            allocation_pct=3.5,
            intrinsic_value=115.0,
            intrinsic_value_low=95.0,
            intrinsic_value_high=135.0,
            confidence=confidence,
            model="jev-test",
        )

    @pytest.mark.unit
    def test_high_confidence_jev_takes_precedence_over_pm(self, monkeypatch):
        result, captured = self._run(monkeypatch, self._jev(0.98), self._decision())
        final = result["final_trade_decision"]

        # Jev's rating overrides the LLM's Sell, and both judgments are recorded.
        assert "**Rating**: Overweight" in final
        assert "Jev Decision Tool" in final
        assert "3.5% of NAV" in final
        assert "**Decision Reconciliation**" in final
        assert "Jev's rating takes precedence" in final
        assert "Portfolio Manager's own rating: **Sell**" in final
        assert "takes precedence" in captured["prompt"]

    @pytest.mark.unit
    def test_low_confidence_jev_is_advisory_and_pm_rating_stands(self, monkeypatch):
        result, captured = self._run(monkeypatch, self._jev(0.80), self._decision())
        final = result["final_trade_decision"]

        # Below threshold Jev is advisory; the PM's own Sell rating stands.
        assert "**Rating**: Sell" in final
        assert "Jev Decision Tool" in final
        assert "**Decision Reconciliation**" in final
        assert "portfolio manager reconciled" in final
        assert "advisory only" in captured["prompt"]

    @pytest.mark.unit
    def test_no_jev_leaves_llm_rating_untouched(self, monkeypatch):
        result, captured = self._run(monkeypatch, None, self._decision())
        final = result["final_trade_decision"]

        assert "**Rating**: Sell" in final
        assert "Jev Decision Tool" not in final
        assert "Decision Reconciliation" not in final
        assert "Jev" not in captured["prompt"]

    @pytest.mark.unit
    def test_freetext_fallback_recovers_pm_own_rating(self, monkeypatch):
        # No structured output -> the mutate hook never runs; the reconciliation
        # must still report the PM's own prose rating instead of "n/a".
        final = self._run_freetext(
            monkeypatch, self._jev(0.80), "**Rating**: Underweight\n\nBearish."
        )

        assert "Portfolio Manager's own rating: **Underweight**" in final
        assert "**Underweight** stands" in final
        assert "rating **n/a** stands" not in final

    @pytest.mark.unit
    def test_freetext_fallback_without_parseable_rating(self, monkeypatch):
        final = self._run_freetext(monkeypatch, self._jev(0.80), "No rating stated here.")

        assert "Portfolio Manager's own rating: **n/a**" in final
        assert "could not" in final

    @pytest.mark.unit
    def test_confidence_exactly_at_threshold_does_not_take_precedence(self):
        assessment = self._jev(JEV_PRECEDENCE_CONFIDENCE)
        assert jev_takes_precedence(assessment) is False

    @pytest.mark.unit
    def test_confidence_above_threshold_takes_precedence(self):
        assert jev_takes_precedence(self._jev(0.951)) is True
        assert jev_takes_precedence(None) is False
