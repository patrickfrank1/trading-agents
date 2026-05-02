"""Unit tests for the DCF analysis engine."""

import math
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.utils.dcf import (
    DCFScenarioResult,
    _compute_cagr,
    _estimate_cost_of_debt,
    _estimate_tax_rate,
    _estimate_wacc,
    _fetch_risk_free_rate,
    _find_row,
    _fmt,
    _latest,
    _safe_get,
    _to_clean_list,
    compute_single_dcf,
    run_three_scenario_dcf,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestSafeGet:
    def test_returns_value_when_present(self):
        assert _safe_get({"a": 1}, "a") == 1

    def test_returns_default_when_missing(self):
        assert _safe_get({}, "a", 42) == 42

    def test_returns_default_for_nan(self):
        assert _safe_get({"a": float("nan")}, "a", 42) == 42

    def test_returns_default_for_none(self):
        assert _safe_get({"a": None}, "a", 42) == 42


class TestToCleanList:
    def test_filters_nones_and_nans(self):
        assert _to_clean_list([1.0, None, float("nan"), 3.0]) == [1.0, 3.0]

    def test_empty(self):
        assert _to_clean_list([]) == []


class TestLatest:
    def test_returns_first_non_none(self):
        assert _latest([None, 5, 10]) == 5

    def test_returns_zero_for_all_none(self):
        assert _latest([None, None]) == 0.0


class TestFmt:
    def test_billions(self):
        assert _fmt(3.5e9) == "$3.50B"

    def test_millions(self):
        assert _fmt(500e6) == "$500.00M"

    def test_thousands(self):
        assert _fmt(12345) == "$12.3K"

    def test_small(self):
        assert _fmt(42.5) == "$42.50"

    def test_nan(self):
        assert _fmt(float("nan")) == "N/A"


class TestComputeCAGR:
    def test_positive_growth(self):
        assert _compute_cagr([121, 110, 100]) == pytest.approx(0.10, rel=1e-4)

    def test_single_value_returns_none(self):
        assert _compute_cagr([100]) is None

    def test_empty_returns_none(self):
        assert _compute_cagr([]) is None

    def test_negative_values_filtered(self):
        assert _compute_cagr([110, 100, -50]) == pytest.approx(0.10, rel=1e-4)


class TestEstimateTaxRate:
    def test_from_historical(self):
        h = {"tax_provision": [20], "pretax_income": [100]}
        assert _estimate_tax_rate(h) == pytest.approx(0.20)

    def test_fallback(self):
        assert _estimate_tax_rate({}) == pytest.approx(0.21)


class TestEstimateCostOfDebt:
    def test_from_historical(self):
        h = {"interest_expense": [5], "total_debt": [100]}
        assert _estimate_cost_of_debt(h) == pytest.approx(0.05)

    def test_fallback(self):
        assert _estimate_cost_of_debt({}) == pytest.approx(0.06)


class TestEstimateWACC:
    def _make_info(self, market_cap=1e9, beta=1.0):
        return {"marketCap": market_cap, "beta": beta}

    def test_basic(self):
        info = self._make_info()
        hist = {"total_debt": [2e8], "interest_expense": [1.2e7], "tax_provision": [40], "pretax_income": [100]}
        wacc, coe, cod, tax, we, wd = _estimate_wacc(info, hist, 0.045, 0.055, 0.0)
        assert 0 < wacc < 0.3
        assert coe > 0
        assert cod > 0

    def test_no_market_cap_uses_defaults(self):
        info = {"marketCap": 0, "beta": 1.0}
        wacc, *_ = _estimate_wacc(info, {}, 0.045, 0.055, 0.0)
        assert wacc == pytest.approx(0.10)

    def test_beta_adjustment(self):
        info = self._make_info(beta=1.5)
        wacc_high, *_ = _estimate_wacc(info, {}, 0.045, 0.055, 0.3)
        wacc_low, *_ = _estimate_wacc(info, {}, 0.045, 0.055, -0.3)
        assert wacc_high > wacc_low


# ---------------------------------------------------------------------------
# Integration-style tests (mocked yfinance)
# ---------------------------------------------------------------------------

def _make_mock_ticker(**kwargs):
    """Build a mock yf.Ticker with controlled financial data."""
    ticker = MagicMock()
    ticker.info = {
        "longName": "TestCo",
        "marketCap": 1e9,
        "beta": 1.2,
        "sharesOutstanding": 1e8,
        "currentPrice": 10.0,
        "freeCashflow": 5e7,
        "totalRevenue": 2e8,
        "operatingMargins": 0.20,
        **kwargs,
    }

    import pandas as pd

    ticker.cashflow = pd.DataFrame(
        {"2023": [8e7, -3e7], "2022": [7e7, -2.5e7], "2021": [6e7, -2e7]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    ticker.income_stmt = pd.DataFrame(
        {
            "2023": [2e8, 4e7, 1e6, 8e6, 4e7],
            "2022": [1.8e8, 3.5e7, 1e6, 7e6, 3.5e7],
            "2021": [1.5e8, 3e7, 1e6, 6e6, 3e7],
        },
        index=["Total Revenue", "Operating Income", "Interest Expense", "Tax Provision", "Pretax Income"],
    )
    ticker.balance_sheet = pd.DataFrame(
        {
            "2023": [2e8, 5e7, 1e8],
            "2022": [1.8e8, 4e7, 9e7],
            "2021": [1.5e8, 3e7, 8e7],
        },
        index=["Long Term Debt", "Short Term Debt", "Cash And Cash Equivalents"],
    )
    return ticker


class TestComputeSingleDCF:
    @patch("tradingagents.agents.utils.dcf.yf")
    def test_basic_valuation(self, mock_yf):
        mock_yf.Ticker.return_value = _make_mock_ticker()

        result = compute_single_dcf(
            "TEST", "Base Case",
            revenue_growth_rate=0.05,
            terminal_growth_rate=0.025,
        )

        assert isinstance(result, DCFScenarioResult)
        assert result.base_fcf > 0
        assert result.enterprise_value > 0
        assert result.equity_value > 0
        assert result.fair_value_per_share > 0
        assert result.projection_years == 5
        assert len(result.projected_fcfs) == 5

    @patch("tradingagents.agents.utils.dcf.yf")
    def test_upside_calculation(self, mock_yf):
        mock_yf.Ticker.return_value = _make_mock_ticker()

        result = compute_single_dcf(
            "TEST", "Base Case",
            revenue_growth_rate=0.05,
            terminal_growth_rate=0.025,
        )

        expected_upside = ((result.fair_value_per_share / result.current_price) - 1) * 100
        assert result.upside_pct == pytest.approx(expected_upside, rel=1e-6)

    @patch("tradingagents.agents.utils.dcf.yf")
    def test_pessimistic_lower_valuation(self, mock_yf):
        mock_yf.Ticker.return_value = _make_mock_ticker()

        base = compute_single_dcf(
            "TEST", "Base", revenue_growth_rate=0.05, terminal_growth_rate=0.025,
        )
        pess = compute_single_dcf(
            "TEST", "Pessimistic", revenue_growth_rate=0.00, terminal_growth_rate=0.015,
            beta_adjustment=0.2, equity_risk_premium=0.07,
        )

        assert pess.enterprise_value < base.enterprise_value

    @patch("tradingagents.agents.utils.dcf.yf")
    def test_optimistic_higher_valuation(self, mock_yf):
        mock_yf.Ticker.return_value = _make_mock_ticker()

        base = compute_single_dcf(
            "TEST", "Base", revenue_growth_rate=0.05, terminal_growth_rate=0.025,
        )
        opt = compute_single_dcf(
            "TEST", "Optimistic", revenue_growth_rate=0.10, terminal_growth_rate=0.035,
            beta_adjustment=-0.2, equity_risk_premium=0.04,
        )

        assert opt.enterprise_value > base.enterprise_value


class TestThreeScenarioDCF:
    @patch("tradingagents.agents.utils.dcf.yf")
    def test_returns_markdown(self, mock_yf):
        mock_yf.Ticker.return_value = _make_mock_ticker()

        report = run_three_scenario_dcf("TEST")

        assert "Discounted Cash Flow" in report
        assert "Pessimistic" in report
        assert "Base Case" in report
        assert "Optimistic" in report
        assert "Fair Value" in report
        assert "WACC" in report

    @patch("tradingagents.agents.utils.dcf.yf")
    def test_scenarios_ordered(self, mock_yf):
        mock_yf.Ticker.return_value = _make_mock_ticker()

        report = run_three_scenario_dcf("TEST")

        fv_section = report[report.index("### Fair Value"):]
        pess_pos = fv_section.index("Pessimistic Case")
        base_pos = fv_section.index("Base Case")
        opt_pos = fv_section.index("Optimistic Case")
        assert pess_pos < base_pos < opt_pos


class TestFindRow:
    def test_exact_match(self):
        import pandas as pd
        df = pd.DataFrame({"val": [1]}, index=["Revenue"])
        row = _find_row(df, ["Revenue"])
        assert row is not None
        assert row.iloc[0] == 1

    def test_case_insensitive(self):
        import pandas as pd
        df = pd.DataFrame({"val": [1]}, index=["Total Revenue"])
        row = _find_row(df, ["total revenue"])
        assert row is not None

    def test_no_match(self):
        import pandas as pd
        df = pd.DataFrame({"val": [1]}, index=["Revenue"])
        assert _find_row(df, ["CostOfGoods"]) is None
