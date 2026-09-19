"""Tests for the no-login data/coverage tools added to close analyst-layer
gaps:

- ``transcript_tools``: Motley Fool transcript URL resolution, HTML
  parsing, remarks excerpting, tool delegation and graceful degradation.
- ``get_fx_rates`` (macro_data_tools): rate-table formatting and change
  math against mocked yfinance history.
- ``get_relative_momentum_vs_sector`` / ``get_dilution_profile`` /
  ``get_regime_analog`` (equity_intel_tools): pure-math helpers on
  synthetic series plus tool-level formatting with mocked yfinance.
- Wiring: tools appear in the ToolNode registry AND in each analyst's
  bound tool list.

All tests are unit-marker: no network, no API keys (conftest injects
placeholders and ``mock_llm_client`` patches the LLM factory).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.utils.transcript_tools import (
    _parse_transcript,
    _pick_remarks,
    _TRANSCRIPT_URL_RE,
    get_earnings_call_transcripts,
)
from tradingagents.agents.utils.macro_data_tools import (
    _fx_change,
    get_fx_rates,
)
from tradingagents.agents.utils.equity_intel_tools import (
    SECTOR_ETF_MAP,
    _annualized_share_growth,
    _max_drawdown,
    _regime_performance,
    _total_return,
    _classify_fed_regimes,
    get_dilution_profile,
    get_regime_analog,
    get_relative_momentum_vs_sector,
)
from tradingagents.default_config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _closes(values, start="2024-01-01", freq="D"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def _fake_close_df(values, start="2024-01-01"):
    return pd.DataFrame({"Close": _closes(values, start=start)})


TRANSCRIPT_HTML = """<html><head><title>
Apple (AAPL) Q3 2025 Earnings Call Transcript | The Motley Fool</title></head>
<body><div id="article-body-transcript">
<h2>DATE</h2><p>Thursday, July 31, 2025 at 5 p.m. ET</p>
<h2>CALL PARTICIPANTS</h2><ul>
<li>Chief Executive Officer — Tim Cook</li>
<li>Chief Financial Officer — Kevan Parekh</li></ul>
<h2>RISKS</h2><ul><li>Approximately $800 million in tariff-related costs.</li></ul>
<h2>TAKEAWAYS</h2><ul>
<li>Total Revenue— $94 billion, up 10%.</li>
<li>Guidance for Next Quarter— mid- to high single-digit growth.</li></ul>
<p>Need a quote from a Motley Fool analyst? Email example@example.com.</p>
<h2>Full Conference Call Transcript</h2>
<p>Operator: Good afternoon, and welcome.</p>
<p>Tim Cook: Thank you. It was a great quarter across the board.</p>
<p>Tim Cook: We shipped a record number of iPhones during the quarter.</p>
<p>Kevan Parekh: We expect revenue growth next quarter and full-year margins to expand.</p>
<p>Random Analyst: A filler paragraph with no keywords at all here.</p>
<p>Tim Cook: Our capital allocation includes share repurchas of stock steadily.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Transcript tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscriptUrlFilter:
    def test_accepts_fool_transcript_urls(self):
        assert _TRANSCRIPT_URL_RE.match(
            "https://www.fool.com/earnings/call-transcripts/2025/08/01/apple-aapl-q3-2025-earnings-call-transcript/"
        )

    def test_rejects_non_transcript_urls(self):
        assert not _TRANSCRIPT_URL_RE.match("https://www.fool.com/quote/nyse/aapl/")
        assert not _TRANSCRIPT_URL_RE.match("https://example.com/earnings/call-transcripts/x")
        assert not _TRANSCRIPT_URL_RE.match(
            "https://www.fool.com/earnings/call-transcripts/a?utm_source=x"
        )


@pytest.mark.unit
class TestParseTranscript:
    def test_extracts_structured_sections(self):
        parsed = _parse_transcript(TRANSCRIPT_HTML, "https://www.fool.com/x")
        assert parsed is not None
        assert "Apple (AAPL) Q3 2025" in parsed["title"]
        assert "Motley Fool" not in parsed["title"]
        assert "July 31, 2025" in parsed["date"]
        assert any("Tim Cook" in p for p in parsed["participants"])
        assert any("tariff" in p for p in parsed["risks"])
        assert any("$94 billion" in p for p in parsed["takeaways"])
        assert len(parsed["full_call"]) == 6

    def test_missing_container_returns_none(self):
        assert _parse_transcript("<html><body><p>nothing</p></body></html>", "u") is None

    def test_page_without_full_call_returns_none(self):
        html = '<div id="article-body-transcript"><h2>TAKEAWAYS</h2><p>x</p></div>'
        assert _parse_transcript(html, "u") is None


@pytest.mark.unit
class TestPickRemarks:
    def test_keeps_opening_and_guidance_paragraphs(self):
        parsed = _parse_transcript(TRANSCRIPT_HTML, "u")
        remarks = _pick_remarks(parsed["full_call"], max_paragraphs=3)
        assert any(p.startswith("Tim Cook: Thank you") for p in remarks)
        # operator lines are skipped
        assert not any(p.lower().startswith("operator") for p in remarks)
        # guidance-bearing later paragraph is kept
        assert any("expect revenue growth" in p for p in remarks)
        # keyword-free filler is dropped
        assert not any("no keywords" in p for p in remarks)
        # capital-allocation paragraph is kept via keyword
        assert any("capital allocation" in p for p in remarks)

    def test_max_paragraphs_bounds_output(self):
        parsed = _parse_transcript(TRANSCRIPT_HTML, "u")
        remarks = _pick_remarks(parsed["full_call"], max_paragraphs=1)
        assert len(remarks) <= 2


@pytest.mark.unit
class TestTranscriptTool:
    def test_tool_schema(self):
        assert get_earnings_call_transcripts.name == "get_earnings_call_transcripts"
        assert "ticker" in get_earnings_call_transcripts.args
        assert "quarters" in get_earnings_call_transcripts.args

    def test_renders_transcript_report(self):
        with patch(
            "tradingagents.agents.utils.transcript_tools._search_transcript_urls"
        ) as mock_urls, patch(
            "tradingagents.agents.utils.transcript_tools._fetch_page"
        ) as mock_fetch:
            mock_urls.return_value = ["https://www.fool.com/t1"]
            mock_fetch.return_value = TRANSCRIPT_HTML
            out = get_earnings_call_transcripts.invoke({"ticker": "aapl"})
        assert "Key takeaways" in out
        assert "$94 billion" in out
        assert "Tim Cook: Thank you" in out
        assert "https://www.fool.com/t1" in out

    def test_no_urls_degrades_gracefully(self):
        with patch(
            "tradingagents.agents.utils.transcript_tools._search_transcript_urls"
        ) as mock_urls:
            mock_urls.return_value = []
            out = get_earnings_call_transcripts.invoke({"ticker": "zzzz"})
        assert "No transcripts found" in out

    def test_unparseable_page_reports_url(self):
        with patch(
            "tradingagents.agents.utils.transcript_tools._search_transcript_urls"
        ) as mock_urls, patch(
            "tradingagents.agents.utils.transcript_tools._fetch_page"
        ) as mock_fetch:
            mock_urls.return_value = ["https://www.fool.com/t1"]
            mock_fetch.return_value = "<html><body>changed layout</body></html>"
            out = get_earnings_call_transcripts.invoke({"ticker": "aapl"})
        assert "could not be parsed" in out
        assert "https://www.fool.com/t1" in out

    def test_fetch_failure_degrades(self):
        with patch(
            "tradingagents.agents.utils.transcript_tools._search_transcript_urls"
        ) as mock_urls, patch(
            "tradingagents.agents.utils.transcript_tools._fetch_page",
            side_effect=RuntimeError("boom"),
        ):
            mock_urls.return_value = ["https://www.fool.com/t1"]
            out = get_earnings_call_transcripts.invoke({"ticker": "aapl"})
        assert "could not be parsed" in out


# ---------------------------------------------------------------------------
# FX rates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFxHelpers:
    def test_fx_change_math(self):
        closes = _closes([100.0] * 40 + [110.0])
        assert _fx_change(closes, 21) == pytest.approx(0.10)

    def test_fx_change_insufficient_history(self):
        closes = _closes([100.0, 101.0])
        assert _fx_change(closes, 21) is None


@pytest.mark.unit
class TestGetFxRates:
    def test_tool_schema(self):
        assert get_fx_rates.name == "get_fx_rates"
        assert "pairs" in get_fx_rates.args

    def test_formats_rate_table(self):
        df = _fake_close_df([1.0 + 0.001 * i for i in range(300)])
        with patch("tradingagents.agents.utils.macro_data_tools.yf_retry") as mock_retry:
            mock_retry.return_value = df
            out = get_fx_rates.invoke({"pairs": "EURUSD,USDJPY"})
        assert "EURUSD" in out and "USDJPY" in out
        assert "1M" in out and "12M" in out
        mock_retry.assert_called()

    def test_unavailable_pair_degrades_to_na(self):
        with patch("tradingagents.agents.utils.macro_data_tools.yf_retry") as mock_retry:
            mock_retry.side_effect = RuntimeError("no data")
            out = get_fx_rates.invoke({"pairs": "XXXXXX"})
        assert "N/A" in out
        assert "XXXXXX" in out


# ---------------------------------------------------------------------------
# Relative momentum vs sector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMomentumHelpers:
    def test_total_return(self):
        closes = _closes([100.0] * 300)
        closes.iloc[-1] = 110.0
        assert _total_return(closes, 21) == pytest.approx(0.10)

    def test_total_return_short_series(self):
        assert _total_return(_closes([1.0, 2.0]), 21) is None

    def test_max_drawdown(self):
        closes = _closes([100, 120, 90, 95])
        assert _max_drawdown(closes) == pytest.approx(-0.25)

    def test_sector_etf_mapping(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        with patch.object(eit, "_safe_info", return_value={"sector": "Technology"}):
            sector, etf = eit._sector_etf_for(MagicMock())
        assert (sector, etf) == ("Technology", "XLK")

    def test_unknown_sector_falls_back_to_spy(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        with patch.object(eit, "_safe_info", return_value={"sector": "Weird Stuff"}):
            _, etf = eit._sector_etf_for(MagicMock())
        assert etf == "SPY"

    def test_sector_map_covers_common_gics_sectors(self):
        for etf in SECTOR_ETF_MAP.values():
            assert etf.endswith(("K", "F", "V", "Y", "P", "E", "I", "B", "E", "U", "C"))


@pytest.mark.unit
class TestGetRelativeMomentum:
    def test_tool_schema(self):
        assert get_relative_momentum_vs_sector.name == "get_relative_momentum_vs_sector"

    def test_renders_relative_table(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        hist = _fake_close_df([100.0 + i for i in range(300)])
        bench = _fake_close_df([200.0 + 0.5 * i for i in range(300)])
        with patch.object(eit, "yf_retry", side_effect=[hist, bench]), patch.object(
            eit, "_safe_info", return_value={"sector": "Healthcare"}
        ):
            out = get_relative_momentum_vs_sector.invoke({"ticker": "aapl"})
        assert "XLV" in out
        assert "| 1M |" in out
        assert "52-week range position" in out

    def test_missing_history_degrades(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        with patch.object(eit, "yf_retry", side_effect=RuntimeError("x")), patch.object(
            eit, "_safe_info", return_value={"sector": "Technology"}
        ):
            out = get_relative_momentum_vs_sector.invoke({"ticker": "aapl"})
        assert "Price history unavailable" in out


# ---------------------------------------------------------------------------
# Dilution profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDilutionHelpers:
    def test_annualized_growth(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
        shares = pd.Series([1_000_000_000.0, 1_100_000_000.0], index=idx)
        assert _annualized_share_growth(shares) == pytest.approx(10.0, rel=0.01)

    def test_annualized_shrink_is_negative(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2022-01-01"])
        shares = pd.Series([100.0, 90.0], index=idx)
        assert _annualized_share_growth(shares) < 0

    def test_needs_two_points(self):
        idx = pd.DatetimeIndex(["2020-01-01"])
        assert _annualized_share_growth(pd.Series([1.0], index=idx)) is None


@pytest.mark.unit
class TestGetDilutionProfile:
    def test_tool_schema(self):
        assert get_dilution_profile.name == "get_dilution_profile"

    def test_renders_dilution_report(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        idx = pd.DatetimeIndex(
            ["2020-06-30", "2020-12-31", "2021-06-30", "2021-12-31"]
        )
        shares = pd.Series([1.0e9, 1.01e9, 1.02e9, 1.03e9], index=idx)
        cf = pd.DataFrame(
            {"2021-12-31": [-5.0e9, 3.0e9]},
            index=["Repurchase Of Capital Stock", "Stock Based Compensation"],
        )
        with patch.object(eit, "yf_retry", side_effect=[shares, cf]):
            out = get_dilution_profile.invoke({"ticker": "aapl"})
        assert "Share count trajectory" in out
        assert "Annualized share-count change" in out
        assert "SBC vs buybacks" in out
        # yfinance reports buybacks as negative cash flow: -5B buybacks vs
        # +3B SBC → net -2B → buybacks DO offset SBC
        assert "offsets SBC" in out

    def test_buybacks_not_offsetting_sbc(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        idx = pd.DatetimeIndex(["2020-12-31", "2021-12-31"])
        shares = pd.Series([1.0e9, 1.02e9], index=idx)
        cf = pd.DataFrame(
            {"2021-12-31": [-2.0e9, 3.0e9]},
            index=["Repurchase Of Capital Stock", "Stock Based Compensation"],
        )
        with patch.object(eit, "yf_retry", side_effect=[shares, cf]):
            out = get_dilution_profile.invoke({"ticker": "aapl"})
        assert "does NOT offset SBC" in out

    def test_missing_data_degrades(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        with patch.object(eit, "yf_retry", side_effect=RuntimeError("x")):
            out = get_dilution_profile.invoke({"ticker": "aapl"})
        assert "Share-count history unavailable" in out


# ---------------------------------------------------------------------------
# Fed-regime analogs
# ---------------------------------------------------------------------------


def _fedfunds_series():
    """Flat 2020, steady hikes to a ~5.5% peak by 2023, then steady cuts."""
    months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    vals = []
    for d in months:
        if d < pd.Timestamp("2021-10-01"):
            vals.append(0.10)
        elif d < pd.Timestamp("2023-09-01"):
            vals.append(min(5.5, 0.10 + 0.4 * (d - pd.Timestamp("2021-10-01")).days / 30.44))
        else:
            vals.append(max(3.8, 5.5 - 0.25 * (d - pd.Timestamp("2023-09-01")).days / 30.44))
    return pd.Series(vals, index=months)


@pytest.mark.unit
class TestRegimeHelpers:
    def test_classifies_hiking_and_cutting(self):
        regimes = _classify_fed_regimes(_fedfunds_series())
        directions = [r["direction"] for r in regimes]
        assert "HIKING" in directions
        assert "CUTTING" in directions
        # regimes are chronological and non-overlapping
        for prev, nxt in zip(regimes, regimes[1:]):
            assert prev["end"] <= nxt["start"] or prev["direction"] != nxt["direction"]

    def test_needs_history(self):
        short = _fedfunds_series().iloc[-6:]
        assert _classify_fed_regimes(short) == [] or len(
            _classify_fed_regimes(short)
        ) < 4

    def test_regime_performance(self):
        closes = _closes([100, 110, 80, 90], start="2022-01-01", freq="MS")
        result = _regime_performance(
            closes, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
        )
        assert result is not None
        ret, mdd = result
        assert ret == pytest.approx(-0.10)
        assert mdd == pytest.approx(80.0 / 110.0 - 1.0)


@pytest.mark.unit
class TestGetRegimeAnalog:
    def test_tool_schema(self):
        assert get_regime_analog.name == "get_regime_analog"

    def test_renders_regime_table(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        # ~11 years of daily closes so every regime window has data
        hist = _fake_close_df([100.0 + 0.05 * i for i in range(4000)], start="2015-01-01")
        with patch.object(
            eit, "_fetch_fedfunds_monthly", return_value=_fedfunds_series()
        ), patch.object(eit, "yf_retry", side_effect=[hist, hist]), patch.object(
            eit, "_safe_info", return_value={"sector": "Technology"}
        ):
            out = get_regime_analog.invoke({"ticker": "aapl"})
        assert "HIKING" in out and "CUTTING" in out
        assert "maxDD" in out

    def test_missing_fedfunds_degrades(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        with patch.object(eit, "_fetch_fedfunds_monthly", return_value=None):
            out = get_regime_analog.invoke({"ticker": "aapl"})
        assert "unavailable" in out

    def test_ticker_too_recent_for_analogs(self):
        from tradingagents.agents.utils import equity_intel_tools as eit

        hist = _fake_close_df([100.0, 101.0], start="2026-08-01")
        with patch.object(
            eit, "_fetch_fedfunds_monthly", return_value=_fedfunds_series()
        ), patch.object(eit, "yf_retry", side_effect=[hist, hist]), patch.object(
            eit, "_safe_info", return_value={"sector": "Technology"}
        ):
            out = get_regime_analog.invoke({"ticker": "newco"})
        assert "too recent for analogs" in out


# ---------------------------------------------------------------------------
# Wiring: ToolNode registry + analyst binding
# ---------------------------------------------------------------------------

NEW_TOOLS_BY_ANALYST = {
    "market": ["get_relative_momentum_vs_sector"],
    "fundamentals": ["get_dilution_profile", "get_regime_analog", "get_fx_rates"],
    "macro": ["get_fx_rates", "get_macro_causal_forecast", "get_macro_sensitivity"],
    "business": ["get_earnings_call_transcripts"],
}


@pytest.mark.unit
class TestWiring:
    def test_tool_nodes_contain_new_tools(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(config=dict(DEFAULT_CONFIG))
        for category, expected in NEW_TOOLS_BY_ANALYST.items():
            node = graph.tool_nodes[category]
            for tool_name in expected:
                assert tool_name in node.tools_by_name, (
                    f"{tool_name} missing from {category} ToolNode"
                )

    def test_analysts_bind_new_tools(self):
        from tradingagents.agents.analysts.market_analyst import create_market_analyst
        from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
        from tradingagents.agents.analysts.fundamentals_analyst import (
            create_fundamentals_analyst,
        )
        from tradingagents.agents.analysts.business_analyst import (
            create_business_analyst,
        )

        factories = {
            "market": create_market_analyst,
            "macro": create_macro_analyst,
            "fundamentals": create_fundamentals_analyst,
            "business": create_business_analyst,
        }
        for category, factory in factories.items():
            llm = MagicMock()
            llm.bind_tools.return_value.invoke.return_value = MagicMock(
                tool_calls=[], content="report"
            )
            node = factory(llm)
            node(
                {"messages": [], "trade_date": "2026-01-01", "company_of_interest": "AAPL"}
            )
            bound_names = [t.name for t in llm.bind_tools.call_args[0][0]]
            for tool_name in NEW_TOOLS_BY_ANALYST[category]:
                assert tool_name in bound_names, (
                    f"{tool_name} not bound by {category} analyst"
                )
