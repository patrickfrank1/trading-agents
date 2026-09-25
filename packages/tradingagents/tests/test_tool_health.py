"""Tests for the pre-flight tool health check (``tool_health``).

Covers:
- ``_classify_output``: ok / warn (no-data) / fail ([Tool Error], empty,
  None) classification.
- ``_build_checks``: representative tools per selected analyst category;
  nothing built for no analysts; late imports resolve.
- ``run_tool_health_checks``: parallel execution, success mapping,
  exception → fail, per-check timeout → fail, result ordering.
- ``summarize_tool_health``: verdict lines incl. the all-broken case.

All unit-marker: no network (check callables are monkeypatched).
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from tradingagents.dataflows.tool_health import (
    ToolCheckResult,
    _build_checks,
    _classify_output,
    run_tool_health_checks,
    summarize_tool_health,
)


@pytest.mark.unit
class TestClassifyOutput:
    def test_normal_output_is_ok(self):
        status, _ = _classify_output("Date,Close\n2025-01-02,100")
        assert status == "ok"

    def test_tool_error_prefix_is_fail(self):
        status, detail = _classify_output("[Tool Error] get_news: boom")
        assert status == "fail"
        assert "boom" in detail

    def test_empty_and_none_are_fail(self):
        assert _classify_output("")[0] == "fail"
        assert _classify_output(None)[0] == "fail"

    def test_no_data_markers_are_warn(self):
        status, _ = _classify_output("No results found for: XYZ")
        assert status == "warn"

    def test_long_output_with_marker_is_still_ok(self):
        # A "not found" phrase buried in a real report should not warn
        status, _ = _classify_output(
            "Revenue not found in guidance section. " + "x" * 500
        )
        assert status == "ok"


@pytest.mark.unit
class TestBuildChecks:
    def test_check_per_selected_category(self):
        checks = _build_checks("AAPL", ["market", "news", "macro"], "2025-06-02")
        names = [name for name, _, _ in checks]
        assert "get_stock_data" in names
        assert "get_relative_momentum_vs_sector" in names
        assert "get_news" in names
        assert "get_fx_rates" in names

    def test_fundamentals_and_business_checks(self):
        checks = _build_checks("AAPL", ["fundamentals", "business"], "2025-06-02")
        names = [name for name, _, _ in checks]
        assert "get_fundamentals" in names
        assert "get_dilution_profile" in names
        assert "get_company_profile" in names
        assert any("transcript" in n for n in names)

    def test_no_analysts_no_checks(self):
        assert _build_checks("AAPL", [], "2025-06-02") == []

    def test_check_callables_are_invokable(self):
        checks = _build_checks("AAPL", ["market"], "2025-06-02")
        assert all(callable(fn) for _, _, fn in checks)


@pytest.mark.unit
class TestRunToolHealthChecks:
    def _run_with_stub(self, behavior, analysts, timeout=5.0):
        """Run checks with every check callable stubbed to *behavior*."""
        with patch(
            "tradingagents.dataflows.tool_health._build_checks"
        ) as mock_build:
            mock_build.return_value = [
                (name, "test", fn) for name, fn in behavior.items()
            ]
            return run_tool_health_checks(
                "AAPL", analysts, trade_date="2025-06-02", timeout=timeout
            )

    def test_ok_result(self):
        results = self._run_with_stub(
            {"tool_a": lambda: "some data"}, analysts=["market"]
        )
        assert len(results) == 1
        assert results[0].status == "ok"
        assert results[0].name == "tool_a"
        assert results[0].seconds >= 0

    def test_exception_is_fail(self):
        results = self._run_with_stub(
            {"tool_a": lambda: 1 / 0}, analysts=["market"]
        )
        assert results[0].status == "fail"
        assert "ZeroDivisionError" in results[0].detail

    def test_timeout_is_fail(self):
        import time

        def slow():
            time.sleep(2.0)

        results = self._run_with_stub({"tool_a": slow}, analysts=["market"], timeout=0.2)
        assert results[0].status == "fail"
        assert "timed out" in results[0].detail

    def test_warn_classification_passes_through(self):
        results = self._run_with_stub(
            {"tool_a": lambda: "No results found for: XYZ"}, analysts=["market"]
        )
        assert results[0].status == "warn"

    def test_results_preserve_check_order(self):
        def slow_b():
            import time

            time.sleep(0.15)
            return "data b"

        results = self._run_with_stub(
            {"tool_a": lambda: "data a", "tool_b": slow_b},
            analysts=["market"],
        )
        # tool_a finishes first but a's result stays at index 0
        assert [r.name for r in results] == ["tool_a", "tool_b"]

    def test_parallel_execution_bounded_by_slowest(self):
        import time

        start = time.monotonic()
        results = self._run_with_stub(
            {
                "tool_a": lambda: (time.sleep(0.4), "a")[1],
                "tool_b": lambda: (time.sleep(0.4), "b")[1],
                "tool_c": lambda: (time.sleep(0.4), "c")[1],
            },
            analysts=["market"],
        )
        elapsed = time.monotonic() - start
        assert all(r.status == "ok" for r in results)
        assert elapsed < 1.0  # serial would be >= 1.2s

    def test_uses_thread_pool(self):
        # Sanity: the function must accept the same args regardless of
        # executor internals.
        with patch(
            "tradingagents.dataflows.tool_health._build_checks"
        ) as mock_build:
            mock_build.return_value = []
            assert run_tool_health_checks("AAPL", ["market"]) == []
            mock_build.assert_called_once()


@pytest.mark.unit
class TestSummarizeToolHealth:
    def _results(self, statuses):
        return [
            ToolCheckResult(f"tool_{i}", "cat", s, "", 0.1)
            for i, s in enumerate(statuses)
        ]

    def test_all_ok(self):
        text = summarize_tool_health(self._results(["ok", "ok"]))
        assert "all tools working" in text
        assert "2 ok" in text

    def test_partial_failures(self):
        text = summarize_tool_health(self._results(["ok", "fail", "warn"]))
        assert "1 tool(s) broken" in text

    def test_all_broken(self):
        text = summarize_tool_health(self._results(["fail", "fail"]))
        assert "ALL TOOLS BROKEN" in text

    def test_warn_only(self):
        text = summarize_tool_health(self._results(["ok", "warn"]))
        assert "no data" in text

    def test_empty(self):
        assert summarize_tool_health([]) != ""
