"""Pre-flight tool health check for the analysis pipeline.

Before an analysis burns LLM tokens, this module exercises one
representative tool per selected analyst category with real (keyless /
vendor-routed) calls and reports which data paths are working or broken:

- Each check runs in parallel with a per-check timeout, so one hung
  source cannot stall startup.
- Results are tri-state: ``ok`` (tool returned data), ``warn`` (tool
  works but returned no data for this ticker — may be legitimate), and
  ``fail`` (exception, timeout, or ``[Tool Error]`` output from
  ``safe_tool``).
- The check is informational: the CLI proceeds unless *every* check
  failed, which almost certainly means there is no network at all.

The heavy tools (full transcript scrape, macro market snapshot,
regime-analog full-history download) are deliberately probed only in a
cheap mode or skipped so the whole suite stays quick (~seconds).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger("tradingagents.tools.health")

#: Per-check wall-clock budget (seconds). Checks run in parallel, so the
#: total startup delay is bounded by roughly this value.
DEFAULT_CHECK_TIMEOUT = 30.0


@dataclass
class ToolCheckResult:
    name: str
    category: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    seconds: float


def _classify_output(output) -> tuple[str, str]:
    """Map raw tool output to (status, detail)."""
    if output is None:
        return "fail", "tool returned None"
    text = " ".join(str(output).split()).strip()
    if not text:
        return "fail", "tool returned empty output"
    if text.startswith("[Tool Error]"):
        return "fail", text[:160]
    lowered = text.lower()
    no_data_markers = (
        "no results found",
        "no news found",
        "no data available",
        "no filings found",
        "no transcripts found",
        "not found",
        "unavailable for this ticker",
    )
    if any(m in lowered for m in no_data_markers) and len(text) < 400:
        return "warn", text[:160]
    return "ok", text[:160]


def _build_checks(ticker: str, analysts: list[str], trade_date: str) -> list:
    """Build (name, category, callable) checks for the selected analysts."""
    ticker = ticker.upper().strip()
    try:
        end = datetime.strptime(trade_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        end = datetime.now()
    start = (end - timedelta(days=35)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # Late imports so the module can be imported without pulling the
    # whole tool stack into lightweight contexts (e.g. tests).
    from tradingagents.agents.utils.agent_utils import (
        get_company_profile,
        get_fundamentals,
        get_fx_rates,
        get_news,
        get_relative_momentum_vs_sector,
        get_stock_data,
        get_dilution_profile,
    )
    from tradingagents.agents.utils.transcript_tools import _search_transcript_urls

    checks: list[tuple[str, str, callable]] = []

    if "market" in analysts:
        checks.append(
            (
                "get_stock_data",
                "market",
                lambda: get_stock_data.invoke(
                    {"symbol": ticker, "start_date": start, "end_date": end_str}
                ),
            )
        )
        checks.append(
            (
                "get_relative_momentum_vs_sector",
                "market",
                lambda: get_relative_momentum_vs_sector.invoke({"ticker": ticker}),
            )
        )
    if "social" in analysts or "news" in analysts:
        checks.append(
            (
                "get_news",
                "social/news",
                lambda: get_news.invoke(
                    {"ticker": ticker, "start_date": start, "end_date": end_str}
                ),
            )
        )
    if "fundamentals" in analysts:
        checks.append(
            (
                "get_fundamentals",
                "fundamentals",
                lambda: get_fundamentals.invoke(
                    {"ticker": ticker, "curr_date": end_str}
                ),
            )
        )
        checks.append(
            (
                "get_dilution_profile",
                "fundamentals",
                lambda: get_dilution_profile.invoke({"ticker": ticker}),
            )
        )
    if "macro" in analysts:
        checks.append(
            (
                "get_fx_rates",
                "macro",
                lambda: get_fx_rates.invoke({"pairs": "EURUSD,USDJPY"}),
            )
        )
    if "business" in analysts:
        checks.append(
            (
                "get_company_profile",
                "business",
                lambda: get_company_profile.invoke({"ticker": ticker}),
            )
        )
        # Cheap probe: URL resolution only (no page fetch / parse).
        checks.append(
            (
                "transcripts (url resolve)",
                "business",
                lambda: (
                    "Transcript URLs found: " + u
                    if (u := (_search_transcript_urls(ticker, 1) or [""])[0])
                    else "No transcripts found for this ticker"
                ),
            )
        )

    return checks


def run_tool_health_checks(
    ticker: str,
    analysts: list[str],
    trade_date: str | None = None,
    timeout: float = DEFAULT_CHECK_TIMEOUT,
) -> list[ToolCheckResult]:
    """Run all checks in parallel and return their results.

    Args:
        ticker: The ticker about to be analyzed (used in probe calls).
        analysts: Selected analyst category keys (market, news, ...).
        trade_date: Analysis date (yyyy-mm-dd); defaults to today.
        timeout: Per-check wall-clock budget in seconds.

    Returns:
        Results in the order the checks were defined; a check that
        exceeds ``timeout`` or raises is reported as ``fail``.
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    checks = _build_checks(ticker, analysts, trade_date)
    if not checks:
        return []

    results: list[ToolCheckResult | None] = [None] * len(checks)
    deadline = time.monotonic() + timeout

    with ThreadPoolExecutor(max_workers=min(8, len(checks))) as pool:
        futures = {
            pool.submit(fn): i for i, (_, _, fn) in enumerate(checks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            name, category, _ = checks[idx]
            t0 = time.monotonic()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results[idx] = ToolCheckResult(
                    name, category, "fail", "timed out", timeout
                )
                continue
            try:
                output = future.result(timeout=remaining)
                status, detail = _classify_output(output)
            except Exception as exc:
                status, detail = "fail", f"{type(exc).__name__}: {exc}"[:160]
            results[idx] = ToolCheckResult(
                name, category, status, detail, time.monotonic() - t0
            )

    # Anything never completed (shouldn't happen with as_completed, but
    # be defensive) becomes a timeout failure.
    final: list[ToolCheckResult] = []
    for idx, (name, category, _) in enumerate(checks):
        if results[idx] is None:
            final.append(
                ToolCheckResult(name, category, "fail", "timed out", timeout)
            )
        else:
            final.append(results[idx])
    return final


def summarize_tool_health(results: list[ToolCheckResult]) -> str:
    """One-line summary: counts plus the overall verdict."""
    if not results:
        return "No tool checks configured for the selected analysts."
    n_ok = sum(1 for r in results if r.status == "ok")
    n_warn = sum(1 for r in results if r.status == "warn")
    n_fail = sum(1 for r in results if r.status == "fail")
    if n_fail == len(results):
        verdict = "ALL TOOLS BROKEN — check your network/data-source config before running"
    elif n_fail:
        verdict = f"{n_fail} tool(s) broken — affected analyst reports may be degraded"
    elif n_warn:
        verdict = f"{n_warn} tool(s) returned no data for this ticker — may be legitimate"
    else:
        verdict = "all tools working"
    return f"Tool check: {n_ok} ok, {n_warn} no-data, {n_fail} broken — {verdict}"
