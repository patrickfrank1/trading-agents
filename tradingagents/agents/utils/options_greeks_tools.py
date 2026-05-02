import math
from typing import Annotated

import numpy as np
import yfinance as yf

from langchain_core.tools import tool

from tradingagents.dataflows.stockstats_utils import yf_retry


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(d1_val: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1_val - sigma * math.sqrt(T)


def _black_scholes_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    d1_val = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return _norm_cdf(d1_val)
    return _norm_cdf(d1_val) - 1.0


def _black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1_val = _d1(S, K, T, r, sigma)
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    return _norm_pdf(d1_val) / (S * sigma * math.sqrt(T))


def _nearest_strikes(current_price: float, all_strikes: list, n: int = 5) -> list:
    sorted_strikes = sorted(
        all_strikes,
        key=lambda k: abs(k - current_price),
    )
    return sorted_strikes[:n]


@tool
def get_option_greeks(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    Retrieve option Greeks (delta and gamma) for near-the-money call and put
    options using Black-Scholes with implied volatility from the options chain.

    Returns delta and gamma for approximately 5 nearest-to-money strikes for
    the nearest expiration date.

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
    Returns:
        str: A formatted report with delta and gamma values per strike.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        current_price = yf_retry(lambda: ticker.history(period="1d"))

        if current_price.empty:
            return f"No price data available for {symbol}."

        S = float(current_price["Close"].iloc[-1])

        options_chain = yf_retry(lambda: ticker.options)
        if not options_chain:
            return f"No options chain data available for {symbol}."

        nearest_expiry = options_chain[0]
        opt = yf_retry(lambda: ticker.option_chain(nearest_expiry))

        calls_df = opt.calls[["strike", "impliedVolatility"]].dropna()
        puts_df = opt.puts[["strike", "impliedVolatility"]].dropna()

        all_strikes = sorted(set(calls_df["strike"].tolist() + puts_df["strike"].tolist()))
        if not all_strikes:
            return f"No strike prices available for {symbol}."

        selected_strikes = _nearest_strikes(S, all_strikes, n=5)
        calls_iv = dict(zip(calls_df["strike"], calls_df["impliedVolatility"]))
        puts_iv = dict(zip(puts_df["strike"], puts_df["impliedVolatility"]))

        r = 0.05
        from datetime import datetime as _dt
        try:
            expiry_dt = _dt.strptime(nearest_expiry, "%Y-%m-%d")
            curr_dt = _dt.strptime(curr_date, "%Y-%m-%d")
            T = max((expiry_dt - curr_dt).days / 365.0, 1.0 / 365.0)
        except (ValueError, TypeError):
            T = 30.0 / 365.0

        lines = [
            f"# Option Greeks for {symbol.upper()}",
            f"Current Price: {S:.2f}",
            f"Expiration: {nearest_expiry}  (T = {T:.4f} years)",
            f"Risk-Free Rate: {r:.1%}\n",
            f"{'Strike':>10} | {'Call IV':>8} | {'Call Delta':>10} | {'Put IV':>8} | {'Put Delta':>10} | {'Gamma':>10}",
            "-" * 72,
        ]

        for K in selected_strikes:
            call_iv = float(calls_iv.get(K, 0))
            put_iv = float(puts_iv.get(K, 0))
            sigma_call = call_iv if call_iv > 0 else 0.30
            sigma_put = put_iv if put_iv > 0 else 0.30
            sigma_avg = (sigma_call + sigma_put) / 2.0 if sigma_call > 0 and sigma_put > 0 else max(sigma_call, sigma_put)

            call_delta = _black_scholes_delta(S, K, T, r, sigma_call, "call")
            put_delta = _black_scholes_delta(S, K, T, r, sigma_put, "put")
            gamma = _black_scholes_gamma(S, K, T, r, sigma_avg)

            lines.append(
                f"{K:>10.2f} | {call_iv:>8.2%} | {call_delta:>+10.4f} | {put_iv:>8.2%} | {put_delta:>+10.4f} | {gamma:>10.6f}"
            )

        lines.append("")
        lines.append("Delta: Rate of change of option price per $1 move in the underlying.")
        lines.append("Gamma: Rate of change of delta per $1 move in the underlying.")
        lines.append("Call delta range [0, 1]; Put delta range [-1, 0]; Gamma is always positive.")

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving option Greeks for {symbol}: {str(e)}"
