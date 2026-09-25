import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 15 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache uses a fixed window (15y to today) so one file per symbol
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        CUSTOM_INDICATORS = {"volume", "donchian_upper", "donchian_lower", "donchian_mid", "fibonacci"}

        if indicator in CUSTOM_INDICATORS:
            if indicator == "volume":
                pass  # volume column already exists in stockstats wrapped data
            elif indicator == "donchian_upper":
                df[indicator] = df["high"].rolling(window=20, min_periods=1).max()
            elif indicator == "donchian_lower":
                df[indicator] = df["low"].rolling(window=20, min_periods=1).min()
            elif indicator == "donchian_mid":
                upper = df["high"].rolling(window=20, min_periods=1).max()
                lower = df["low"].rolling(window=20, min_periods=1).min()
                df[indicator] = (upper + lower) / 2
            elif indicator == "fibonacci":
                rolling_high = df["high"].rolling(window=60, min_periods=2).max()
                rolling_low = df["low"].rolling(window=60, min_periods=2).min()
                diff = rolling_high - rolling_low
                df[indicator] = rolling_high - diff * 0.618  # store golden ratio level as representative value
        else:
            df[indicator]  # trigger stockstats to calculate the indicator

        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            if indicator == "fibonacci":
                high_val = rolling_high.loc[matching_rows.index[0]]
                low_val = rolling_low.loc[matching_rows.index[0]]
                if pd.isna(high_val) or pd.isna(low_val):
                    return "N/A"
                parts = [f"High:{high_val:.2f}", f"Low:{low_val:.2f}"]
                for level_name, level_pct in [("23.6%", 0.236), ("38.2%", 0.382), ("50.0%", 0.500), ("61.8%", 0.618), ("78.6%", 0.786)]:
                    val = high_val - (high_val - low_val) * level_pct
                    parts.append(f"{level_name}:{val:.2f}")
                return " | ".join(parts)
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
