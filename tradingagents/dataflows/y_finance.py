from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
import os
from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry, load_ohlcv, filter_financials_by_date

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # Price Channels
        "donchian_upper": (
            "Donchian Upper Channel: Highest high over a 20-period window. "
            "Usage: Identify breakout levels and resistance; price above the upper channel signals upward momentum. "
            "Tips: Best in trending markets; avoid in choppy/sideways conditions."
        ),
        "donchian_lower": (
            "Donchian Lower Channel: Lowest low over a 20-period window. "
            "Usage: Identify support levels and breakdown zones; price below the lower channel signals downward momentum. "
            "Tips: Combine with volume for confirmation."
        ),
        "donchian_mid": (
            "Donchian Mid-Channel: Average of the upper and lower Donchian channels. "
            "Usage: Serves as a trend-neutral reference level. "
            "Tips: Price above mid-channel suggests bullish bias, below suggests bearish bias."
        ),
        # Support/Resistance Levels
        "fibonacci": (
            "Fibonacci Retracement Levels: Key levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) calculated from the period's high and low. "
            "Usage: Identify potential support/resistance zones for pullback entries and trend reversals. "
            "Tips: The 61.8% (golden ratio) level is the most significant; zones where Fibonacci and other indicators align are strongest."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "volume": (
            "Volume: The actual number of shares traded per day — the only indicator not derived from price. "
            "Usage: Confirm the strength of price moves; high volume on breakouts validates the move, low volume signals weak participation. "
            "Tips: Compare to average volume to spot anomalies; volume precedes price."
        ),
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    CUSTOM_INDICATORS = {"volume", "donchian_upper", "donchian_lower", "donchian_mid", "fibonacci"}

    if indicator in CUSTOM_INDICATORS:
        if indicator == "volume":
            df["volume"] = df["close"]  # stockstats lowercases; raw volume is already present
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
            fib_levels = rolling_high - diff * pd.Series({
                0.236: 0.236, 0.382: 0.382, 0.500: 0.500, 0.618: 0.618, 0.786: 0.786,
            })
    else:
        df[indicator]  # This triggers stockstats to calculate the indicator

    result_dict = {}
    for idx, row in df.iterrows():
        date_str = row["Date"]

        if indicator == "fibonacci":
            high_val = rolling_high.iloc[idx]
            low_val = rolling_low.iloc[idx]
            if pd.isna(high_val) or pd.isna(low_val):
                result_dict[date_str] = "N/A"
            else:
                parts = [f"High:{high_val:.2f}", f"Low:{low_val:.2f}"]
                for level_name, level_pct in [("23.6%", 0.236), ("38.2%", 0.382), ("50.0%", 0.500), ("61.8%", 0.618), ("78.6%", 0.786)]:
                    val = high_val - (high_val - low_val) * level_pct
                    parts.append(f"{level_name}:{val:.2f}")
                result_dict[date_str] = " | ".join(parts)
        else:
            indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_income_stmt)
        else:
            data = yf_retry(lambda: ticker_obj.income_stmt)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.insider_transactions)
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"


def get_company_profile(
    ticker: Annotated[str, "ticker symbol of the company"],
):
    """Get detailed company profile and business model information from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            return f"No company profile data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Country", info.get("country")),
            ("City", info.get("city")),
            ("Website", info.get("website")),
            ("Employees", info.get("fullTimeEmployees")),
            ("Company Type", info.get("quoteType")),
            ("Currency", info.get("currency")),
            ("Exchange", info.get("exchange")),
            ("Market Cap", info.get("marketCap")),
            ("Enterprise Value", info.get("enterpriseValue")),
            ("Business Summary", info.get("longBusinessSummary")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit Margin", info.get("grossMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Net Profit Margin", info.get("profitMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Revenue Growth", info.get("revenueGrowth")),
            ("Earnings Growth", info.get("earningsGrowth")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Payout Ratio", info.get("payoutRatio")),
            ("Recommendation", info.get("recommendationKey")),
            ("Number of Analysts", info.get("numberOfAnalystOpinions")),
            ("Target High Price", info.get("targetHighPrice")),
            ("Target Low Price", info.get("targetLowPrice")),
            ("Target Mean Price", info.get("targetMeanPrice")),
            ("Target Median Price", info.get("targetMedianPrice")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("Current Price", info.get("currentPrice")),
            ("Trailing PE", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("Price to Sales", info.get("priceToSalesTrailing12Months")),
            ("Enterprise to Revenue", info.get("enterpriseToRevenue")),
            ("Enterprise to EBITDA", info.get("enterpriseToEbitda")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Quick Ratio", info.get("quickRatio")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Business Profile for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving company profile for {ticker}: {str(e)}"


def get_sector_performance(
    sector: Annotated[str, "sector name (e.g. Technology, Healthcare)"],
    period: Annotated[str, "time period: '1mo', '3mo', '6mo', '1y', 'ytd'"] = "1y",
):
    """Get sector performance data and compare against an ETF benchmark."""
    import pandas as pd

    SECTOR_ETF_MAP = {
        "technology": "XLK",
        "healthcare": "XLV",
        "financials": "XLF",
        "consumer discretionary": "XLY",
        "consumer staples": "XLP",
        "energy": "XLE",
        "industrials": "XLI",
        "materials": "XLB",
        "real estate": "XLRE",
        "utilities": "XLU",
        "communication services": "XLC",
    }

    sector_lower = sector.lower().strip()
    if sector_lower not in SECTOR_ETF_MAP:
        available = ", ".join(sorted(SECTOR_ETF_MAP.keys()))
        return (
            f"Sector '{sector}' not found. Available sectors: {available}. "
            f"Please choose from these GICS sectors."
        )

    etf = SECTOR_ETF_MAP[sector_lower]

    period_map = {
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "ytd": "ytd",
    }
    period_param = period_map.get(period.lower(), "1y")

    try:
        etf_ticker = yf.Ticker(etf)
        spy_ticker = yf.Ticker("SPY")

        etf_hist = yf_retry(lambda: etf_ticker.history(period=period_param))
        spy_hist = yf_retry(lambda: spy_ticker.history(period=period_param))

        lines = [f"# Sector Performance: {sector.title()} (ETF: {etf})"]
        lines.append(f"Period: {period_param}")
        lines.append(f"Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if not etf_hist.empty:
            etf_start = etf_hist["Close"].iloc[0]
            etf_end = etf_hist["Close"].iloc[-1]
            etf_return = (etf_end - etf_start) / etf_start * 100
            lines.append(f"{etf} (Sector ETF):")
            lines.append(f"  Start Price: {etf_start:.2f}")
            lines.append(f"  End Price: {etf_end:.2f}")
            lines.append(f"  Return: {etf_return:.2f}%")
            lines.append(f"  Highest: {etf_hist['High'].max():.2f}")
            lines.append(f"  Lowest: {etf_hist['Low'].min():.2f}")
            lines.append(f"  Average Volume: {etf_hist['Volume'].mean():,.0f}")

        if not spy_hist.empty:
            spy_start = spy_hist["Close"].iloc[0]
            spy_end = spy_hist["Close"].iloc[-1]
            spy_return = (spy_end - spy_start) / spy_start * 100
            lines.append(f"\nSPY (S&P 500 Benchmark):")
            lines.append(f"  Return: {spy_return:.2f}%")

            if not etf_hist.empty:
                alpha = etf_return - spy_return
                lines.append(f"\nRelative Performance (Alpha vs S&P 500): {alpha:.2f}%")

        holdings = yf_retry(lambda: etf_ticker.get_info().get("holdings", None))
        if not holdings:
            try:
                holdings_data = yf_retry(lambda: etf_ticker.institutional_holders)
                if holdings_data is not None and not holdings_data.empty:
                    lines.append(f"\nTop Institutional Holders in {etf}:")
                    lines.append(holdings_data.head(10).to_csv(index=False))
            except Exception:
                pass

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving sector performance for {sector}: {str(e)}"


def get_peer_comparison(
    ticker: Annotated[str, "ticker symbol of the company"],
):
    """Get peer comparison data by finding companies in the same sector."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            return f"No data found for symbol '{ticker}'"

        sector = info.get("sector")
        industry = info.get("industry")

        if not sector:
            return f"Could not determine sector for '{ticker}'"

        from yfinance import EquityQuery
        from yfinance.screener import screen

        lines = [f"# Peer Comparison for {ticker.upper()}"]
        lines.append(f"Sector: {sector}")
        lines.append(f"Industry: {industry}")
        lines.append(f"Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        company_data = {
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Price to Book": info.get("priceToBook"),
            "Profit Margin": info.get("profitMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "Return on Equity": info.get("returnOnEquity"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Dividend Yield": info.get("dividendYield"),
            "Debt to Equity": info.get("debtToEquity"),
            "Beta": info.get("beta"),
        }

        lines.append(f"\n{ticker.upper()} key metrics:")
        for metric, value in company_data.items():
            if value is not None:
                if isinstance(value, float):
                    lines.append(f"  {metric}: {value:.4f}")
                else:
                    lines.append(f"  {metric}: {value}")

        try:
            sector_valid_values = EquityQuery.__new__(EquityQuery).valid_values.get("sector", set())
            if sector in sector_valid_values:
                query = EquityQuery("eq", ["sector", sector])
                data = yf_retry(lambda: screen(query, size=100))
                quotes = data.get("quotes", []) if isinstance(data, dict) else []
            else:
                quotes = []
        except Exception:
            quotes = []

        if quotes:
            quotes = [q for q in quotes if q.get("symbol", "").upper() != ticker.upper()]
            quotes.sort(key=lambda q: q.get("marketCap") or 0, reverse=True)
            quotes = quotes[:10]

            lines.append(f"\nSector peers (top by market cap in {sector}):")

            def fmt_num(v):
                if v is None:
                    return "N/A"
                if abs(v) >= 1e12:
                    return f"${v / 1e12:.2f}T"
                if abs(v) >= 1e9:
                    return f"${v / 1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"${v / 1e6:.2f}M"
                return f"${v:,.2f}"

            for q in quotes:
                sym = q.get("symbol", "N/A")
                name = q.get("shortName", "N/A")
                mcap = q.get("marketCap")
                pe = q.get("forwardPE")
                price = q.get("regularMarketPrice")
                change_pct = q.get("fiftyTwoWeekChangePercent")

                line = f"  {sym} — {name}"
                details = []
                if mcap is not None:
                    details.append(f"MktCap: {fmt_num(mcap)}")
                if pe is not None:
                    details.append(f"FwdPE: {pe:.2f}")
                if price is not None:
                    details.append(f"Price: ${price:.2f}")
                if change_pct is not None:
                    details.append(f"52wChg: {change_pct * 100:.1f}%")
                if details:
                    line += f" ({', '.join(details)})"
                lines.append(line)
        else:
            lines.append(f"\nSector: {sector}")
            lines.append(f"Industry: {industry}")
            lines.append(
                "\nNote: Peer-level screener data is not available. "
                "Use the company metrics above combined with sector ETF performance to assess "
                "relative positioning."
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving peer comparison for {ticker}: {str(e)}"