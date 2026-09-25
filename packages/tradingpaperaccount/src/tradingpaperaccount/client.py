"""Thin alpaca-py wrapper.

Only this module imports ``alpaca``; the rebalance math, config and models stay
pure and importable without the SDK. All Alpaca-specific quirks (fractional
qty, crypto time-in-force, latest-trade lookups) are contained here.
"""

from __future__ import annotations

from typing import Any

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoLatestTradeRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tradingpaperaccount.models import AccountState, OrderIntent, Position


def is_crypto_symbol(symbol: str) -> bool:
    return "/" in symbol


class AlpacaPaperClient:
    """Convenience wrapper around the Alpaca trading + market-data clients."""

    def __init__(self, api_key: str, secret_key: str, *, paper: bool = True) -> None:
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._stock_data = StockHistoricalDataClient(api_key, secret_key)
        self._crypto_data = CryptoHistoricalDataClient(api_key, secret_key)

    # -- account -----------------------------------------------------------
    @staticmethod
    def _to_position(raw: Any) -> Position:
        asset_class = getattr(raw, "asset_class", "us_equity")
        asset_class_str = getattr(asset_class, "value", asset_class) or "us_equity"
        return Position(
            symbol=raw.symbol,
            qty=float(raw.qty),
            market_value=float(raw.market_value),
            avg_entry_price=float(raw.avg_entry_price),
            current_price=float(raw.current_price),
            asset_class=str(asset_class_str),
        )

    def get_positions(self) -> list[Position]:
        """Return all open positions for this account."""
        return [self._to_position(raw) for raw in self._trading.get_all_positions()]

    def get_account_state(self) -> AccountState:
        account = self._trading.get_account()
        positions = {pos.symbol: pos for pos in self.get_positions()}
        return AccountState(
            account_number=str(account.account_number),
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            positions=positions,
        )

    # -- market data -------------------------------------------------------
    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Return the latest trade price for each symbol (stocks and crypto)."""
        symbols = list(dict.fromkeys(symbols))  # de-dupe, preserve order
        stock_symbols = [s for s in symbols if not is_crypto_symbol(s)]
        crypto_symbols = [s for s in symbols if is_crypto_symbol(s)]

        prices: dict[str, float] = {}
        if stock_symbols:
            trades = self._stock_data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=stock_symbols)
            )
            prices.update({sym: float(trade.price) for sym, trade in trades.items()})
        if crypto_symbols:
            trades = self._crypto_data.get_crypto_latest_trade(
                CryptoLatestTradeRequest(symbol_or_symbols=crypto_symbols)
            )
            prices.update({sym: float(trade.price) for sym, trade in trades.items()})
        return prices

    # -- trading -----------------------------------------------------------
    def submit_order(self, intent: OrderIntent) -> Any:
        """Submit a single market order for an :class:`OrderIntent`.

        Fractional ``qty`` is used for both equities and crypto. Crypto cannot
        use a DAY time-in-force, so it uses GTC.
        """
        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if intent.is_crypto else TimeInForce.DAY
        order_data = MarketOrderRequest(
            symbol=intent.symbol,
            qty=intent.qty,
            side=side,
            time_in_force=tif,
        )
        return self._trading.submit_order(order_data=order_data)

    def get_open_orders(self) -> list[Any]:
        return list(self._trading.get_orders())

    def cancel_all_orders(self) -> None:
        self._trading.cancel_orders()
