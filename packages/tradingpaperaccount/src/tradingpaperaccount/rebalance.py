"""Pure, deterministic rebalance math.

This module has no Alpaca (or any network) dependency: given a snapshot of the
account, a set of target weights and a price per symbol, it produces the exact
set of orders required to reach the target. This is what the unit tests pin
down.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from tradingpaperaccount.models import AccountState, OrderIntent, RebalancePlan

DEFAULT_CASH_BUFFER = 0.05
DEFAULT_MIN_ORDER_VALUE = 1.0

WeightMap = Mapping[str, float]
PriceMap = Mapping[str, float]


class RebalanceError(ValueError):
    """Raised when a rebalance request is internally inconsistent."""


def validate_weights(weights: WeightMap) -> dict[str, float]:
    """Validate and copy a symbol -> weight mapping.

    Weights may be negative (shorts). The gross exposure (sum of absolute
    values) must be > 0 so targets are well defined; the mapping is otherwise
    normalised at plan time.
    """
    if not weights:
        raise RebalanceError("target weights are empty")

    cleaned: dict[str, float] = {}
    for symbol, weight in weights.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise RebalanceError(f"invalid symbol: {symbol!r}")
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise RebalanceError(f"weight for {symbol!r} is not numeric: {weight!r}") from exc
        if not math.isfinite(value):
            raise RebalanceError(f"weight for {symbol!r} is not finite: {weight!r}")
        cleaned[symbol.strip().upper()] = value

    if all(v == 0 for v in cleaned.values()):
        raise RebalanceError("all target weights are zero")
    return cleaned


def compute_rebalance_plan(
    account: AccountState,
    target_weights: WeightMap,
    prices: PriceMap,
    *,
    cash_buffer: float = DEFAULT_CASH_BUFFER,
    min_order_value: float = DEFAULT_MIN_ORDER_VALUE,
) -> RebalancePlan:
    """Compute the orders that move ``account`` to ``target_weights``.

    ``target_weights`` are relative weights whose *gross* exposure is scaled to
    ``(1 - cash_buffer)`` of equity. For the common all-long case where the
    weights sum to 1, each target value is simply
    ``equity * (1 - cash_buffer) * weight``. Negative weights open short
    positions.

    Args:
        account: current account snapshot.
        target_weights: symbol -> weight (may be negative). Normalised by gross
            exposure, so only relative magnitudes matter.
        prices: latest price per symbol. Required for every tradable symbol; for
            a held position with no provided price the position's own
            ``current_price`` is used as a fallback.
        cash_buffer: fraction of equity to keep uninvested, in ``[0, 1)``.
        min_order_value: skip orders whose absolute traded value is below this.

    Returns:
        A :class:`RebalancePlan` with sells/covering trades ordered before buys.
    """
    if not 0 <= cash_buffer < 1:
        raise RebalanceError(f"cash_buffer must be in [0, 1), got {cash_buffer}")
    if min_order_value < 0:
        raise RebalanceError(f"min_order_value must be >= 0, got {min_order_value}")
    if account.equity <= 0:
        raise RebalanceError(f"account equity must be positive, got {account.equity}")

    weights = validate_weights(target_weights)
    gross = sum(abs(w) for w in weights.values())
    investable = account.equity * (1.0 - cash_buffer)
    target_values = {sym: investable * w / gross for sym, w in weights.items()}

    symbols: list[str] = list(target_values)
    for sym in account.positions:
        if sym not in symbols:
            symbols.append(sym)

    orders: list[OrderIntent] = []
    skipped: list[dict[str, object]] = []

    for symbol in symbols:
        position = account.positions.get(symbol)
        price = prices.get(symbol)
        if price is None and position is not None:
            price = position.current_price
        if price is None:
            raise RebalanceError(f"no price available for {symbol}")
        if price <= 0 or not math.isfinite(price):
            raise RebalanceError(f"invalid price for {symbol}: {price}")

        current_qty = position.qty if position is not None else 0.0
        current_value = current_qty * price
        target_value = target_values.get(symbol, 0.0)
        target_qty = target_value / price
        delta_value = target_value - current_value
        delta_qty = target_qty - current_qty

        if abs(delta_value) < min_order_value or delta_qty == 0:
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "below_min_order_value",
                    "delta_value": delta_value,
                    "current_value": current_value,
                    "target_value": target_value,
                }
            )
            continue

        is_crypto = bool(position and position.is_crypto) or "/" in symbol
        orders.append(
            OrderIntent(
                symbol=symbol,
                side="buy" if delta_qty > 0 else "sell",
                qty=abs(delta_qty),
                notional=abs(delta_value),
                price=price,
                current_qty=current_qty,
                target_qty=target_qty,
                current_value=current_value,
                target_value=target_value,
                is_crypto=is_crypto,
            )
        )

    # Sells/covering first (frees buying power), largest first within each side.
    orders.sort(key=lambda o: (o.side != "sell", -o.notional))

    return RebalancePlan(
        account=account.account_number,
        equity=account.equity,
        cash_buffer=cash_buffer,
        target_cash=account.equity * cash_buffer,
        orders=orders,
        skipped=skipped,
    )
