"""Typed data structures shared across the tradingpaperaccount package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Position:
    """A single open position in a paper account."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    asset_class: str = "us_equity"

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto" or "/" in self.symbol


@dataclass(frozen=True)
class Order:
    """A live Alpaca order, as reported by the broker."""

    order_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    status: str
    filled_avg_price: float | None = None
    submitted_at: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def remaining_qty(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AccountState:
    """Snapshot of a paper account needed to compute a rebalance."""

    account_number: str
    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(frozen=True)
class OrderIntent:
    """A desired order derived from the delta between current and target value.

    ``qty`` and ``notional`` are always positive; direction is carried by ``side``.
    """

    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    notional: float
    price: float
    current_qty: float
    target_qty: float
    current_value: float
    target_value: float
    is_crypto: bool = False

    @property
    def delta_qty(self) -> float:
        return self.target_qty - self.current_qty

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RebalancePlan:
    """A full, ordered set of orders that moves the account to target weights.

    Orders are ordered so that sells/covering trades precede buys (freeing
    buying power before opening new positions).
    """

    account: str
    equity: float
    cash_buffer: float
    target_cash: float
    orders: list[OrderIntent] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sell_orders(self) -> list[OrderIntent]:
        return [o for o in self.orders if o.side == "sell"]

    @property
    def buy_orders(self) -> list[OrderIntent]:
        return [o for o in self.orders if o.side == "buy"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "equity": self.equity,
            "cash_buffer": self.cash_buffer,
            "target_cash": self.target_cash,
            "orders": [o.to_dict() for o in self.orders],
            "skipped": self.skipped,
        }


@dataclass
class OrderResult:
    """Outcome of submitting a single :class:`OrderIntent`."""

    intent: OrderIntent
    ok: bool
    order_id: str | None = None
    status: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionReport:
    """Result of a (possibly dry-run) rebalance request."""

    plan: RebalancePlan
    dry_run: bool
    results: list[OrderResult] = field(default_factory=list)

    @property
    def submitted(self) -> list[OrderResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[OrderResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "plan": self.plan.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
