"""High-level rebalance orchestration: snapshot -> plan -> (optionally) orders."""

from __future__ import annotations

from collections.abc import Mapping

from tradingpaperaccount.client import AlpacaPaperClient
from tradingpaperaccount.models import ExecutionReport, OrderResult, RebalancePlan
from tradingpaperaccount.rebalance import (
    DEFAULT_CASH_BUFFER,
    DEFAULT_MIN_ORDER_VALUE,
    compute_rebalance_plan,
    validate_weights,
)


class RebalanceExecutor:
    """Builds and optionally executes rebalance plans for one account."""

    def __init__(self, client: AlpacaPaperClient) -> None:
        self._client = client

    def plan(
        self,
        target_weights: Mapping[str, float],
        *,
        cash_buffer: float = DEFAULT_CASH_BUFFER,
        min_order_value: float = DEFAULT_MIN_ORDER_VALUE,
    ) -> RebalancePlan:
        """Compute a plan without placing any orders."""
        weights = validate_weights(target_weights)
        account = self._client.get_account_state()

        symbols = set(weights)
        symbols.update(account.positions)
        prices = self._client.get_latest_prices(sorted(symbols))

        return compute_rebalance_plan(
            account,
            weights,
            prices,
            cash_buffer=cash_buffer,
            min_order_value=min_order_value,
        )

    def execute(
        self,
        target_weights: Mapping[str, float],
        *,
        cash_buffer: float = DEFAULT_CASH_BUFFER,
        min_order_value: float = DEFAULT_MIN_ORDER_VALUE,
        dry_run: bool = True,
    ) -> ExecutionReport:
        """Plan, then submit orders unless ``dry_run``.

        Orders are submitted in plan order (sells/covering first). A failing
        order is recorded and does not abort the remaining orders.
        """
        plan = self.plan(
            target_weights,
            cash_buffer=cash_buffer,
            min_order_value=min_order_value,
        )
        if dry_run:
            return ExecutionReport(plan=plan, dry_run=True)

        results: list[OrderResult] = []
        for intent in plan.orders:
            try:
                order = self._client.submit_order(intent)
                results.append(
                    OrderResult(
                        intent=intent,
                        ok=True,
                        order_id=str(getattr(order, "id", "")) or None,
                        status=str(getattr(getattr(order, "status", None), "value", None)),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report, keep going
                results.append(OrderResult(intent=intent, ok=False, error=str(exc)))

        return ExecutionReport(plan=plan, dry_run=False, results=results)
