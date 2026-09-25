import pytest

from tradingpaperaccount.executor import RebalanceExecutor
from tradingpaperaccount.models import AccountState, Position


class FakeClient:
    def __init__(self, state, prices, fail_symbols=()):
        self._state = state
        self._prices = prices
        self._fail_symbols = set(fail_symbols)
        self.submitted = []

    def get_account_state(self):
        return self._state

    def get_latest_prices(self, symbols):
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def submit_order(self, intent):
        if intent.symbol in self._fail_symbols:
            raise RuntimeError(f"rejected {intent.symbol}")
        self.submitted.append(intent)

        class _Order:
            id = f"order-{intent.symbol}"
            status = "accepted"

        return _Order()


def empty_account():
    return AccountState("A", 10_000.0, 10_000.0, 20_000.0, {})


def test_dry_run_submits_nothing():
    client = FakeClient(empty_account(), {"AAPL": 100.0, "MSFT": 200.0})
    report = RebalanceExecutor(client).execute({"AAPL": 0.5, "MSFT": 0.5}, dry_run=True)
    assert report.dry_run is True
    assert client.submitted == []
    assert len(report.plan.orders) == 2


def test_execute_submits_in_plan_order():
    positions = {"AAPL": Position("AAPL", 100.0, 10_000.0, 90.0, 100.0)}
    state = AccountState("A", 10_000.0, 0.0, 0.0, positions)
    client = FakeClient(state, {"AAPL": 100.0, "MSFT": 200.0})
    report = RebalanceExecutor(client).execute(
        {"MSFT": 1.0}, cash_buffer=0.0, dry_run=False
    )
    assert [i.side for i in client.submitted] == ["sell", "buy"]
    assert all(r.ok for r in report.results)
    assert len(report.submitted) == 2


def test_failed_order_does_not_abort_others():
    client = FakeClient(
        empty_account(), {"AAPL": 100.0, "MSFT": 200.0}, fail_symbols={"AAPL"}
    )
    report = RebalanceExecutor(client).execute({"AAPL": 0.5, "MSFT": 0.5}, dry_run=False)
    assert len(report.failed) == 1
    assert len(report.submitted) == 1
    assert report.failed[0].intent.symbol == "AAPL"
