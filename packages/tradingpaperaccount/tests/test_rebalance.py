import pytest

from tradingpaperaccount.models import AccountState, Position
from tradingpaperaccount.rebalance import (
    RebalanceError,
    compute_rebalance_plan,
    implied_cash_buffer,
    mirror_weights,
    validate_weights,
)


def make_account(equity=10_000.0, positions=None) -> AccountState:
    return AccountState(
        account_number="TEST123",
        equity=equity,
        cash=equity,
        buying_power=equity * 2,
        positions=positions or {},
    )


def test_buys_from_empty_account_with_cash_buffer():
    plan = compute_rebalance_plan(
        make_account(),
        {"AAPL": 0.5, "MSFT": 0.5},
        {"AAPL": 100.0, "MSFT": 200.0},
        cash_buffer=0.05,
    )
    by_symbol = {o.symbol: o for o in plan.orders}
    assert set(by_symbol) == {"AAPL", "MSFT"}
    assert by_symbol["AAPL"].side == "buy"
    assert by_symbol["AAPL"].target_value == pytest.approx(4_750.0)
    assert by_symbol["AAPL"].qty == pytest.approx(47.5)
    assert by_symbol["MSFT"].qty == pytest.approx(23.75)
    assert plan.target_cash == pytest.approx(500.0)
    assert plan.skipped == []


def test_sells_are_ordered_before_buys():
    positions = {"AAPL": Position("AAPL", 100.0, 10_000.0, 90.0, 100.0)}
    plan = compute_rebalance_plan(
        make_account(positions=positions),
        {"MSFT": 1.0},
        {"AAPL": 100.0, "MSFT": 200.0},
        cash_buffer=0.0,
    )
    assert [o.side for o in plan.orders] == ["sell", "buy"]
    assert plan.orders[0].symbol == "AAPL"
    assert plan.orders[0].qty == pytest.approx(100.0)
    assert plan.orders[1].symbol == "MSFT"
    assert plan.orders[1].qty == pytest.approx(50.0)


def test_weights_are_normalised_by_gross_exposure():
    # 2:2 gross -> equal 50/50 split of the investable 95%.
    plan = compute_rebalance_plan(
        make_account(),
        {"AAPL": 2.0, "MSFT": 2.0},
        {"AAPL": 100.0, "MSFT": 100.0},
        cash_buffer=0.05,
    )
    values = sorted(o.target_value for o in plan.orders)
    assert values == pytest.approx([4_750.0, 4_750.0])


def test_negative_weight_opens_a_short():
    plan = compute_rebalance_plan(
        make_account(),
        {"TSLA": -1.0},
        {"TSLA": 250.0},
        cash_buffer=0.0,
    )
    assert len(plan.orders) == 1
    order = plan.orders[0]
    assert order.side == "sell"
    assert order.target_qty == pytest.approx(-40.0)
    assert order.qty == pytest.approx(40.0)


def test_crypto_detected_by_slash():
    plan = compute_rebalance_plan(
        make_account(),
        {"BTC/USD": 1.0},
        {"BTC/USD": 50_000.0},
        cash_buffer=0.0,
    )
    assert plan.orders[0].is_crypto is True


def test_small_deltas_are_skipped():
    # weights 0.5/0.5 -> AAPL target 5000 vs current 49.995*100 = 4999.50;
    # the 0.50 delta is below the 1.0 floor and is skipped, MSFT is not.
    positions = {"AAPL": Position("AAPL", 49.995, 4_999.50, 90.0, 100.0)}
    plan = compute_rebalance_plan(
        make_account(positions=positions),
        {"AAPL": 0.5, "MSFT": 0.5},
        {"AAPL": 100.0, "MSFT": 200.0},
        cash_buffer=0.0,
        min_order_value=1.0,
    )
    assert [o.symbol for o in plan.orders] == ["MSFT"]
    assert [s["symbol"] for s in plan.skipped] == ["AAPL"]
    assert plan.skipped[0]["reason"] == "below_min_order_value"


def test_position_price_used_as_fallback():
    positions = {"AAPL": Position("AAPL", 100.0, 10_000.0, 90.0, 100.0)}
    plan = compute_rebalance_plan(
        make_account(positions=positions),
        {"MSFT": 1.0},
        {"MSFT": 200.0},
        cash_buffer=0.0,
    )
    assert {o.symbol for o in plan.orders} == {"AAPL", "MSFT"}


@pytest.mark.parametrize(
    "weights",
    [{}, {"AAPL": 0.0, "MSFT": 0.0}],
)
def test_invalid_weights(weights):
    with pytest.raises(RebalanceError):
        validate_weights(weights)


def test_cash_buffer_out_of_range():
    with pytest.raises(RebalanceError):
        compute_rebalance_plan(make_account(), {"AAPL": 1.0}, {"AAPL": 100.0}, cash_buffer=1.0)


def test_missing_price_raises():
    with pytest.raises(RebalanceError):
        compute_rebalance_plan(make_account(), {"AAPL": 1.0}, {}, cash_buffer=0.0)


def test_zero_equity_raises():
    with pytest.raises(RebalanceError):
        compute_rebalance_plan(make_account(equity=0.0), {"AAPL": 1.0}, {"AAPL": 100.0})


def test_mirror_weights_scales_market_value_by_equity():
    positions = {
        "AAPL": Position("AAPL", 50.0, 5_000.0, 90.0, 100.0),
        "MSFT": Position("MSFT", 20.0, 4_000.0, 190.0, 200.0),
    }
    weights = mirror_weights(make_account(equity=10_000.0, positions=positions))
    assert weights == {"AAPL": 0.5, "MSFT": 0.4}


def test_implied_cash_buffer_is_uninvested_share():
    positions = {"AAPL": Position("AAPL", 90.0, 9_000.0, 90.0, 100.0)}
    account = make_account(equity=10_000.0, positions=positions)
    assert implied_cash_buffer(account) == pytest.approx(0.1)


def test_mirror_with_implied_buffer_reproduces_source_allocation():
    positions = {
        "AAPL": Position("AAPL", 50.0, 5_000.0, 90.0, 100.0),
        "MSFT": Position("MSFT", 20.0, 4_000.0, 190.0, 200.0),
    }
    source = make_account(equity=10_000.0, positions=positions)
    target = make_account(equity=20_000.0)
    plan = compute_rebalance_plan(
        target,
        mirror_weights(source),
        {"AAPL": 100.0, "MSFT": 200.0},
        cash_buffer=implied_cash_buffer(source),
    )
    by_symbol = {o.symbol: o for o in plan.orders}
    assert by_symbol["AAPL"].target_value == pytest.approx(10_000.0)
    assert by_symbol["MSFT"].target_value == pytest.approx(8_000.0)


def test_mirror_weights_without_positions_raises():
    with pytest.raises(RebalanceError):
        mirror_weights(make_account(positions={}))


def test_projected_positions_reflect_orders():
    positions = {"AAPL": Position("AAPL", 100.0, 10_000.0, 90.0, 100.0)}
    plan = compute_rebalance_plan(
        make_account(positions=positions),
        {"MSFT": 1.0},
        {"AAPL": 100.0, "MSFT": 200.0},
        cash_buffer=0.0,
    )
    projected = {p.symbol: p for p in plan.projected_positions}
    assert projected["AAPL"].projected_qty == pytest.approx(0.0)
    assert projected["AAPL"].current_qty == pytest.approx(100.0)
    assert projected["MSFT"].projected_qty == pytest.approx(50.0)
    assert projected["MSFT"].current_qty == pytest.approx(0.0)
