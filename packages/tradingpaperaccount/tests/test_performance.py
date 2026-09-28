import pytest

from tradingpaperaccount.client import AlpacaPaperClient


def test_to_history_from_dict():
    raw = {
        "timestamp": [1_700_000_000, 1_700_086_400],
        "equity": [10_000.0, 11_000.0],
        "profit_loss": [0.0, 1_000.0],
        "profit_loss_pct": [0.0, 0.1],
        "base_value": 10_000.0,
        "timeframe": "1D",
    }
    history = AlpacaPaperClient._to_history(raw, account="2", period="1M", timeframe=None)
    assert history.account == "2"
    assert history.timeframe == "1D"
    assert history.start_equity == pytest.approx(10_000.0)
    assert history.latest_equity == pytest.approx(11_000.0)
    assert history.total_return_pct == pytest.approx(0.1)
    assert history.points[0].date == "2023-11-14"


def test_to_history_from_object():
    class _Raw:
        timestamp = [1_700_000_000]
        equity = [10_500.0]
        profit_loss = [500.0]
        profit_loss_pct = [0.05]
        base_value = 10_000.0
        timeframe = "1D"

    history = AlpacaPaperClient._to_history(_Raw(), account="3", period="1W", timeframe="1D")
    assert history.points[0].equity == pytest.approx(10_500.0)
    assert history.total_return_pct == pytest.approx(0.05)


def test_to_history_empty():
    history = AlpacaPaperClient._to_history({}, account="1", period="1M", timeframe=None)
    assert history.points == []
    assert history.total_return_pct is None
    assert history.latest_equity is None
