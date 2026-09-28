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


def test_to_history_trims_pre_inception_zero_points():
    raw = {
        "timestamp": [1_700_000_000, 1_700_086_400, 1_700_172_800],
        "equity": [0.0, 100_000.0, 101_000.0],
        "profit_loss": [0.0, 0.0, 1_000.0],
        "profit_loss_pct": [0.0, 0.0, 0.01],
        "base_value": 100_000.0,
        "timeframe": "1D",
    }
    history = AlpacaPaperClient._to_history(raw, account="2", period="1M", timeframe=None)
    assert len(history.points) == 2
    assert history.points[0].date == "2023-11-15"
    assert history.start_equity == pytest.approx(100_000.0)
    assert history.latest_equity == pytest.approx(101_000.0)
    assert history.total_return_pct == pytest.approx(0.01)


def test_to_history_all_zero_points_is_empty():
    raw = {
        "timestamp": [1_700_000_000, 1_700_086_400],
        "equity": [0.0, 0.0],
        "profit_loss": [0.0, 0.0],
        "profit_loss_pct": [0.0, 0.0],
        "base_value": 0.0,
        "timeframe": "1D",
    }
    history = AlpacaPaperClient._to_history(raw, account="3", period="1M", timeframe=None)
    assert history.points == []
    assert history.start_equity is None
    assert history.total_return_pct is None

