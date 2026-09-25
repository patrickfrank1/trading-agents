import json

import pytest

from tradingpaperaccount import cli
from tradingpaperaccount.models import AccountState, Position


class FakeAlpacaClient:
    positions = {"AAPL": Position("AAPL", 100.0, 10_000.0, 90.0, 100.0)}
    prices = {"AAPL": 100.0, "MSFT": 200.0}

    def __init__(self, api_key, secret_key, paper=True):
        self.paper = paper

    def get_account_state(self):
        return AccountState("TEST", 10_000.0, 0.0, 0.0, dict(self.positions))

    def get_positions(self):
        return list(self.positions.values())

    def get_latest_prices(self, symbols):
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def submit_order(self, intent):
        FakeAlpacaClient.positions.setdefault(
            intent.symbol, Position(intent.symbol, 0.0, 0.0, 0.0, intent.price)
        )

        class _Order:
            id = "o1"
            status = "accepted"

        return _Order()


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)


@pytest.fixture
def fake_alpaca(monkeypatch):
    import tradingpaperaccount.client as client_mod

    monkeypatch.setattr(client_mod, "AlpacaPaperClient", FakeAlpacaClient)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    return FakeAlpacaClient


def test_accounts_lists_configured(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY_2", "k2")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY_2", "s2")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    rc = cli.main(["accounts", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [a["index"] for a in payload["accounts"]] == [2]


def test_positions_json(fake_alpaca, capsys):
    rc = cli.main(["positions", "-a", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["account"] == 1
    assert [p["symbol"] for p in payload["positions"]] == ["AAPL"]


def test_rebalance_dry_run_default(fake_alpaca, tmp_path, capsys):
    weights = tmp_path / "w.json"
    weights.write_text(json.dumps({"MSFT": 1.0}))

    rc = cli.main(["rebalance", "--account", "1", "--weights", str(weights), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    sides = [o["side"] for o in payload["plan"]["orders"]]
    assert sides == ["sell", "buy"]


def test_rebalance_execute(fake_alpaca, tmp_path, capsys):
    weights = tmp_path / "w.json"
    weights.write_text(json.dumps({"cash_buffer": 0.0, "weights": {"MSFT": 1.0}}))

    rc = cli.main(
        ["rebalance", "--account", "1", "--weights", str(weights), "--execute", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    assert len(payload["results"]) == 2
    assert all(r["ok"] for r in payload["results"])


def test_status_json(fake_alpaca, capsys):
    rc = cli.main(["status", "--account", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["account_number"] == "TEST"
    assert "AAPL" in payload["positions"]


def test_missing_credentials_returns_error(monkeypatch):
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER_API_KEY_1",
                "ALPACA_PAPER_SECRET_KEY_1"):
        monkeypatch.delenv(key, raising=False)
    rc = cli.main(["status", "--account", "1"])
    assert rc == 1


def test_invalid_weights_file(fake_alpaca, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = cli.main(["rebalance", "--account", "1", "--weights", str(bad)])
    assert rc == 1
