import json

import pytest

from tradingpaperaccount import cli
from tradingpaperaccount.models import AccountState, Order, Position


class FakeAlpacaClient:
    positions = {"AAPL": Position("AAPL", 100.0, 10_000.0, 90.0, 100.0)}
    prices = {"AAPL": 100.0, "MSFT": 200.0}
    open_orders = []

    def __init__(self, api_key, secret_key, paper=True):
        self.paper = paper

    def get_account_state(self):
        return AccountState("TEST", 10_000.0, 0.0, 0.0, dict(self.positions))

    def get_positions(self):
        return list(self.positions.values())

    def get_open_orders(self):
        return list(self.open_orders)

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
    monkeypatch.setenv("ALPACA_PAPER_API_KEY_2", "k2")  # allow: key
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY_2", "s2")  # allow: key
    return FakeAlpacaClient


def test_accounts_lists_configured(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY_2", "k2")  # allow: key
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY_2", "s2")  # allow: key
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


def test_fill_check_reports_open_orders(fake_alpaca, capsys):
    fake_alpaca.open_orders = [
        Order("o1", "MSFT", "buy", 10.0, 4.0, "partially_filled", 200.0)
    ]
    rc = cli.main(["fill-check", "-a", "1", "-a", "2", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_filled"] is False
    assert [a["index"] for a in payload["accounts"]] == [1, 2]
    assert payload["accounts"][0]["open_orders"][0]["symbol"] == "MSFT"


def test_fill_check_all_filled(fake_alpaca, capsys):
    fake_alpaca.open_orders = []
    rc = cli.main(["fill-check", "-a", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_filled"] is True


def test_missing_credentials_returns_error(monkeypatch):
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER_API_KEY_1",  # allow: key
                "ALPACA_PAPER_SECRET_KEY_1"):  # allow: key
        monkeypatch.delenv(key, raising=False)
    rc = cli.main(["status", "--account", "1"])
    assert rc == 1


def test_invalid_weights_file(fake_alpaca, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = cli.main(["rebalance", "--account", "1", "--weights", str(bad)])
    assert rc == 1


class FakeMirrorClient:
    """One fake for both roles: paper when ``paper=True``, live otherwise."""

    submitted = []

    def __init__(self, api_key, secret_key, paper=True):
        self.paper = paper

    def get_account_state(self):
        if self.paper:
            positions = {
                "AAPL": Position("AAPL", 50.0, 5_000.0, 90.0, 100.0),
                "MSFT": Position("MSFT", 20.0, 4_000.0, 190.0, 200.0),
            }
            return AccountState("PAPER", 10_000.0, 1_000.0, 1_000.0, positions)
        positions = {"TSLA": Position("TSLA", 100.0, 10_000.0, 100.0, 100.0)}
        return AccountState("LIVE", 20_000.0, 10_000.0, 20_000.0, positions)

    def get_latest_prices(self, symbols):
        prices = {"AAPL": 100.0, "MSFT": 200.0, "TSLA": 100.0}
        return {s: prices[s] for s in symbols if s in prices}

    def submit_order(self, intent):
        FakeMirrorClient.submitted.append(intent)

        class _Order:
            id = "o1"
            status = "accepted"

        return _Order()


@pytest.fixture
def fake_mirror(monkeypatch):
    import tradingpaperaccount.client as client_mod

    monkeypatch.setattr(client_mod, "AlpacaPaperClient", FakeMirrorClient)
    FakeMirrorClient.submitted = []
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setenv("ALPACA_TRADING_API_KEY", "trade-key")  # allow: key
    monkeypatch.setenv("ALPACA_TRADING_SECRET_KEY", "trade-secret")  # allow: key
    return FakeMirrorClient


def test_mirror_dry_run_derives_weights_from_paper(fake_mirror, capsys):
    rc = cli.main(["mirror", "-a", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["weights"] == {"AAPL": 0.5, "MSFT": 0.4}
    assert payload["cash_buffer"] == pytest.approx(0.1)
    projected = {p["symbol"]: p for p in payload["plan"]["projected_positions"]}
    assert projected["TSLA"]["projected_qty"] == pytest.approx(0.0)
    assert projected["AAPL"]["projected_qty"] == pytest.approx(100.0)
    assert fake_mirror.submitted == []


def test_mirror_dry_run_prints_before_and_after(fake_mirror, capsys):
    rc = cli.main(["mirror", "-a", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Current positions" in out
    assert "Positions after rebalance" in out
    assert "TSLA" in out
    assert "Pass --execute" in out


def test_mirror_execute_requires_confirmation(fake_mirror, monkeypatch, capsys):
    def _eof(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    rc = cli.main(["mirror", "-a", "1", "--execute"])
    assert rc == 1
    assert fake_mirror.submitted == []
    captured = capsys.readouterr()
    assert "aborted" in captured.err
    # The before/after positions are shown before the prompt.
    assert "Positions after rebalance" in captured.out


def test_mirror_execute_with_yes_submits(fake_mirror, capsys):
    rc = cli.main(["mirror", "-a", "1", "--execute", "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    assert len(payload["results"]) == 3
    assert all(r["ok"] for r in payload["results"])
    assert [i.symbol for i in fake_mirror.submitted] == ["TSLA", "AAPL", "MSFT"]


def test_mirror_execute_reports_submitted(fake_mirror, capsys):
    rc = cli.main(["mirror", "-a", "1", "--execute", "--yes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Submitted 3/3 order(s)" in out


def test_mirror_missing_trading_credentials(fake_mirror, monkeypatch):
    monkeypatch.delenv("ALPACA_TRADING_API_KEY", raising=False)  # allow: key
    monkeypatch.delenv("ALPACA_TRADING_SECRET_KEY", raising=False)  # allow: key
    rc = cli.main(["mirror", "-a", "1"])
    assert rc == 1
