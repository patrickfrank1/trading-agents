import json

import pytest

from tradingpaperaccount import cli
from tradingpaperaccount.models import AccountState, Position
from tradingpaperaccount.overlap import metric_value, portfolio_overlap


def test_overlap_identical_books():
    result = portfolio_overlap({"A": 0.3, "B": 0.2}, {"A": 0.3, "B": 0.2})
    assert result.shared == ("A", "B")
    assert result.nav_overlap == pytest.approx(0.5)
    assert result.invested_overlap == pytest.approx(1.0)
    assert result.jaccard == pytest.approx(1.0)
    assert result.shared_fraction_smaller == pytest.approx(1.0)


def test_overlap_disjoint_books():
    result = portfolio_overlap({"A": 0.5}, {"B": 0.5})
    assert result.shared == ()
    assert result.nav_overlap == 0.0
    assert result.invested_overlap == 0.0
    assert result.jaccard == 0.0


def test_overlap_partial_and_metrics():
    result = portfolio_overlap({"A": 0.3, "B": 0.2}, {"A": 0.1, "C": 0.4})
    assert result.shared == ("A",)
    assert result.nav_overlap == pytest.approx(0.1)
    assert result.invested_overlap == pytest.approx(0.2)
    assert result.jaccard == pytest.approx(1 / 3)
    assert result.shared_fraction_smaller == pytest.approx(0.5)
    assert metric_value(result, "nav_overlap") == pytest.approx(0.1)
    assert metric_value(result, "shared_fraction_smaller") == pytest.approx(0.5)


def test_overlap_ignores_shorts_and_empty():
    result = portfolio_overlap({"A": -0.2, "B": 0.5}, {"B": 0.5})
    assert result.shared == ("B",)
    assert result.nav_overlap == pytest.approx(0.5)
    empty = portfolio_overlap({}, {})
    assert empty.union_count == 0
    assert empty.invested_overlap == 0.0


def test_metric_value_rejects_unknown():
    result = portfolio_overlap({"A": 0.5}, {"A": 0.5})
    with pytest.raises(ValueError):
        metric_value(result, "nope")


class FakeAccountClient:
    def __init__(self, api_key, secret_key, paper=True):
        self.paper = paper

    def get_account_state(self):
        if self.paper and self.api_key_source == "2":
            return AccountState(
                "A2",
                100.0,
                50.0,
                0.0,
                {
                    "AAPL": Position("AAPL", 1.0, 30.0, 0.0, 30.0),
                    "MSFT": Position("MSFT", 1.0, 20.0, 0.0, 20.0),
                },
            )
        return AccountState(
            "A3",
            100.0,
            50.0,
            0.0,
            {
                "AAPL": Position("AAPL", 1.0, 10.0, 0.0, 10.0),
                "NVDA": Position("NVDA", 1.0, 40.0, 0.0, 40.0),
            },
        )


@pytest.fixture
def fake_accounts(monkeypatch):
    import tradingpaperaccount.client as client_mod

    def _factory(api_key, secret_key, paper=True):
        client = FakeAccountClient(api_key, secret_key, paper)
        # which account depends on the resolved key
        client.api_key_source = "2" if api_key == "k2" else "3"
        return client

    monkeypatch.setattr(client_mod, "AlpacaPaperClient", _factory)
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setenv("ALPACA_PAPER_API_KEY_2", "k2")  # allow: key
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY_2", "s2")  # allow: key
    monkeypatch.setenv("ALPACA_PAPER_API_KEY_3", "k3")  # allow: key
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY_3", "s3")  # allow: key


def test_overlap_cli_from_weights_files(tmp_path, capsys):
    wa = tmp_path / "w2.json"
    wb = tmp_path / "w3.json"
    wa.write_text(json.dumps({"A": 0.3, "B": 0.2}))
    wb.write_text(json.dumps({"A": 0.1, "C": 0.4}))
    rc = cli.main(
        ["overlap", "--weights-a", str(wa), "--weights-b", str(wb), "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["value"] == pytest.approx(0.1)
    assert payload["overlap"]["shared"] == ["A"]
    assert payload["exceeded"] is False


def test_overlap_cli_max_overlap_exits_nonzero(tmp_path, capsys):
    wa = tmp_path / "w2.json"
    wb = tmp_path / "w3.json"
    wa.write_text(json.dumps({"A": 0.5}))
    wb.write_text(json.dumps({"A": 0.5}))
    rc = cli.main(
        [
            "overlap",
            "--weights-a",
            str(wa),
            "--weights-b",
            str(wb),
            "--max-overlap",
            "0.1",
            "--json",
        ]
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["exceeded"] is True


def test_overlap_cli_exact_limit_fails(tmp_path, capsys):
    wa = tmp_path / "w2.json"
    wb = tmp_path / "w3.json"
    wa.write_text(json.dumps({"A": 0.3, "B": 0.2}))
    wb.write_text(json.dumps({"A": 0.1, "C": 0.4}))
    rc = cli.main(
        ["overlap", "--weights-a", str(wa), "--weights-b", str(wb),
         "--max-overlap", "0.1", "--json"]
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["exceeded"] is True


def test_overlap_cli_from_accounts(fake_accounts, capsys):
    rc = cli.main(["overlap", "-a", "2", "-a", "3", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["value"] == pytest.approx(0.1)
    assert payload["overlap"]["shared"] == ["AAPL"]
