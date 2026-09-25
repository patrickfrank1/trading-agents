from tradingpaperaccount.accounts import get_positions
from tradingpaperaccount.config import list_paper_accounts
from tradingpaperaccount.models import Position


class FakeClient:
    def __init__(self, api_key, secret_key, paper=True):
        self.api_key = api_key
        self.positions = [Position("AAPL", 3.0, 300.0, 90.0, 100.0)]

    def get_positions(self):
        return self.positions


def factory(api_key, secret_key, paper):
    return FakeClient(api_key, secret_key, paper)


ENV = {"ALPACA_PAPER_API_KEY_1": "key-abc1", "ALPACA_PAPER_SECRET_KEY_1": "secret1"}


def test_get_positions():
    positions = get_positions(1, env=ENV, client_factory=factory)
    assert [p.symbol for p in positions] == ["AAPL"]


def test_list_paper_accounts_reads_env():
    env = {
        **ENV,
        "ALPACA_PAPER_API_KEY_3": "key-abc3",
        "ALPACA_PAPER_SECRET_KEY_3": "secret3",
    }
    summaries = {s.index: s for s in list_paper_accounts(env)}
    assert set(summaries) == {1, 3}
    assert summaries[1].api_key_masked.endswith("abc1")


def test_list_paper_accounts_empty():
    assert list_paper_accounts({}) == []
