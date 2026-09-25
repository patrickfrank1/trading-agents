import json

import pytest

from tradingpaperaccount.config import (
    ConfigError,
    configured_accounts,
    parse_weights,
    resolve_account,
)


def test_resolve_numbered_account():
    env = {"ALPACA_PAPER_API_KEY_2": "k2", "ALPACA_PAPER_SECRET_KEY_2": "s2"}
    config = resolve_account(2, env)
    assert (config.index, config.api_key, config.secret_key) == (2, "k2", "s2")


def test_account_one_falls_back_to_unnumbered():
    env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
    config = resolve_account(1, env)
    assert config.api_key == "k" and config.secret_key == "s"


def test_numbered_takes_priority_over_unnumbered():
    env = {
        "ALPACA_PAPER_API_KEY_1": "paper",
        "ALPACA_PAPER_SECRET_KEY_1": "paper-secret",
        "ALPACA_API_KEY": "generic",
        "ALPACA_SECRET_KEY": "generic-secret",
    }
    config = resolve_account(1, env)
    assert config.api_key == "paper"


def test_missing_credentials_raises():
    with pytest.raises(ConfigError):
        resolve_account(3, {})


def test_index_out_of_range_raises():
    with pytest.raises(ConfigError):
        resolve_account(4, {})


def test_configured_accounts():
    env = {
        "ALPACA_PAPER_API_KEY_1": "k1",
        "ALPACA_PAPER_SECRET_KEY_1": "s1",
        "ALPACA_PAPER_API_KEY_3": "k3",
        "ALPACA_PAPER_SECRET_KEY_3": "s3",
    }
    assert configured_accounts(env) == [1, 3]


def test_parse_weights_flat():
    weights, cash_buffer = parse_weights({"AAPL": 0.5, "MSFT": 0.5})
    assert weights == {"AAPL": 0.5, "MSFT": 0.5}
    assert cash_buffer is None


def test_parse_weights_wrapped():
    payload = {"cash_buffer": 0.1, "weights": {"AAPL": 1}}
    weights, cash_buffer = parse_weights(payload)
    assert weights == {"AAPL": 1.0}
    assert cash_buffer == pytest.approx(0.1)


def test_parse_weights_invalid():
    with pytest.raises(ConfigError):
        parse_weights([1, 2, 3])
    with pytest.raises(ConfigError):
        parse_weights({})


def test_repr_hides_secret():
    config = resolve_account(1, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "supersecret"})
    assert "supersecret" not in repr(config)
