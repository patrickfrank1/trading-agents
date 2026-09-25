"""Account-level read helpers.

Alpaca has no API to create paper accounts: create them in the dashboard and
put each account's API key/secret in the environment (see
:mod:`tradingpaperaccount.config`). This module wires credential resolution to
the Alpaca client for read operations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from tradingpaperaccount.config import resolve_account
from tradingpaperaccount.models import Position

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from tradingpaperaccount.client import AlpacaPaperClient

ClientFactory = Callable[[str, str, bool], "AlpacaPaperClient"]


def _default_client_factory(api_key: str, secret_key: str, paper: bool) -> Any:
    from tradingpaperaccount.client import AlpacaPaperClient

    return AlpacaPaperClient(api_key, secret_key, paper=paper)


def get_positions(
    index: int,
    *,
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
) -> list[Position]:
    """List the open positions of paper account ``index``."""
    config = resolve_account(index, env)
    factory = client_factory or _default_client_factory
    client = factory(config.api_key, config.secret_key, config.paper)
    return client.get_positions()


__all__ = ["ClientFactory", "get_positions"]
