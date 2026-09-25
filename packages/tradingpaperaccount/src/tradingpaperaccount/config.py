"""Account credentials and target-weight file loading.

Up to three paper accounts are supported, indexed ``1..3``. Credentials are
resolved from environment variables, with account 1 falling back to the
unnumbered names so a single-account setup keeps working:

    account 1: ALPACA_PAPER_API_KEY_1 / ALPACA_PAPER_SECRET_KEY_1
               (fallback ALPACA_API_KEY / ALPACA_SECRET_KEY)
    account 2: ALPACA_PAPER_API_KEY_2 / ALPACA_PAPER_SECRET_KEY_2
    account 3: ALPACA_PAPER_API_KEY_3 / ALPACA_PAPER_SECRET_KEY_3

``ALPACA_API_KEY_<n>`` / ``ALPACA_SECRET_KEY_<n>`` are also accepted.

Alpaca has no API to create paper accounts; create them in the dashboard and
put their keys in the environment (e.g. the repo's ``.env``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_ACCOUNT_INDEX = 1
MAX_ACCOUNT_INDEX = 3


class ConfigError(ValueError):
    """Raised when account configuration cannot be resolved."""


def mask_secret(value: str | None) -> str:
    """Mask a credential for display, keeping at most the last four chars."""
    if not value:
        return "****"
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def _env_variants(index: int, kind: str) -> list[str]:
    """Candidate env var names, in priority order, for ``kind`` in {API_KEY, SECRET_KEY}."""
    names = [
        f"ALPACA_PAPER_{kind}_{index}",
        f"ALPACA_{kind}_{index}",
    ]
    if index == 1:
        names += [f"ALPACA_{kind}"]
    return names


@dataclass(frozen=True)
class PaperAccountConfig:
    index: int
    api_key: str
    secret_key: str
    paper: bool = True

    def __repr__(self) -> str:  # never leak the secret
        return (
            f"PaperAccountConfig(index={self.index}, "
            f"api_key='***', paper={self.paper})"
        )


@dataclass(frozen=True)
class PaperAccountSummary:
    """A non-secret description of a configured paper account."""

    index: int
    api_key_masked: str
    paper: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "api_key_masked": self.api_key_masked,
            "paper": self.paper,
        }


def env_credentials(
    index: int,
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(api_key, secret_key)`` for ``index`` from the environment."""
    env = env if env is not None else os.environ

    def _lookup(kind: str) -> str | None:
        for name in _env_variants(index, kind):
            value = env.get(name)
            if value:
                return value
        return None

    return _lookup("API_KEY"), _lookup("SECRET_KEY")


def _missing_credentials_message(index: int) -> str:
    api_names = " or ".join(_env_variants(index, "API_KEY"))
    secret_names = " or ".join(_env_variants(index, "SECRET_KEY"))
    return f"no credentials for paper account {index}; set {api_names} and {secret_names}"


def _validate_index(index: int) -> None:
    if not MIN_ACCOUNT_INDEX <= index <= MAX_ACCOUNT_INDEX:
        raise ConfigError(
            f"account index must be between {MIN_ACCOUNT_INDEX} and "
            f"{MAX_ACCOUNT_INDEX}, got {index}"
        )


def resolve_account(
    index: int,
    env: Mapping[str, str] | None = None,
) -> PaperAccountConfig:
    """Resolve credentials for paper account ``index`` from the environment."""
    _validate_index(index)
    api_key, secret_key = env_credentials(index, env)
    if not api_key or not secret_key:
        raise ConfigError(_missing_credentials_message(index))
    return PaperAccountConfig(index=index, api_key=api_key, secret_key=secret_key)


def configured_accounts(env: Mapping[str, str] | None = None) -> list[int]:
    """Return the indices of accounts that have resolvable credentials."""
    env = env if env is not None else os.environ
    found = []
    for index in range(MIN_ACCOUNT_INDEX, MAX_ACCOUNT_INDEX + 1):
        try:
            resolve_account(index, env)
        except ConfigError:
            continue
        found.append(index)
    return found


def list_paper_accounts(
    env: Mapping[str, str] | None = None,
) -> list[PaperAccountSummary]:
    """Describe every configured paper account, without touching the network."""
    env = env if env is not None else os.environ
    summaries: list[PaperAccountSummary] = []
    for index in range(MIN_ACCOUNT_INDEX, MAX_ACCOUNT_INDEX + 1):
        api_key, secret_key = env_credentials(index, env)
        if api_key and secret_key:
            summaries.append(
                PaperAccountSummary(
                    index=index,
                    api_key_masked=mask_secret(api_key),
                )
            )
    return summaries


def load_weights_file(path: str | os.PathLike[str]) -> tuple[dict[str, float], float | None]:
    """Load target weights from a JSON file.

    Two shapes are accepted::

        {"AAPL": 0.4, "MSFT": 0.6}

        {"cash_buffer": 0.05, "weights": {"AAPL": 0.4, "MSFT": 0.6}}

    Returns:
        ``(weights, cash_buffer)`` where ``cash_buffer`` is ``None`` if the file
        did not specify one.
    """
    weight_path = Path(path)
    try:
        raw = weight_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read weights file {weight_path}: {exc}") from exc
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"weights file {weight_path} is not valid JSON: {exc}") from exc
    return parse_weights(data)


def parse_weights(data: Any) -> tuple[dict[str, float], float | None]:
    """Parse an already-decoded weights payload (see :func:`load_weights_file`)."""
    cash_buffer: float | None = None
    payload = data
    if isinstance(data, Mapping) and "weights" in data:
        payload = data["weights"]
        if data.get("cash_buffer") is not None:
            try:
                cash_buffer = float(data["cash_buffer"])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"cash_buffer is not numeric: {data['cash_buffer']!r}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("weights payload must be a JSON object of symbol -> weight")
    weights: dict[str, float] = {}
    for symbol, value in payload.items():
        try:
            weights[str(symbol)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"weight for {symbol!r} is not numeric: {value!r}") from exc
    if not weights:
        raise ConfigError("weights payload is empty")
    return weights, cash_buffer
