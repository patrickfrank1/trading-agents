# TradingPaperAccount

Rebalance existing Alpaca **paper** accounts to a target portfolio composition.

Given a mapping of `ticker -> weight`, it computes the exact set of market orders
that move the account to those weights, keeping a configurable cash buffer.
Up to three paper accounts are supported and selected by index.

This is a thin, testable library plus a CLI, not an MCP server: the rebalance
math is pure Python and unit-tested, and only `client.py` imports `alpaca-py`.
An agent can drive it through the CLI.

## Install

```bash
uv sync --all-packages --all-extras
```

## Paper accounts

Alpaca has **no API to create paper accounts** — you create them (up to 3) in
the web dashboard, and each new account issues its own API key/secret. Put each
account's keys in the environment under an index (1–3); that is the single
source of truth.

```bash
uv run tradingpaperaccount accounts              # which indices are configured (no network)
uv run tradingpaperaccount accounts --json
uv run tradingpaperaccount status   --account 1  # balances + positions
uv run tradingpaperaccount positions --account 1 # positions only
uv run tradingpaperaccount positions --account 1 --json
```

Programmatically:

```python
from tradingpaperaccount import get_positions, list_paper_accounts, resolve_account

list_paper_accounts()   # non-secret summaries of configured accounts
get_positions(2)        # list[Position] for account 2
resolve_account(2)      # PaperAccountConfig (credentials) for account 2
```

## Credentials

Credentials are read from the environment. The CLI loads the repo's `.env` and
`.env.enterprise` (real exports still win), and the library accepts an explicit
mapping via the `env=` argument. See the root `.env.example`.

```
account 1: ALPACA_PAPER_API_KEY_1 / ALPACA_PAPER_SECRET_KEY_1
           (fallback: ALPACA_API_KEY / ALPACA_SECRET_KEY)
account 2: ALPACA_PAPER_API_KEY_2 / ALPACA_PAPER_SECRET_KEY_2
account 3: ALPACA_PAPER_API_KEY_3 / ALPACA_PAPER_SECRET_KEY_3
```

`ALPACA_API_KEY_<n>` / `ALPACA_SECRET_KEY_<n>` are also accepted.

## Weights file

```json
{"AAPL": 0.4, "MSFT": 0.35, "NVDA": 0.25}
```

Or with an embedded cash buffer:

```json
{"cash_buffer": 0.05, "weights": {"AAPL": 0.5, "MSFT": 0.5}}
```

Weights are **relative** and normalised by gross exposure
(`sum(abs(weight))`); that gross is then scaled to `1 - cash_buffer` of equity.
So all-long weights that sum to 1 behave exactly as you'd expect, and negative
weights open shorts. Results are rounded to whole/fractional shares using
Alpaca fractional market orders.

## CLI

```bash
# which accounts are configured
uv run tradingpaperaccount accounts --json

# inspect balances/positions
uv run tradingpaperaccount status --account 1 --json

# preview a rebalance (DRY RUN by default — submits nothing)
uv run tradingpaperaccount rebalance --account 1 --weights weights.json

# actually submit: sells/covering trades are placed before buys
uv run tradingpaperaccount rebalance --account 1 --weights weights.json --execute
```

Flags: `--cash-buffer 0.05` overrides the file/default (default `0.05`),
`--min-order-value` (library default `1.0`) skips dust, `--json` is for agents.

## Library

```python
from tradingpaperaccount import RebalanceExecutor, resolve_account
from tradingpaperaccount.client import AlpacaPaperClient

config = resolve_account(1)
client = AlpacaPaperClient(config.api_key, config.secret_key)
executor = RebalanceExecutor(client)

plan = executor.plan({"AAPL": 0.5, "MSFT": 0.5}, cash_buffer=0.05)
for order in plan.orders:
    print(order.side, order.symbol, order.qty)

report = executor.execute({"AAPL": 0.5, "MSFT": 0.5}, dry_run=False)
```

Note: `AlpacaPaperClient` lives in `tradingpaperaccount.client`; the top-level
`__init__` avoids importing the SDK so the pure core stays importable without it.

## Development

```bash
uv run pytest packages/tradingpaperaccount/tests
```
