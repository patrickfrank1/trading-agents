# TradingPaperAccount

Rebalance existing Alpaca **paper** accounts to a target portfolio composition.

Given a mapping of `ticker -> weight`, it computes the exact set of market orders
that move the account to those weights, keeping a configurable cash buffer.
Up to three paper accounts are supported and selected by index.

It can also **mirror** a paper account's weighting onto a real (live) Alpaca
trading account, with an explicit before/after preview and confirmation step.

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
uv run tradingpaperaccount fill-check -a 1 -a 2  # any orders still open? (exit 1 if so)
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
account 1: ALPACA_PAPER_API_KEY_1 / ALPACA_PAPER_SECRET_KEY_1  # allow: key
           (fallback: ALPACA_API_KEY / ALPACA_SECRET_KEY)
account 2: ALPACA_PAPER_API_KEY_2 / ALPACA_PAPER_SECRET_KEY_2  # allow: key
account 3: ALPACA_PAPER_API_KEY_3 / ALPACA_PAPER_SECRET_KEY_3  # allow: key
```

`ALPACA_API_KEY_<n>` / `ALPACA_SECRET_KEY_<n>` are also accepted.

The real (live) trading account for `mirror` is separate and always live:

```
trading:  ALPACA_TRADING_API_KEY / ALPACA_TRADING_SECRET_KEY  # allow: key
```

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

### Post-open fill check

After submitting a rebalance, confirm it filled once the market is open
(orders queue outside market hours). `fill-check` accepts repeatable `-a`
indices and exits non-zero while anything is still open, so it works as a cron
or alert hook:

```bash
uv run tradingpaperaccount fill-check -a 1 -a 2
uv run tradingpaperaccount fill-check -a 1 -a 2 --json
```

A repo-root wrapper defaults to accounts 1 and 2:

```bash
bin/check_fills.sh              # accounts 1 and 2
bin/check_fills.sh 1 2 3        # explicit accounts
bin/check_fills.sh --json 1 2   # machine-readable
```

To apply a new target only after the account's current orders have filled
(avoids double-submitting a symbol whose order is still working):

```bash
bin/rebalance_after_fill.sh <account> <weights.json> [timeout_minutes]
```

## Mirroring a paper account into the real trading account

`mirror` copies the **weighting** of a paper account onto the real (live)
Alpaca trading account configured under `ALPACA_TRADING_API_KEY` / <!-- # allow: key -->
`ALPACA_TRADING_SECRET_KEY`. It is the only command that touches the live <!-- # allow: key -->
account, and it cannot be given arbitrary weights: the paper account is always
the source, so the live book can only ever reflect a book you already ran on
paper.

```bash
# preview: prints the live account's current positions AND the positions it
# would hold after the mirror (submits nothing)
uv run tradingpaperaccount mirror --account 1

# same, machine-readable
uv run tradingpaperaccount mirror --account 1 --json

# submit: prints the before/after positions, then asks for confirmation
uv run tradingpaperaccount mirror --account 1 --execute

# non-interactive: skip the prompt (positions are still printed)
uv run tradingpaperaccount mirror --account 1 --execute --yes
```

How the target is derived:

- each symbol's target weight is its paper-account `market_value / equity`;
- the live account's cash buffer defaults to the paper account's uninvested
  share (`1 - gross_exposure / equity`), so the mirror is exact. Override with
  `--cash-buffer`.
- `--execute` always prints the live account's current positions and its
  projected post-rebalance positions before asking `[y/N]`. A non-`y` answer
  (or no input) aborts without submitting anything.

`mirror` is a live-money operation: the dry run is the default, and nothing is
submitted until `--execute` plus confirmation.

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
