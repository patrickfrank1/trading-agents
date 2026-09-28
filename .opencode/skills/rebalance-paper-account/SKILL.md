---
name: rebalance-paper-account
description: Use when the user wants to rebalance, allocate, or apply target weights to an Alpaca paper trading account (e.g. "rebalance my paper account", "rebalance portfolio to these weights", "apply these allocations", "set target weights", "rebalance account 2"), or to mirror a paper account's weighting onto the real (live) Alpaca trading account (e.g. "mirror account 1 to live", "copy my paper account to my real account"). Drives the `tradingpaperaccount` CLI: dry-run first, then `--execute` after confirmation. Trigger keywords: rebalance, paper account, mirror, live account, Alpaca, target weights, allocation, portfolio. Use ONLY when the user supplies (or a prior report provides) symbol→weight targets AND asks to apply them to an account, or explicitly asks to mirror a paper account to the live account; do NOT use to create accounts, pick tickers, or compute weights.
---

# Rebalance an Alpaca Paper Account

Apply a set of target `symbol → weight` values to one of the user's Alpaca
paper accounts using the `tradingpaperaccount` CLI. The CLI computes the orders
needed to reach the targets (fractional market orders, a cash buffer is kept)
and skips dust. **Nothing is submitted unless you pass `--execute`.**

## Secrets policy

- **Never read, print, echo, or pass API keys/secrets** anywhere in commands or output. Do not `cat .env`, do not `source .env` manually, do not reference `ALPACA_*` values.
- The CLI loads the repo's `.env` / `.env.enterprise` automatically (existing
  environment variables win), so credentials are already available when you run
  it from the repo root. You do not need to source anything.
- If an account is not configured, stop and tell the user to add the account's
  keys to `.env` (see the package README). Do not try to handle the keys.

## Preconditions

- Run every command from the repository root (`/home/pafrank/coding/trading-agents`).
- The CLI entry point is `uv run tradingpaperaccount ...`.

## Workflow

### Step 1 — Identify the account

List configured accounts and pick the one the user asked for:

```bash
uv run tradingpaperaccount accounts --json
```

Output `accounts[].index` are the available indices (1–3). If the requested
account is missing, stop and report that its keys are not in `.env`. If the user
did not specify an account, ask which index to use.

Optionally show current state so the user can sanity-check:

```bash
uv run tradingpaperaccount positions -a <INDEX> --json
```

### Step 2 — Obtain the target weights

Get `symbol → weight` from the user, or from an existing weights file / a
generated portfolio-comparison report. **Never invent weights or pick tickers
yourself.** Weights are *relative*: the CLI normalises by gross exposure
(`sum(abs(weight))`), so `{0.5, 0.3, 0.2}`, `{5, 3, 2}`, and `{50%, 30%, 20%}`
are all equivalent. Negative weights open shorts. Confirm the intended cash
buffer with the user (default `0.05`).

### Step 3 — Write a weights file

Write the targets to a JSON file (use a `mktemp` path or `reports/weights_<date>.json`):

```json
{"AAPL": 0.4, "MSFT": 0.35, "NVDA": 0.25}
```

Or embed the cash buffer:

```json
{"cash_buffer": 0.05, "weights": {"AAPL": 0.5, "MSFT": 0.5}}
```

### Step 4 — Dry run (always)

Preview the exact orders. This submits nothing:

```bash
uv run tradingpaperaccount rebalance -a <INDEX> -w <WEIGHTS_FILE> --json
```

Present the plan to the user: equity, target cash, and each order
(`side`, `symbol`, `qty`, `notional`, `target_value`). Note that sells/covering
trades execute before buys.

### Step 5 — Execute only after explicit confirmation

Do **not** run `--execute` until the user confirms the dry-run plan. Then:

```bash
uv run tradingpaperaccount rebalance -a <INDEX> -w <WEIGHTS_FILE> --execute --json
```

Report the result: submitted vs failed orders, and any error per order. Exit
code is non-zero if any order failed. If the user changes weights, dry-run again
before executing.

### Step 6 — Verify (optional)

```bash
uv run tradingpaperaccount positions -a <INDEX> --json
```

## Mirroring a paper account to the real (live) trading account

This copies the **weighting** of a paper account onto the live Alpaca account
configured under `ALPACA_TRADING_API_KEY` / `ALPACA_TRADING_SECRET_KEY`. It is <!-- # allow: key -->
live money, so be extra careful.

1. Confirm **which paper index** is the source and that the user wants this on
   their real account.
2. Preview (submits nothing). This prints the live account's current positions
   and the positions it would hold after the mirror:

   ```bash
   uv run tradingpaperaccount mirror -a <INDEX> --json
   ```

3. Show the before/after positions and every order to the user. Do NOT pass
   `--execute` without explicit confirmation.
4. Execute. `--execute` itself prints before/after and prompts `[y/N]` on the
   terminal; only pass `--yes` when the user has already confirmed and the run
   is non-interactive:

   ```bash
   uv run tradingpaperaccount mirror -a <INDEX> --execute
   ```

The target weights are derived from the paper account (`market_value /
equity`); the cash buffer defaults to the paper account's uninvested share, so
the live book matches the paper book. Arbitrary weights are impossible here by
design — the live account can only mirror a paper account.

## Guardrails

- Default to dry-run; only `--execute` trades. Never pass `--execute` without
  the user's explicit go-ahead. This applies doubly to `mirror`, which trades
  real money.
- Never run `mirror --execute` without first showing the printed before/after
  positions and getting explicit confirmation.
- Never handle credentials — rely on the CLI's automatic `.env` loading.
- Never fabricate weights or instruments; source them from the user or a report.
- Account index must be 1–3. If unsure which account, ask.
- Floor dust with `--cash-buffer`/default `min_order_value`; do not try to force
  exact share counts — the CLI handles fractional sizing.

## CLI reference

| Command | Purpose |
| --- | --- |
| `accounts [--json]` | List configured account indices (no network). |
| `status -a N [--json]` | Balances + positions for account N. |
| `positions -a N [--json]` | Positions for account N. |
| `performance -a N [--period 1M] [--timeframe 1D] [--json]` | Equity/P&L history for account N. |
| `overlap (-a A -a B \| --weights-a F --weights-b F) [--metric M] [--max-overlap F] [--json]` | Holdings overlap between two accounts or two weights files. |
| `rebalance -a N -w FILE [--cash-buffer F] [--execute] [--json]` | Plan (dry run) or submit rebalancing orders. |
| `mirror -a N [--cash-buffer F] [--execute] [--yes] [--json]` | Mirror paper account N's weights onto the live trading account (dry run by default; `--execute` prints before/after and asks `[y/N]`). |

`--json` is preferred so you can parse and summarise reliably. Full details:
`packages/tradingpaperaccount/README.md`.
