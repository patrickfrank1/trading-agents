# Autonomous Value-Investing Allocator

An autonomous opencode agent that researches one investment idea per trading
day and, once a month, rebalances **paper accounts 2 (moderate risk)** and **3
(high risk)** toward the best available ideas. Account **1** is a read-only
baseline. The live **trading account must never be queried or traded**.

## 1. Safety model (the "never touch the trading account" rule)

Three independent layers, so a single mistake cannot reach live money:

1. **opencode permissions** (`opencode.json`) — evaluated last-match-wins:
   - `*--execute*` → `ask` (any executing command pauses for approval);
   - `*mirror*` → `deny` (the only command that resolves live credentials);
   - `*ALPACA_TRADING*` → `deny` (no command may reference the live keys).
2. **Process guard** — the daily/monthly runners launch without
   `ALPACA_TRADING_API_KEY` / `ALPACA_TRADING_SECRET_KEY` in the environment, so <!-- # allow: key -->
   `resolve_trading_account()` fails closed even if invoked.
3. **Instruction guard** — the `value-allocator` agent prompt states the
   prohibition explicitly, and `AGENTS.md` records it as a hard rule.

The allocator may read accounts 1–3 (`accounts`, `status`, `positions`,
`performance`, `fill-check`) and may plan trades on 2/3 (`rebalance`, dry-run).
It never runs `mirror`. `--execute` is performed only by the human-run approval
script (§5).

## 2. Components

| Component | Path | Role |
| --- | --- | --- |
| Allocator agent | `.opencode/agent/value-allocator.md` | Daily research + monthly proposal logic |
| Slash command | `.opencode/command/allocator.md` | Manual trigger |
| Policy | `allocator/policy.json` | Machine-readable risk rules (§6) |
| Performance API | `tradingpaperaccount performance` | Portfolio-history time series |
| Approval script | `bin/allocator_approve.sh` | Human-approved execution of a proposal |
| Notify script | `bin/allocator_notify.sh` | Alerts that a proposal awaits approval |
| Daily/monthly runners | `bin/allocator_daily.sh`, `bin/allocator_monthly.sh` | Cron entry points |
| State | `.state/` | Persistent memory bank (gitignored) |

## 3. Persistent state (`.state/`)

```
.state/
  universe.json          # candidate ideas: metrics, status, report_path, timestamps
  ideas/<TICKER>.json    # full research record (thesis, catalyst, decision)
  snapshots/{1,2,3}.jsonl  # append-only daily equity/positions from the API
  proposals/<YYYY-MM>/   # weights_2.json, weights_3.json, PROPOSAL.md, approval.json
  decisions.jsonl        # audit log: proposed / approved / executed / aborted
  logs/                  # runner logs
  run.lock               # prevents overlapping runs
```

`universe.json` entry status values:

- `watch` — screened, passed red-flag filter, not yet researched.
- `researched` — a TradingAgents report exists (with `report_path`).
- `needs_update` — report older than 90 days **or** its catalyst has passed.
  This is a **to-do the user actions out of band**; it is never a daily-research
  candidate.
- `held` — currently held in account 2 or 3.
- `rejected` — failed the screen; keep the reason so it is not re-screened for
  ~90 days.

## 4. Daily pipeline (one idea per trading day)

1. **Load state**, write `run.lock` (exit if a run is in progress).
2. **Build the candidate pool** from: user-maintained screeners, ETF top
   holdings, news, and `watch` entries. **Exclude** any ticker that is `held`,
   already `researched`, or flagged `needs_update` — daily research only chases
   genuinely new ideas. Broad universe: global equities/ADRs, REITs, ETFs
   (sector/country/commodity/bond), crypto.
3. **Cheap heuristic screen** (fast/cheap model, no deep analysis): forward P/E,
   P/B, EV/EBITDA, net-debt/EBITDA, FCF yield, ROIC, margin trend, ADV, mcap.
   Reject glaring red flags (see §6 universal filters) and record why.
4. **Pick one** candidate and run the full pipeline:
   `uv run tradingagents analyze --non-interactive -t <TICKER> -d <today> …`.
   If it fails or data is missing, fall to the next candidate (log the fallback).
5. **Persist** the report path and an `ideas/<TICKER>.json` record; update
   `universe.json`.
6. **Snapshot** accounts 1–3 (`performance`/`positions --json`) into
   `.state/snapshots/`.

No idea clears the screen → optionally log "no idea today" and exit 0. The user
triggers stale (`needs_update`) refreshes manually, out of band.

## 5. Monthly pipeline + approval

Runs on the first trading day of the month.

1. **Refresh gate**: list holdings and candidates whose reports are
   `needs_update`. These must be refreshed by the user before a valid
   comparison; if any are stale, stop and notify.
2. **Ensure reports exist** for every current holding (generate one if absent)
   so the `portfolio-comparison` agent's `5_portfolio/decision.md` contract holds.
3. **Compare** the merged candidate set: run the `portfolio-comparison` agent
   (one pass) → ranked, Kelly-sized universe.
4. **Derive two allocations** by re-applying the policy caps and Kelly fraction
   from `allocator/policy.json`: account 2 (1/4 Kelly, moderate caps), account 3
   (1/2 Kelly, high caps). Respect the rebalance band to control churn.
5. **Write** `.state/proposals/<YYYY-MM>/weights_2.json`, `weights_3.json`, and
   `PROPOSAL.md` (diffs, orders, rationale, risks).
6. **Dry-run** `rebalance -a 2|3 -w … --json` and attach the projected orders.
7. **Notify** (`bin/allocator_notify.sh`) that a proposal awaits approval and
   **stop**. No `--execute` is ever run by the agent.
8. The user reviews `PROPOSAL.md` and runs
   `bin/allocator_approve.sh <YYYY-MM>`; that script shows the plan, prompts
   `[y/N]`, appends the decision to `decisions.jsonl`, and on approval runs
   `rebalance … --execute` for accounts 2 and 3. Account 1 is never touched.

## 6. Portfolio policy

Universal: long-only; fractional shares allowed; allowed instruments are global
equities/ADRs, REITs, ETFs (sector/country/commodity/bond) and crypto.
Disallowed: options, leveraged/inverse funds, margin/leverage, OTC, individual
bonds (Alpaca cannot). Universal red-flag filters for the screen: forward P/E >
40, net-debt/EBITDA > 4, negative FCF yield, ADV < $2M, market cap < $300M.
Rebalance band: trade a name only when `|target − current| > max(1% NAV, 20% of
target)`.

| Rule | Account 2 — Moderate | Account 3 — High |
| --- | --- | --- |
| Objective | Beat SPX by ≥2%/yr | Beat SPX by ≥8%/yr |
| Target realized vol (soft / hard) | 8–14% / 16% | 18–30% / 35% |
| Max drawdown tolerance | −20% | −45% |
| Kelly fraction | 1/4 | 1/2 |
| Single-name cap | 8% NAV | 15% NAV |
| Single-ETF cap | 15% | 25% |
| Crypto (total) | ≤5% | ≤15% |
| Sector/theme cap | 30% of deployed | 45% |
| Single-country / non-USD currency | 25% / 30% | 40% / 50% |
| Position count | 8–18 | 6–14 |
| Deployed equity | 50–70% | 75–95% |
| Cash floor | 5% | 2% |
| Turnover cap (1-way/mo) | 25% NAV | 45% NAV |
| p < 0.5 or E[r] ≤ 0 | weight 0 | weight 0 |

Sizing: the portfolio-comparison agent's Kelly math
(`f* = p − (1−p)/b`), then `min(f, Kelly fraction, instrument cap, sector cap)`,
then a common scale-down so deployed equity stays in band. A portfolio-vol check
(weights × asset vol/correlation) trims the whole book if estimated vol exceeds
the hard cap.

Benchmarks: SPY total return and **account 1** (live baseline).

## 7. Scheduling (cron)

```
# daily, ~07:00 ET, weekdays
0 11 * * 1-5  cd /home/pafrank/coding/trading-agents && bin/allocator_daily.sh  >> .state/logs/daily.log 2>&1
# monthly, first trading day ~09:35 ET (approval happens out of band afterwards)
35 13 1-7 * *  cd /home/pafrank/coding/trading-agents && bin/allocator_monthly.sh >> .state/logs/monthly.log 2>&1
```

The monthly cron only produces a proposal and notifies; execution is manual via
`bin/allocator_approve.sh`.

## 8. Manual workflows

- **Stale refresh**: for each ticker the monthly gate reports as `needs_update`,
  run `uv run tradingagents analyze --non-interactive -t <TICKER> -d <today> …`
  out of band, then reply "refreshed" so the proposal run can proceed.
- **Approve a proposal**: `bin/allocator_approve.sh <YYYY-MM>`.
- **Abort a proposal**: edit `approval.json` to `"decision": "aborted"` or just
  decline the prompt; nothing trades.

## 9. Open decisions

- Confirm the notification transport wired into `bin/allocator_notify.sh`
  (default: `notify-send`, overridable via `ALLOCATOR_NOTIFY_CMD`).
- Confirm the concrete LLM provider/models for screening (cheap) vs the full
  report (deep).
