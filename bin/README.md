# run_batch.sh — Sequential TradingAgents Batch Runner

A `screen`-based batch runner that analyses every ticker in `bin/TICKERS.yaml`
**one at a time, strictly sequentially**, inside a single detached `screen`
session — safe to close your terminal while it runs.

Reports are written to a dedicated subfolder per run:
`reports/batch_<YYYYMMDD_HHMMSS>/<TICKER>/`.

## Requirements

- `screen`
- The TradingAgents environment (`uv` / `uv run tradingagents`)
- `bin/TICKERS.yaml` (a YAML file with a `tickers:` list)

## Quick start

```bash
# Default command for every ticker
bin/run_batch.sh start

# Override provider / models / depth / analysts / date / language
bin/run_batch.sh start --provider anthropic \
    --shallow-model claude-haiku-4-5 --deep-model claude-opus-4-6 \
    --research-depth medium -a market -a news -d 2025-04-01

# Watch progress (from another terminal)
bin/run_batch.sh status

# Watch the live controller log
tail -f .runstate/batch/batch.log

# Attach to the screen session (detach with Ctrl-a d)
bin/run_batch.sh attach

# Kill the batch
bin/run_batch.sh stop
```

## The per-ticker command

Each ticker (in YAML order, one after another) is run synchronously with:

```bash
uv run tradingagents \
  --refresh-rate 0.1 \
  --non-interactive \
  --checkpoint \
  --display-report \
  --save \
  --ticker <TICKER> \
  --save-path <REPO>/reports/batch_<ts>/<TICKER> \
  --research-depth deep \
  --provider deepseek \
  --shallow-model deepseek-flash \
  --deep-model deepseek-flash
```

The next ticker starts only after the current one has exited and its reports
have been written.

## Subcommands

| Command | Description |
| --- | --- |
| `start [options]` | Start a fresh batch run in a detached screen session (`tabatch`). Wipes previous state and creates a new `reports/batch_<ts>/` directory. Refuses to start if the session already exists. |
| `status` | Show session state, report directory, and a done/failed/pending table with the currently running ticker. |
| `attach` | Attach to the screen session (Ctrl-a d to detach). Shows the live output of the current analysis. |
| `stop` | Kill the screen session, including the analysis in flight. That ticker's `--checkpoint` data survives, so it can be re-run manually to resume. |

## Options for `start`

```
-f, --file PATH          Tickers YAML file (default: bin/TICKERS.yaml)
    --provider P         LLM provider        (default: deepseek)
    --shallow-model M    Quick-thinking model (default: deepseek-flash)
    --deep-model M       Deep-thinking model (default: deepseek-flash)
    --research-depth D   shallow|medium|deep (default: deep)
-d, --date DATE          Analysis date YYYY-MM-DD (default: none)
-a, --analyst NAME       Analyst to include; repeatable (default: CLI defaults)
-l, --language LANG      Output language (default: CLI default)
```

Every `start` is a **fresh run** — there is no batch-level resume. If the batch
is interrupted, re-run `start` to analyse all tickers again. (The per-ticker
`--checkpoint` flag still lets an interrupted *single* analysis resume if you
re-run that ticker manually.)

## How state is tracked

State lives in `TA_STATE_DIR` (default `.runstate/batch/`):

```
.runstate/batch/
├── run.conf          # parameters of the current run (sourced by the controller)
├── tickers.txt       # the ticker list, in order
├── status.tsv        # TICKER \t pending|running|done|failed \t rc  (read by `status`)
├── batch.log         # controller log: start/done/failed lines with exit codes
├── last_cmd.txt      # the exact command of the most recent ticker
└── logs/<TICKER>.log # full stdout+stderr per ticker
```

A failed ticker is logged with its exit code and the batch **continues** with
the next one.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `TA_STATE_DIR` | `.runstate/batch` | Where state and logs live. |
| `TA_SESSION` | `tabatch` | Screen session name. |
| `TICKERS_FILE` | `bin/TICKERS.yaml` | Default tickers file (overridden by `-f`). |

## Notes

- Only one batch can run at a time; `start` refuses while the `tabatch`
  session exists.
- Ticker symbols like `0700.HK` or `AM.PA` are safe as report subdirectory
  names.
- API keys are not needed by the script: `tradingagents` loads `.env` itself.
