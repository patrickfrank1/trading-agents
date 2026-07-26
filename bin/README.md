# run_batch.sh — Parallel TradingAgents Batch Runner

A `screen`-based batch runner that analyses every ticker in `bin/TICKERS.yaml`
in parallel, with concurrency control and pause / resume / crash-recovery.

A controller loop runs in a detached `screen` session (`tabatch`) and spawns up
to `-N` per-ticker `screen` windows at a time. State is persisted to disk, so
you can pause, stop, or even suffer a machine crash and pick up exactly where
you left off — combined with TradingAgents' `--checkpoint` flag, killed
analyses resume from their last completed pipeline node.

## Requirements

- `screen` (already used for parallelization)
- The TradingAgents environment (`uv` / `uv run tradingagents`)
- `bin/TICKERS.yaml` (a YAML file with a `tickers:` list)

## Quick start

```bash
# Run 8 analyses concurrently using the default command
bin/run_batch.sh start -j 8

# Watch progress
bin/run_batch.sh status

# Soft-pause (running jobs finish, no new ones launch)
bin/run_batch.sh pause
bin/run_batch.sh resume

# Full halt (kills running jobs; they resume via --checkpoint on next start)
bin/run_batch.sh stop
bin/run_batch.sh start          # no args = reuse previous config & continue
```

## The default per-ticker command

When you start without a custom command, each ticker is analysed with:

```bash
uv run tradingagents \
  --refresh-rate 0.1 \
  --non-interactive \
  --checkpoint \
  --display-report \
  --save \
  --ticker %%TICKER%% \
  --research-depth deep \
  --provider deepseek \
  --shallow-model deepseek-v4-pro \
  --deep-model deepseek-v4-pro
```

`%%TICKER%%` is substituted with the symbol from `TICKERS.yaml` (e.g. `META`).

## Subcommands

| Command | Description |
| --- | --- |
| `start [-j N] [-f FILE] [-- CMD]` | Launch the controller. `-j` sets concurrency (default 4). `-f` selects a tickers file (default `bin/TICKERS.yaml`). `--` followed by a command overrides the default per-ticker command (must contain `%%TICKER%%`). |
| `pause` | Stop launching new jobs; jobs already running finish normally. |
| `resume` | Lift a pause; new jobs launch as slots free up. |
| `stop` | Halt the controller and kill running windows. State is preserved; re-run `start` to continue. |
| `status` | Print a progress table: done / failed / running / pending counts, percentage, and live window count. |
| `reset [--failed\|--all]` | `--failed` re-queues all failed tickers to pending. `--all` wipes all state and logs (full re-run). Refuses while the controller is running. |
| `logs TICKER` | Tail the log file for a ticker (`tail -f`). |
| `attach` | Attach to the controller `screen` session (detach with `Ctrl-a d`). |

## Options for `start`

```
-j, --concurrency N   Number of analyses to run concurrently (default: 4)
-f, --file PATH        Tickers YAML file (default: bin/TICKERS.yaml)
-- CMD...              Custom per-ticker command (must contain %%TICKER%%)
```

### Custom command example

Everything after `--` replaces the default per-ticker command verbatim. The
literal `%%TICKER%%` is substituted with each ticker symbol:

```bash
bin/run_batch.sh start -j 8 -- \
    uv run tradingagents --non-interactive --checkpoint --ticker %%TICKER%% \
        --provider openai --deep-model gpt-4o
```

## Pause / resume / crash recovery

- **`pause` / `resume`** — a soft control. The controller keeps running but
  stops launching new jobs while paused. Already-running analyses complete
  normally.
- **`stop`** — kills the controller and its per-ticker windows. Because the
  default command includes `--checkpoint`, each killed analysis saves its
  progress to its SQLite checkpoint (`~/.tradingagents/cache/checkpoints/`).
- **Recovery** — after a `stop` or a crash, just run `bin/run_batch.sh start`
  again. It reuses the persisted concurrency, tickers file, and command, and
  automatically re-queues any ticker that was mid-run when killed. Pass new
  `-j` / `-f` / `--` to override the stored config.

## How state is tracked

State lives in `TA_STATE_DIR` (default `.runstate/batch/`):

```
.runstate/batch/
├── state.tsv            # TICKER \t STATUS \t START_TS \t END_TS
├── config.env           # persisted -j / -f / paths
├── cmd_template.sh      # per-ticker command template (with %%TICKER%%)
├── wrapper.sh           # per-ticker runner (cd, exec, sentinel)
├── jobs/<TICKER>.sh     # rendered command for each ticker
├── logs/<TICKER>.log    # stdout+stderr per ticker
├── done/<TICKER>        # sentinel: analysis succeeded
├── failed/<TICKER>      # sentinel: analysis failed (contains exit code)
└── controller.log       # controller debug trace
```

`STATUS` is one of `pending`, `running`, `done`, `failed`. Completion is
detected via the sentinel files written by `wrapper.sh`, not by polling `screen`
exit codes — this is what makes state survive controller restarts and machine
crashes.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `TA_STATE_DIR` | `.runstate/batch` | Where state, logs, and sentinels live. |
| `TA_SESSION` | `tabatch` | `screen` session name. |
| `TICKERS_FILE` | `bin/TICKERS.yaml` | Default tickers file (overridden by `-f`). |
| `TA_CONCURRENCY` | `4` | Default concurrency (overridden by `-j`). |

## Notes

- The per-ticker command **must** contain the literal `%%TICKER%%` placeholder;
  it is not auto-appended. A custom command without it is rejected.
- `reset --all` wipes state **and** logs. Use `reset --failed` to selectively
  re-queue only failures.
- Logs are truncated per run (not appended). Checkpoint state (for resuming
  analyses) lives separately under `~/.tradingagents/cache/checkpoints/`.
- The controller polls every 3 s, so very short jobs may all appear to finish
  between polls — this is normal and does not affect correctness.
