"""Command-line interface.

Designed to be agent-friendly: deterministic exit codes, a ``--json`` output
mode and a dry-run default so nothing trades until ``--execute`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tradingpaperaccount.config import (
    MAX_ACCOUNT_INDEX,
    MIN_ACCOUNT_INDEX,
    ConfigError,
    list_paper_accounts,
    load_weights_file,
    resolve_account,
)
from tradingpaperaccount.rebalance import DEFAULT_CASH_BUFFER, RebalanceError

EXIT_OK = 0
EXIT_ERROR = 1


def _load_dotenv() -> None:
    """Load the monorepo's .env files if python-dotenv is available.

    Existing environment variables win (``override=False``), so real exports
    always take priority over files.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    repo_root = Path(__file__).resolve().parents[4]
    for name in (".env", ".env.enterprise"):
        path = repo_root / name
        if path.exists():
            load_dotenv(path, override=False)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cmd_accounts(args: argparse.Namespace) -> int:
    summaries = list_paper_accounts()
    if args.json:
        _print_json({"accounts": [s.to_dict() for s in summaries]})
        return EXIT_OK
    if not summaries:
        print("No paper accounts configured. Set the ALPACA_* env vars (see README).")
        return EXIT_OK
    for summary in summaries:
        print(f"  [{summary.index}] key={summary.api_key_masked}")
    return EXIT_OK


def _cmd_positions(args: argparse.Namespace) -> int:
    from tradingpaperaccount.client import AlpacaPaperClient

    config = resolve_account(args.account)
    client = AlpacaPaperClient(config.api_key, config.secret_key, paper=config.paper)
    positions = client.get_positions()

    if args.json:
        _print_json(
            {
                "account": config.index,
                "positions": [asdict(position) for position in positions],
            }
        )
        return EXIT_OK

    if not positions:
        print(f"Account {config.index}: no open positions.")
        return EXIT_OK
    print(f"Account {config.index} | {len(positions)} position(s)")
    for pos in positions:
        print(
            f"  {pos.symbol:<12} qty={pos.qty:<14.6f} "
            f"value={pos.market_value:,.2f} price={pos.current_price:,.4f}"
        )
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    from tradingpaperaccount.client import AlpacaPaperClient

    config = resolve_account(args.account)
    client = AlpacaPaperClient(config.api_key, config.secret_key, paper=config.paper)
    state = client.get_account_state()

    if args.json:
        _print_json(state.to_dict())
        return EXIT_OK

    print(f"Account {config.index} ({state.account_number})")
    print(f"  Equity:       {state.equity:,.2f}")
    print(f"  Cash:         {state.cash:,.2f}")
    print(f"  Buying power: {state.buying_power:,.2f}")
    if not state.positions:
        print("  Positions:    (none)")
    else:
        print("  Positions:")
        for pos in state.positions.values():
            print(
                f"    {pos.symbol:<12} qty={pos.qty:<14.6f} "
                f"value={pos.market_value:,.2f} price={pos.current_price:,.4f}"
            )
    return EXIT_OK


def _cmd_rebalance(args: argparse.Namespace) -> int:
    from tradingpaperaccount.client import AlpacaPaperClient
    from tradingpaperaccount.executor import RebalanceExecutor

    weights, file_cash_buffer = load_weights_file(args.weights)
    if args.cash_buffer is not None:
        cash_buffer = args.cash_buffer
    elif file_cash_buffer is not None:
        cash_buffer = file_cash_buffer
    else:
        cash_buffer = DEFAULT_CASH_BUFFER

    config = resolve_account(args.account)
    client = AlpacaPaperClient(config.api_key, config.secret_key, paper=config.paper)
    executor = RebalanceExecutor(client)
    report = executor.execute(
        weights,
        cash_buffer=cash_buffer,
        dry_run=not args.execute,
    )

    if args.json:
        _print_json(report.to_dict())
        return EXIT_OK if not report.failed else EXIT_ERROR

    plan = report.plan
    mode = "DRY RUN (no orders submitted)" if report.dry_run else "EXECUTED"
    print(f"Account {config.index} | equity={plan.equity:,.2f} | mode={mode}")
    print(f"  Cash buffer: {plan.cash_buffer:.2%} (target cash {plan.target_cash:,.2f})")
    if not plan.orders:
        print("  No orders required.")
    else:
        print(f"  {len(plan.orders)} order(s):")
        for order in plan.orders:
            print(
                f"    {order.side.upper():<4} {order.symbol:<12} "
                f"qty={order.qty:<14.6f} notional={order.notional:,.2f} "
                f"(target {order.target_value:,.2f})"
            )
    if plan.skipped:
        print(f"  Skipped {len(plan.skipped)} delta(s) below min order value.")
    for result in report.failed:
        print(
            f"  ERROR {result.intent.symbol}: {result.error}",
            file=sys.stderr,
        )
    if report.dry_run and plan.orders:
        print("  Pass --execute to submit these orders.")
    return EXIT_OK if not report.failed else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradingpaperaccount",
        description="Rebalance Alpaca paper accounts to target portfolio weights.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_accounts = sub.add_parser(
        "accounts", help="list configured paper accounts (no network calls)"
    )
    p_accounts.add_argument("--json", action="store_true", help="machine-readable output")
    p_accounts.set_defaults(func=_cmd_accounts)

    p_status = sub.add_parser("status", help="show account balances and positions")
    p_status.add_argument(
        "-a",
        "--account",
        type=int,
        required=True,
        choices=range(MIN_ACCOUNT_INDEX, MAX_ACCOUNT_INDEX + 1),
        metavar=f"{{{MIN_ACCOUNT_INDEX}..{MAX_ACCOUNT_INDEX}}}",
        help="paper account index",
    )
    p_status.add_argument("--json", action="store_true", help="machine-readable output")
    p_status.set_defaults(func=_cmd_status)

    p_positions = sub.add_parser("positions", help="list positions of a paper account")
    p_positions.add_argument(
        "-a",
        "--account",
        type=int,
        required=True,
        choices=range(MIN_ACCOUNT_INDEX, MAX_ACCOUNT_INDEX + 1),
        metavar=f"{{{MIN_ACCOUNT_INDEX}..{MAX_ACCOUNT_INDEX}}}",
        help="paper account index",
    )
    p_positions.add_argument("--json", action="store_true", help="machine-readable output")
    p_positions.set_defaults(func=_cmd_positions)

    p_rebalance = sub.add_parser(
        "rebalance", help="rebalance an account to target weights (dry-run by default)"
    )
    p_rebalance.add_argument(
        "-a",
        "--account",
        type=int,
        required=True,
        choices=range(MIN_ACCOUNT_INDEX, MAX_ACCOUNT_INDEX + 1),
        metavar=f"{{{MIN_ACCOUNT_INDEX}..{MAX_ACCOUNT_INDEX}}}",
        help="paper account index",
    )
    p_rebalance.add_argument(
        "-w",
        "--weights",
        required=True,
        help="path to a JSON file mapping symbol -> weight",
    )
    p_rebalance.add_argument(
        "-c",
        "--cash-buffer",
        type=float,
        default=None,
        help=f"fraction of equity to keep in cash (default {DEFAULT_CASH_BUFFER})",
    )
    p_rebalance.add_argument(
        "--execute",
        action="store_true",
        help="actually submit orders (default is a dry run)",
    )
    p_rebalance.add_argument("--json", action="store_true", help="machine-readable output")
    p_rebalance.set_defaults(func=_cmd_rebalance)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, RebalanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
