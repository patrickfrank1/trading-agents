"""Fit the causal Bayesian macro models and (optionally) validate them.

Usage::

    # build/refresh the quarterly panel and fit both models
    python scripts/fit_macro_model.py

    # only V1 gold, fewer draws, with the OOS validation harness
    python scripts/fit_macro_model.py --model v1 --draws 500 --validate

Requires FRED_API_KEY for the panel build and the optional ``model``
extra (pymc, arviz, pyarrow). Model files are written to
``<data_cache_dir>/macro_models`` and are picked up by the
``get_macro_causal_forecast`` analyst tool.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.macro_timeseries import build_panel, load_latest_panel, model_dir
from tradingagents.models.macro_bayes.asset_regressions import (
    fit_asset_regressions,
    save_asset_regressions,
)
from tradingagents.models.macro_bayes.joint import fit_joint_model, save_joint_model
from tradingagents.models.macro_bayes.validate import format_validation_report, rolling_oos_gold
from tradingagents.models.macro_bayes.v1_gold import fit_gold_model, save_gold_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["v1", "assets", "joint", "all"], default="all")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default=None, help="panel start date (YYYY-MM-DD)")
    parser.add_argument("--force-panel", action="store_true", help="ignore cached panel")
    parser.add_argument("--validate", action="store_true", help="run rolling-origin OOS validation for V1")
    parser.add_argument("--folds", type=int, default=8, help="OOS folds for --validate")
    args = parser.parse_args()

    if not os.environ.get("FRED_API_KEY"):
        print("WARNING: FRED_API_KEY not set — panel will only contain market data\n")

    if args.force_panel or load_latest_panel() is None:
        print("Building quarterly macro panel (full-history FRED + yfinance)...")
        bundle = build_panel(start=args.start, force_refresh=args.force_panel)
        print(f"Panel saved: {bundle.path} ({bundle.manifest['n_quarters']} quarters, "
              f"{bundle.manifest['n_columns']} columns)")
        if bundle.manifest["warnings"]:
            print(f"Missing series: {', '.join(bundle.manifest['warnings'])}")
    else:
        bundle = load_latest_panel()
        print(f"Using cached panel: {bundle.path}")

    panel = bundle.data

    if args.model in ("v1", "all"):
        print("\nFitting V1 gold model...")
        fit = fit_gold_model(panel, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)
        path = save_gold_model(fit, os.path.join(model_dir(), "v1_gold.pkl"))
        print(f"V1 saved: {path} (n_obs={fit.n_obs}, draws={fit.n_draws})")

    if args.model in ("assets", "all"):
        print("\nFitting per-asset regression layer (gold / SPX / REITs / Treasuries)...")
        payloads = fit_asset_regressions(bundle.full, draws=args.draws, tune=args.tune,
                                         chains=args.chains, seed=args.seed)
        path = save_asset_regressions(payloads, os.path.join(model_dir(), "asset_regressions.pkl"))
        fitted = ", ".join(f"{t}(n={p['n_obs']})" for t, p in payloads.items())
        print(f"Asset regressions saved: {path} ({fitted})")

    if args.model in ("joint", "all"):
        print("\nFitting joint V2/V3 VARX model...")
        jfit = fit_joint_model(panel, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed)
        path = save_joint_model(jfit, os.path.join(model_dir(), "joint_v2v3.pkl"))
        print(
            f"Joint saved: {path} (n_obs={jfit.n_obs}, draws={jfit.n_draws}, "
            f"stability rejection={jfit.stability_rejection_rate:.1%})"
        )

    if args.validate and args.model in ("v1", "all"):
        print("\nRunning rolling-origin OOS validation for V1 (this refits per fold)...")
        summary = rolling_oos_gold(panel, n_folds=args.folds, draws=min(args.draws, 400), tune=min(args.tune, 400), seed=args.seed)
        print(format_validation_report(summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
