"""Posterior-predictive forward simulation, scenarios, and reporting.

Simulation needs only NumPy (the posterior is persisted as arrays), so the
analyst tool never imports PyMC at runtime.
"""

import numpy as np
import pandas as pd

from .common import Scaler
from .joint import ASSET_VARS, JointModelFit
from .v1_gold import GoldModelFit

ASSET_LABELS = {
    "gold_ret": "Gold",
    "sp500_ret": "S&P 500",
    "treasury_bond_ret": "Treasury bonds (TLT)",
    "reits_ret": "REITs (VNQ)",
}

#: scenario interventions in sigma units.
#: ``shock``: one-time addition to the endogenous state entering the first
#: forecast quarter; ``exog``: constant additive shift to an exogenous
#: driver over the whole horizon.
SCENARIOS = {
    "baseline": {},
    "hawkish": {"shock": {"real_yield_chg": 1.5}, "exog": {"usd_ret": 0.5}},
    "inflation_shock": {"shock": {"cpi_qoq_ann": 1.5}, "exog": {"inflation_surprise": 2.0, "wti_oil_ret": 1.0}},
    "risk_off": {"shock": {"nfci_chg": 2.0, "credit_spread_chg": 1.5, "sp500_ret": -1.0}},
    "productivity_boom": {"shock": {"output_gap": 1.0, "profits_growth": 1.0}, "exog": {"usd_ret": -0.5}},
}


def simulate_joint(
    fit: JointModelFit,
    last_state: pd.Series | dict,
    horizon: int = 8,
    scenario: str = "baseline",
    seed: int = 0,
) -> np.ndarray:
    """Forward-simulate the endogenous block from the last observed state.

    Returns an array of shape ``(n_draws, horizon, K)`` in *standardized*
    units. Each draw iterates its own transition matrix; innovations are
    Student-t with the draw's own nu and per-variable sigma.

    ``scenario`` may be a name from ``SCENARIOS`` or a custom spec dict
    with optional ``"shock"`` / ``"exog"`` entries in sigma units.
    """
    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIOS)}")
        spec = SCENARIOS[scenario]
    else:
        spec = scenario
    K, L = len(fit.endog_vars), len(fit.exog_vars)
    endog_ix = {v: i for i, v in enumerate(fit.endog_vars)}
    exog_ix = {v: i for i, v in enumerate(fit.exog_vars)}

    x0 = np.zeros(K)
    for var in fit.endog_vars:
        if var in last_state:
            mean = fit.scaler_endog.means[var]
            sd = fit.scaler_endog.sds[var]
            x0[endog_ix[var]] = (float(last_state[var]) - mean) / sd
    x0 = x0 + _sigma_vec(spec.get("shock", {}), endog_ix, K)

    u_path = np.zeros(L)
    for var in fit.exog_vars:
        if var in last_state:
            mean = fit.scaler_exog.means[var]
            sd = fit.scaler_exog.sds[var]
            u_path[exog_ix[var]] = (float(last_state[var]) - mean) / sd
    u_path = u_path + _sigma_vec(spec.get("exog", {}), exog_ix, L)

    n = fit.n_draws
    rng = np.random.default_rng(seed)
    path = np.empty((n, horizon, K))
    x = np.repeat(x0[None, :], n, axis=0)
    for h in range(horizon):
        eps = fit.sigma * rng.standard_t(fit.nu)[:, None]
        x = (
            fit.intercept
            + np.einsum("nij,nj->ni", fit.A, x)
            + np.einsum("nkl,l->nk", fit.B, u_path)
            + eps
        )
        path[:, h, :] = x
    return path


def _sigma_vec(d: dict, ix: dict, size: int) -> np.ndarray:
    v = np.zeros(size)
    for k, s in d.items():
        v[ix[k]] = float(s)
    return v


def cumulative_returns(paths: np.ndarray, fit: JointModelFit, var: str) -> np.ndarray:
    """Cumulative % return over the horizon for one endogenous variable."""
    ix = fit.endog_vars.index(var)
    sd = fit.scaler_endog.sds[var]
    mean = fit.scaler_endog.means[var]
    return paths[:, :, ix].sum(axis=1) * sd + paths.shape[1] * mean


def summarize_cumulative(cum: np.ndarray) -> dict:
    return {
        "median": float(np.median(cum)),
        "p25": float(np.percentile(cum, 25)),
        "p75": float(np.percentile(cum, 75)),
        "p10": float(np.percentile(cum, 10)),
        "p90": float(np.percentile(cum, 90)),
        "p_positive": float((cum > 0).mean()),
        "p_below_10": float((cum < -10).mean()),
    }


def render_joint_forecast(
    fit: JointModelFit,
    panel_row: pd.Series,
    horizon: int = 8,
    scenario: str = "baseline",
    seed: int = 0,
    stock_beta: float = 0.0,
) -> str:
    paths = simulate_joint(fit, panel_row, horizon=horizon, scenario=scenario, seed=seed)
    lines = [
        "# Causal Bayesian Macro Model — Forward Scenarios",
        "",
        f"*Model: joint VARX(1) quarterly | scenario: **{scenario}** | horizon: "
        f"{horizon} quarters | draws: {fit.n_draws} (post stability filter, "
        f"{fit.stability_rejection_rate:.0%} rejected) | last panel quarter: "
        f"{panel_row.name}*",
        "",
        "Cumulative return distributions by asset:",
        "",
        "| Asset | Median | 25–75% | 10–90% | P(positive) | P(< -10%) |",
        "|---|---|---|---|---|---|",
    ]
    for var in ASSET_VARS:
        s = summarize_cumulative(cumulative_returns(paths, fit, var))
        lines.append(
            f"| {ASSET_LABELS[var]} | {s['median']:+.1f}% | "
            f"[{s['p25']:+.1f}%, {s['p75']:+.1f}%] | "
            f"[{s['p10']:+.1f}%, {s['p90']:+.1f}%] | "
            f"{s['p_positive']:.0%} | {s['p_below_10']:.0%} |"
        )
        if var == "sp500_ret" and stock_beta and stock_beta > 0:
            cum_beta = cumulative_returns(paths, fit, var) * stock_beta
            s = summarize_cumulative(cum_beta)
            lines.append(
                f"| Stock (beta={stock_beta:g}, SPX surrogate) | {s['median']:+.1f}% | "
                f"[{s['p25']:+.1f}%, {s['p75']:+.1f}%] | "
                f"[{s['p10']:+.1f}%, {s['p90']:+.1f}%] | "
                f"{s['p_positive']:.0%} | {s['p_below_10']:.0%} |"
            )
    lines += [
        "",
        "Key macro paths (median cumulative move over the horizon):",
    ]
    for var in ("real_yield_chg", "cpi_qoq_ann", "output_gap", "nfci_chg"):
        if var in fit.endog_vars:
            cum = cumulative_returns(paths, fit, var)
            lines.append(f"- {var}: {np.median(cum):+.2f} (sum of quarterly moves, own units)")
    lines += [
        "",
        "Interpretation notes:",
        "- Quarterly model; returns are total returns over the full horizon.",
        "- Scenario shocks are applied in sigma units to the first forecast",
        "  quarter (state) or across the horizon (exogenous drivers).",
        "- The signal is the DIFFERENCE vs baseline, not the level: level",
        "  forecasts embed the historical drift of a mostly bull sample and",
        "  are not price targets.",
        "- Inputs are free public data (FRED, yfinance); consensus surprises",
        "  are proxied by realized-minus-trend inflation, credit spreads by",
        "  the HYG/LQD ratio, not survey or index data.",
        "- This is a reduced-form probabilistic scenario tool, not a",
        "  structural causal estimate (identified shocks are deferred to V6).",
        f"- Valid scenarios: {', '.join(sorted(SCENARIOS))}.",
    ]
    return "\n".join(lines)


def render_v1_forecast(fit: GoldModelFit, panel_row: pd.Series) -> str:
    from .v1_gold import build_v1_features_from_row

    x = build_v1_features_from_row(panel_row)
    samples = predict_gold(fit, x)
    s = summarize_cumulative(samples)
    lines = [
        "# Causal Bayesian Macro Model — Gold (V1)",
        "",
        f"*Model: V1 gold regression | horizon: 1 quarter | draws: {fit.n_draws} | "
        f"last panel quarter: {panel_row.name}*",
        "",
        f"- Next-quarter gold return: median **{s['median']:+.1f}%**, "
        f"25–75% [{s['p25']:+.1f}%, {s['p75']:+.1f}%], "
        f"P(positive) = {s['p_positive']:.0%}",
        f"- Tail: P(return < -10%) = {s['p_below_10']:.0%}",
        "",
        "Drivers used: change in 10Y real yield, USD return, inflation",
        "surprise (realized minus trailing trend), NFCI change, lagged",
        "gold return" + (" (GPR index)" if "gpr" in fit.feature_names else "") + ".",
        "For multi-asset horizons use the joint model (model='joint').",
    ]
    return "\n".join(lines)


def render_scenario_list() -> str:
    lines = ["Available scenarios:"]
    for name, spec in SCENARIOS.items():
        bits = []
        if spec.get("shock"):
            bits.append("state shock: " + ", ".join(f"{k} {v:+.1f}σ" for k, v in spec["shock"].items()))
        if spec.get("exog"):
            bits.append("exog path: " + ", ".join(f"{k} {v:+.1f}σ" for k, v in spec["exog"].items()))
        lines.append(f"- **{name}**" + (f" ({'; '.join(bits)})" if bits else " (no intervention)"))
    return "\n".join(lines)


def build_last_state_row(panel: pd.DataFrame) -> pd.Series:
    """Latest quarter's real-time row used to seed the simulation."""
    row = panel.iloc[-1]
    row = row.copy()
    row.name = panel.index[-1].date()
    return row
