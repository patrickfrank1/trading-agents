"""Per-asset regression layer for counterfactual and sensitivity queries.

The joint VARX propagates shocks only through *lagged* terms (within-
quarter simultaneity is deliberately absent pre-V6), so a "what happens
next quarter if the Fed hikes 50bp" question cannot be answered with it.
This module provides the contemporaneous instrument: V1-style Bayesian
regressions (Student-t, sign-informed truncated priors) mapping each
asset's next-quarter return to same-quarter drivers.

Gold is handled by :mod:`.v1_gold`; this registry covers SPX, REITs and
Treasury bonds. Fitted offline by ``scripts/fit_macro_model.py`` and
consumed by the ``get_macro_causal_forecast`` (counterfactual mode) and
``get_macro_sensitivity`` analyst tools.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import Scaler, save_model

logger = logging.getLogger(__name__)


@dataclass
class RegressionSpec:
    target: str
    label: str
    price_col: str
    features: list
    centers: dict          # feature -> prior center (per 1sd, pp); None = uninformative
    prior_sds: dict = field(default_factory=dict)   # optional per-feature prior sd


REGRESSION_SPECS = {
    "gold_ret": RegressionSpec(
        target="gold_ret",
        label="Gold",
        price_col="gold",
        features=["real_yield_chg", "usd_ret", "inflation_surprise", "nfci_chg"],
        centers={"real_yield_chg": -0.6, "usd_ret": -0.4, "inflation_surprise": 0.25,
                 "nfci_chg": -0.3},
    ),
    "sp500_ret": RegressionSpec(
        target="sp500_ret",
        label="S&P 500 (SPY)",
        price_col="sp500",
        features=["real_yield_chg", "profits_growth", "nfci_chg", "inflation_surprise", "usd_ret"],
        centers={"real_yield_chg": -1.0, "profits_growth": 0.3, "nfci_chg": -0.3,
                 "inflation_surprise": -0.2, "usd_ret": -0.3},
    ),
    "reits_ret": RegressionSpec(
        target="reits_ret",
        label="REITs (VNQ)",
        price_col="reits",
        features=["real_yield_chg", "mortgage_rate_chg", "nfci_chg", "profits_growth"],
        centers={"real_yield_chg": -1.0, "mortgage_rate_chg": -0.5, "nfci_chg": -0.3,
                 "profits_growth": 0.2},
    ),
    "treasury_bond_ret": RegressionSpec(
        target="treasury_bond_ret",
        label="Treasury bonds (TLT)",
        price_col="treasury_bond",
        features=["real_yield_chg", "inflation_surprise", "nfci_chg"],
        centers={"real_yield_chg": -5.0, "inflation_surprise": -0.3},
        prior_sds={"real_yield_chg": 2.0},
    ),
}


@dataclass
class AssetRegressionFit:
    target: str
    label: str
    price_col: str
    features: list
    scaler: Scaler
    posterior: dict
    n_obs: int = 0

    @property
    def n_draws(self) -> int:
        return len(self.posterior["alpha"])

    @classmethod
    def from_payload(cls, p: dict) -> "AssetRegressionFit":
        return cls(
            target=p["target"], label=p["label"], price_col=p["price_col"],
            features=p["features"], scaler=Scaler.from_dict(p["scaler"]),
            posterior=p["posterior"], n_obs=p.get("n_obs", 0),
        )


def _design_frame(panel: pd.DataFrame, spec: RegressionSpec) -> pd.DataFrame:
    """Same-window design: drivers and the asset return over the SAME quarter.

    This is a conditional (structural) model for counterfactual and
    sensitivity queries — 'conditional on the drivers doing X during the
    quarter, the asset return is Y'. It is deliberately NOT an ex-ante
    forecasting model; the lagged V1 gold model covers that use case.
    """
    df = pd.DataFrame(index=panel.index)
    for f in spec.features:
        df[f] = panel[f]
    df["y"] = panel[spec.target]
    return df.dropna()


def fit_asset_regressions(
    panel: pd.DataFrame,
    draws: int = 1200,
    tune: int = 1200,
    chains: int = 2,
    seed: int = 42,
    min_obs: int = 60,
) -> dict:
    """Fit all registry regressions. Returns {target: payload dict}."""
    import pymc as pm

    payloads = {}
    for target, spec in REGRESSION_SPECS.items():
        df = _design_frame(panel, spec)
        if len(df) < min_obs:
            logger.warning("Asset regression %s skipped: %d aligned obs < %d",
                           target, len(df), min_obs)
            continue
        features = list(spec.features)
        X_raw = df[features]
        scaler = Scaler.fit(X_raw)
        X = scaler.transform(X_raw).to_numpy()
        y = df["y"].to_numpy()

        with pm.Model() as m:
            alpha = pm.Normal("alpha", 0.0, 0.5)
            beta = []
            for name in features:
                c = spec.centers.get(name, 0.0)
                dist = pm.Normal.dist(mu=c, sigma=spec.prior_sds.get(name, 0.8))
                if c > 0:
                    beta.append(pm.Truncated(f"b_{name}", dist, lower=0.0))
                elif c < 0:
                    beta.append(pm.Truncated(f"b_{name}", dist, upper=0.0))
                else:
                    beta.append(pm.Normal(f"b_{name}", 0.0, 0.8))
            beta = pm.math.stack(beta)
            sigma = pm.HalfNormal("sigma", 1.0)
            nu = pm.Gamma("nu", 2.0, 0.1)
            pm.StudentT("obs", nu=nu, mu=alpha + pm.math.dot(X, beta), sigma=sigma, observed=y)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 2),
                              seed=seed, progressbar=False, return_inferencedata=True)

        post = idata.posterior
        beta_arr = np.stack([post[f"b_{n}"].values.reshape(-1) for n in features], axis=1)
        payloads[target] = {
            "target": target, "label": spec.label, "price_col": spec.price_col,
            "features": features, "scaler": scaler.to_dict(),
            "posterior": {
                "alpha": post["alpha"].values.reshape(-1),
                "beta": beta_arr,
                "sigma": post["sigma"].values.reshape(-1),
                "nu": post["nu"].values.reshape(-1),
            },
            "n_obs": int(len(df)),
        }
    if not payloads:
        raise RuntimeError("No asset regression had enough aligned observations to fit")
    return payloads


def save_asset_regressions(payloads: dict, path: str) -> str:
    return save_model(path, {"format_version": 1, "model": "asset_regressions",
                             "assets": payloads})


def load_asset_regressions(path: str) -> dict:
    from .common import load_model

    payload = load_model(path)
    return {t: AssetRegressionFit.from_payload(p) for t, p in payload["assets"].items()}


# ---------------------------------------------------------------------------
# Prediction, sensitivity, counterfactuals
# ---------------------------------------------------------------------------


def predict_asset(fit: AssetRegressionFit, feats: dict) -> np.ndarray:
    """Posterior predictive samples of next-quarter return (%) given features."""
    x = np.array([(float(feats[c]) - fit.scaler.means[c]) / fit.scaler.sds[c]
                  for c in fit.features])
    p = fit.posterior
    mu = p["alpha"] + p["beta"] @ x
    rng = np.random.default_rng(0)
    return mu + p["sigma"] * rng.standard_t(p["nu"])


def parse_shocks(shocks: str, valid_names: set, sds: dict, max_sd: float = 3.0) -> dict:
    """Parse a counterfactual spec string into {var: {"value": float, "unit": str}}.

    Format: comma-separated ``name:value`` entries. ``value`` is either a
    number in the variable's native panel units (percentage points for
    rate/changes, percent for returns) or ``Nsd`` (N standard deviations,
    relative to the latest observed value). Invalid names or magnitudes
    beyond ``max_sd`` raise ``ValueError``.
    """
    out = {}
    for chunk in shocks.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Malformed shock '{chunk}' (expected name:value)")
        name, _, raw = chunk.partition(":")
        name, raw = name.strip(), raw.strip().lower()
        if name not in valid_names:
            raise ValueError(
                f"Unknown shock variable '{name}'. Valid: {sorted(valid_names)}"
            )
        if raw.endswith("sd"):
            unit, value = "sd", float(raw[:-2])
            if abs(value) > max_sd:
                raise ValueError(f"Shock {name}={value}sd exceeds the {max_sd}sd guardrail")
            out[name] = {"value": value, "unit": "sd"}
        else:
            unit, value = "native", float(raw)
            if name in sds and sds[name] and abs(value) > max_sd * sds[name]:
                raise ValueError(
                    f"Shock {name}={value} exceeds the {max_sd}sd guardrail "
                    f"(1sd = {sds[name]:.2f})"
                )
            out[name] = {"value": value, "unit": "native"}
    if not out:
        raise ValueError("Empty shock specification")
    return out


def _apply_shock(feats: dict, name: str, spec: dict, sd: float | None) -> None:
    if spec["unit"] == "native":
        feats[name] = spec["value"]
    else:
        base = float(feats.get(name, 0.0) or 0.0)
        feats[name] = base + spec["value"] * (sd if sd else 1.0)


def _latest_available(panel: pd.DataFrame, col: str) -> float:
    s = panel[col].dropna() if col in panel.columns else pd.Series(dtype=float)
    return float(s.iloc[-1]) if not s.empty else 0.0


def counterfactual_samples(
    fits: dict,
    panel: pd.DataFrame,
    shocks: dict,
    stock_beta: float = 0.0,
) -> dict:
    """Paired baseline/shock posterior samples per asset.

    Each driver starts from its latest available observed value (per-column,
    so recently-published series don't blank the row). Baseline holds every
    shocked variable at zero move; the shock scenario applies the overrides.
    Returns ``{asset_key: {"baseline": arr, "shock": arr, "label": str}}``
    where ``asset_key`` includes ``"stock_beta"`` when ``stock_beta > 0``.
    """
    results = {}
    for target, fit in fits.items():
        feats = {f: _latest_available(panel, f) for f in fit.features}
        base = dict(feats)
        shocked = dict(feats)
        for name, spec in shocks.items():
            if name not in fit.features:
                continue
            sd = fit.scaler.sds.get(name)
            base[name] = 0.0
            _apply_shock(shocked, name, spec, sd)
        results[target] = {
            "baseline": predict_asset(fit, base),
            "shock": predict_asset(fit, shocked),
            "label": fit.label,
            "price_col": fit.price_col,
            "affected": any(name in fit.features for name in shocks),
        }
    if stock_beta and stock_beta > 0 and "sp500_ret" in results:
        s = results["sp500_ret"]
        results["stock_beta"] = {
            "baseline": s["baseline"] * stock_beta,
            "shock": s["shock"] * stock_beta,
            "label": f"Stock (beta={stock_beta:g}, SPX surrogate)",
            "price_col": "sp500",
            "affected": s["affected"],
        }
    return results


def render_sensitivity_table(fits: dict, gold_fit=None) -> str:
    """Per-driver +1sd next-quarter response table from posterior betas."""
    drivers = []
    for fit in fits.values():
        for f in fit.features:
            if f != "lag1" and f not in drivers:
                drivers.append(f)
    sds = {}
    for fit in fits.values():
        for f in drivers:
            if f in fit.scaler.sds:
                sds[f] = fit.scaler.sds[f]

    lines = [
        "# Causal Bayesian Macro Model — Driver Sensitivities",
        "",
        "Next-quarter asset-return response to a +1sd driver move",
        "(posterior mean and 90% interval, percentage points):",
        "",
        "| Driver (1sd =) | " + " | ".join(fit.label for fit in fits.values()) +
        (" | Gold (V1) |" if gold_fit else " |"),
        "|---|" + "---|" * (len(fits) + (1 if gold_fit else 0)),
    ]
    for d in drivers:
        cells = []
        for fit in fits.values():
            if d in fit.features:
                b = fit.posterior["beta"][:, fit.features.index(d)]
                cells.append(f"{b.mean():+.2f} [{np.percentile(b, 5):+.2f}, {np.percentile(b, 95):+.2f}]")
            else:
                cells.append("—")
        if gold_fit and d in gold_fit.feature_names:
            b = gold_fit.posterior["beta"][:, gold_fit.feature_names.index(d)]
            cells.append(f"{b.mean():+.2f} [{np.percentile(b, 5):+.2f}, {np.percentile(b, 95):+.2f}]")
        elif gold_fit:
            cells.append("—")
        sd_txt = f"{sds.get(d, float('nan')):.2f}" if d in sds else "?"
        lines.append(f"| {d} (1sd = {sd_txt}) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Signs are economically constrained by the priors where indicated",
        "in the model docs; magnitudes are learned from quarterly data.",
        "Multiply by (shock / 1sd) for a specific move, e.g. a +50bp hike",
        "with 55% pass-through is a +0.27pp real-yield move, i.e. about",
        "0.9sd for a driver whose 1sd is 0.30pp.",
    ]
    return "\n".join(lines)


def render_counterfactual_report(
    results: dict,
    shocks: dict,
    prices: dict,
    horizon: int = 1,
    joint_block: str | None = None,
) -> str:
    shock_txt = ", ".join(
        f"{n} = {s['value']:+g}{'sd' if s['unit'] == 'sd' else ' (native units)'}"
        for n, s in shocks.items()
    )
    lines = [
        "# Causal Bayesian Macro Model — Counterfactual",
        "",
        f"*Shock: {shock_txt} | horizon: {horizon} quarter(s) | contemporaneous",
        "regression layer (sign-constrained, Student-t)*",
        "",
        "Baseline holds the shocked variables at zero move; all other",
        "drivers stay at their latest observed values.",
        "",
        "Same-window conditional model: the shock and the asset return",
        "occur over the same quarter. This is a conditional scenario,",
        "not an ex-ante forecast.",
        "",
        "| Asset | Baseline (no move) | With shock | Effect |",
        "|---|---|---|---|",
    ]
    for key, r in results.items():
        b, s = r["baseline"], r["shock"]
        spot = prices.get(r["price_col"])
        base_med = np.median(b)
        shock_med = np.median(s)
        effect = shock_med - base_med
        base_txt = f"{base_med:+.1f}% (P+ {np.mean(b > 0):.0%})"
        shock_txt_cell = f"{shock_med:+.1f}% (P+ {np.mean(s > 0):.0%})"
        price_txt = ""
        if spot:
            price_txt = f" → ${np.median(spot * (1 + s / 100)):,.0f}"
        tag = "" if r["affected"] else " (unaffected by this shock)"
        lines.append(
            f"| {r['label']}{tag} | {base_txt} | {shock_txt_cell}{price_txt} | "
            f"{effect:+.2f}pp |"
        )
    lines += [
        "",
        "For a rate shock the pass-through from the policy rate to the",
        "10y real yield is an ASSUMPTION of the query (historical average",
        "is roughly 0.5-0.6 for a 50bp hike); state it explicitly.",
        "- Beta-adjusted equity rows are a linear single-factor surrogate:",
        "  idiosyncratic risk and beta instability are ignored.",
    ]
    if joint_block:
        lines += ["", "---", "", joint_block]
    return "\n".join(lines)
