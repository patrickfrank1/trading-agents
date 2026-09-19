"""Rolling-origin out-of-sample validation harness (plan M4).

For the V1 gold model, repeatedly refit on data up to quarter ``t`` and
score the one-quarter-ahead posterior predictive against the realized
return. Baselines: random walk (zero return) and expanding-window
historical mean. Metrics per plan §7: log predictive density (primary),
RMSE, directional accuracy, 50%/80% interval coverage.

The predictive density is approximated by a Student-t fit (method of
moments) to the posterior-predictive sample of each fold — documented
approximation, sufficient for ranking models.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .common import Scaler
from .v1_gold import V1_SIGN_CENTERS, build_v1_frame

logger = logging.getLogger(__name__)


def _t_params(samples: np.ndarray):
    from scipy import stats

    loc, scale = float(np.mean(samples)), float(np.std(samples, ddof=1)) or 1e-6
    try:
        df, loc_, scale_ = stats.t.fit(samples)
        if np.isfinite(df) and df > 2.1 and scale_ > 0:
            return float(df), float(loc_), float(scale_)
    except Exception:
        pass
    return 5.0, loc, scale


def _log_score(samples: np.ndarray, realized: float) -> float:
    from scipy import stats

    df, loc, scale = _t_params(samples)
    return float(stats.t.logpdf(realized, df, loc, scale))


def _fit_gold_on(df: pd.DataFrame, draws: int, tune: int, seed: int):
    """Refit the V1 model on an arbitrary (already aligned) design frame."""
    import pymc as pm

    feature_names = [c for c in df.columns if c not in ("target", "gold_ret")]
    X_raw = df[feature_names]
    scaler = Scaler.fit(X_raw)
    X = scaler.transform(X_raw).to_numpy()
    y = df["target"].to_numpy()
    n_features = X.shape[1]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0.0, 0.5)
        beta = []
        for name in feature_names:
            center = V1_SIGN_CENTERS.get(name, 0.0)
            dist = pm.Normal.dist(mu=center, sigma=0.8)
            if center > 0:
                beta.append(pm.Truncated(f"beta_{name}", dist, lower=0.0))
            elif center < 0:
                beta.append(pm.Truncated(f"beta_{name}", dist, upper=0.0))
            else:
                beta.append(pm.Normal(f"beta_{name}", 0.0, 0.8))
        beta = pm.math.stack(beta)
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        pm.StudentT("y_obs", nu=nu, mu=alpha + pm.math.dot(X, beta), sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=1, cores=1, seed=seed, progressbar=False, return_inferencedata=True)

    post = idata.posterior
    beta_arr = np.stack([post[f"beta_{n}"].values.reshape(-1) for n in feature_names], axis=1)
    return {
        "feature_names": feature_names,
        "scaler": scaler,
        "posterior": {
            "alpha": post["alpha"].values.reshape(-1),
            "beta": beta_arr,
            "sigma": post["sigma"].values.reshape(-1),
            "nu": post["nu"].values.reshape(-1),
        },
    }


def _predict_samples(params: dict, x_raw: pd.Series, n_sims: int = 50, seed: int = 1) -> np.ndarray:
    scaler: Scaler = params["scaler"]
    x = np.array(
        [(float(x_raw[c]) - scaler.means[c]) / scaler.sds[c] for c in params["feature_names"]]
    )
    p = params["posterior"]
    mu = p["alpha"] + p["beta"] @ x
    rng = np.random.default_rng(seed)
    sims = rng.standard_t(p["nu"][:, None], size=(len(mu), n_sims)) * p["sigma"][:, None] + mu[:, None]
    return sims.reshape(-1)


def rolling_oos_gold(
    panel: pd.DataFrame,
    n_folds: int = 8,
    draws: int = 300,
    tune: int = 300,
    seed: int = 42,
    min_train: int = 80,
) -> dict:
    """Rolling-origin evaluation of the V1 gold model vs baselines."""
    df = build_v1_frame(panel).dropna()
    n = len(df)
    if n < min_train + n_folds:
        raise RuntimeError(
            f"Not enough aligned observations ({n}) for {n_folds} folds with {min_train} train rows"
        )

    results = {k: {"log_score": [], "rmse": [], "direction": [], "cov50": [], "cov80": []} for k in ("model", "historical_mean", "random_walk")}

    for f in range(n_folds):
        cut = n - n_folds + f
        train, test = df.iloc[:cut], df.iloc[cut]
        x_row = test.iloc[0].drop("target")
        realized = float(test["target"].iloc[0])
        history = df["target"].iloc[:cut]

        params = _fit_gold_on(train, draws=draws, tune=tune, seed=seed + f)
        samples = _predict_samples(params, x_row, seed=seed + f)

        _score(results["model"], samples, realized)
        hm = np.random.default_rng(f).normal(
            float(history.mean()), float(history.std(ddof=1)), size=4000
        )
        _score(results["historical_mean"], hm, realized)
        rw = np.random.default_rng(f + 1000).normal(0.0, float(history.std(ddof=1)), size=4000)
        _score(results["random_walk"], rw, realized)
        logger.info("OOS fold %d/%d done", f + 1, n_folds)

    summary = {}
    for k, v in results.items():
        summary[k] = {
            "log_score": float(np.mean(v["log_score"])),
            "rmse": float(np.mean(v["rmse"])),
            "directional_accuracy": float(np.mean(v["direction"])),
            "coverage_50": float(np.mean(v["cov50"])),
            "coverage_80": float(np.mean(v["cov80"])),
            "n_folds": n_folds,
        }
    summary["beats_random_walk"] = summary["model"]["log_score"] > summary["random_walk"]["log_score"]
    summary["beats_historical_mean"] = summary["model"]["log_score"] > summary["historical_mean"]["log_score"]
    summary["generated_at"] = datetime.utcnow().isoformat() + "Z"
    return summary


def _score(bucket: dict, samples: np.ndarray, realized: float):
    bucket["log_score"].append(_log_score(samples, realized))
    bucket["rmse"].append((float(np.mean(samples)) - realized) ** 2)
    q50 = np.percentile(samples, [25, 75])
    q80 = np.percentile(samples, [10, 90])
    bucket["direction"].append(float(np.sign(np.mean(samples)) == np.sign(realized)))
    bucket["cov50"].append(float(q50[0] <= realized <= q50[1]))
    bucket["cov80"].append(float(q80[0] <= realized <= q80[1]))


def format_validation_report(summary: dict) -> str:
    lines = [
        "# V1 Gold Model — Out-of-Sample Validation",
        "",
        f"*Generated: {summary.get('generated_at', 'n/a')} | rolling origin, "
        f"{summary['model']['n_folds']} folds, one-quarter horizon*",
        "",
        "| Model | Log predictive density | RMSE | Directional acc. | 50% cov. | 80% cov. |",
        "|---|---|---|---|---|---|",
    ]
    labels = {"model": "V1 Bayesian", "historical_mean": "Historical mean", "random_walk": "Random walk"}
    for k in ("model", "historical_mean", "random_walk"):
        s = summary[k]
        lines.append(
            f"| {labels[k]} | {s['log_score']:.3f} | {s['rmse']:.2f} | "
            f"{s['directional_accuracy']:.0%} | {s['coverage_50']:.0%} | {s['coverage_80']:.0%} |"
        )
    lines += [
        "",
        f"- Beats random walk (log score): **{summary['beats_random_walk']}**",
        f"- Beats historical mean (log score): **{summary['beats_historical_mean']}**",
        "",
        "Acceptance gate (plan §7): the model must beat both baselines on log",
        "predictive density and reach roughly its nominal interval coverage.",
    ]
    return "\n".join(lines)
