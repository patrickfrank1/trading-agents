"""V1 — gold baseline model (quarterly Bayesian regression, Student-t).

Implements the plan's V1 specification::

    gold_ret_t = alpha + beta1*real_yield_chg + beta2*usd_ret
                 + beta3*inflation_surprise + beta4*nfci_chg
                 [+ beta5*gpr] + rho*gold_ret_lag1 + eps_t,   eps ~ Student-t

Features are standardized internally; coefficients carry sign-informed
truncated-Normal priors so the economic sign is imposed by construction
(plan §4, §7 sign-consistency gate). PyMC is imported lazily.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import Scaler, save_model

logger = logging.getLogger(__name__)

V1_FEATURES = [
    "real_yield_chg",
    "usd_ret",
    "inflation_surprise",
    "nfci_chg",
    "gold_ret_lag1",
]

#: prior center per feature (in standardized units, % gold return per 1sd
#: feature move). Sign is enforced with a truncated prior; ``None`` means
#: an uninformative two-sided prior.
V1_SIGN_CENTERS = {
    "real_yield_chg": -0.6,
    "usd_ret": -0.4,
    "inflation_surprise": 0.25,
    "nfci_chg": -0.3,
    "gpr": 0.3,
    "gold_ret_lag1": 0.1,
}

V1_PRIOR_SD = 0.8

TARGET = "gold_ret"


@dataclass
class GoldModelFit:
    feature_names: list
    scaler: Scaler
    posterior: dict            # alpha, beta (draws, K), sigma, nu
    sign_centers: dict
    n_obs: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return len(self.posterior["alpha"])

    @classmethod
    def from_payload(cls, payload: dict) -> "GoldModelFit":
        from .common import Scaler

        return cls(
            feature_names=payload["feature_names"],
            scaler=Scaler.from_dict(payload["scaler"]),
            posterior=payload["posterior"],
            sign_centers=payload.get("sign_centers", {}),
            n_obs=payload.get("n_obs", 0),
            metadata=payload.get("metadata", {}),
        )


def build_v1_frame(panel: pd.DataFrame, include_gpr: bool = True) -> pd.DataFrame:
    """Assemble the V1 design frame from the (real-time aligned) panel.

    Row ``t`` contains features known by the end of quarter ``t``; the
    target is the gold return over quarter ``t+1``.
    """
    df = pd.DataFrame(index=panel.index)
    df["gold_ret"] = panel[TARGET]
    df["real_yield_chg"] = panel["real_yield_chg"]
    df["usd_ret"] = panel["usd_ret"]
    df["inflation_surprise"] = panel.get("inflation_surprise")
    df["nfci_chg"] = panel.get("nfci_chg", panel.get("nfci"))
    if include_gpr and "gpr" in panel.columns:
        df["gpr"] = panel["gpr"]
    df["gold_ret_lag1"] = panel[TARGET]
    df["target"] = panel[TARGET].shift(-1)
    return df


def fit_gold_model(
    panel: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 42,
    include_gpr: bool = True,
    min_obs: int = 60,
) -> GoldModelFit:
    import pymc as pm

    df = build_v1_frame(panel, include_gpr=include_gpr).dropna()
    if len(df) < min_obs:
        raise RuntimeError(
            f"V1 gold model needs >= {min_obs} aligned observations, got {len(df)}"
        )

    feature_names = [c for c in df.columns if c not in ("target", "gold_ret")]
    X_raw = df[feature_names]
    y = df["target"].to_numpy()

    scaler = Scaler.fit(X_raw)
    X = scaler.transform(X_raw).to_numpy()
    n_features = X.shape[1]

    coords = {"feature": feature_names}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 0.5)
        beta = []
        for k, name in enumerate(feature_names):
            center = V1_SIGN_CENTERS.get(name, 0.0)
            dist = pm.Normal.dist(mu=center, sigma=V1_PRIOR_SD)
            if center > 0:
                beta.append(pm.Truncated(f"beta_{name}", dist, lower=0.0))
            elif center < 0:
                beta.append(pm.Truncated(f"beta_{name}", dist, upper=0.0))
            else:
                beta.append(pm.Normal(f"beta_{name}", 0.0, V1_PRIOR_SD))
        beta = pm.math.stack(beta)
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)

        mu = alpha + pm.math.dot(X, beta)
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 2),
            seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )

    post = idata.posterior
    beta_arr = np.stack([post[f"beta_{name}"].values.reshape(-1) for name in feature_names], axis=1)
    posterior = {
        "alpha": post["alpha"].values.reshape(-1),
        "beta": beta_arr,
        "sigma": post["sigma"].values.reshape(-1),
        "nu": post["nu"].values.reshape(-1),
    }
    return GoldModelFit(
        feature_names=feature_names,
        scaler=scaler,
        posterior=posterior,
        sign_centers={k: v for k, v in V1_SIGN_CENTERS.items() if k in feature_names},
        n_obs=int(len(df)),
        metadata={"target": TARGET, "n_features": n_features},
    )


def build_v1_features_from_row(panel_row: pd.Series) -> dict:
    """Extract V1 features from the latest real-time panel row."""
    feats = {
        "real_yield_chg": panel_row.get("real_yield_chg"),
        "usd_ret": panel_row.get("usd_ret"),
        "inflation_surprise": panel_row.get("inflation_surprise"),
        "nfci_chg": panel_row.get("nfci_chg", panel_row.get("nfci")),
        "gold_ret_lag1": panel_row.get("gold_ret"),
    }
    if "gpr" in panel_row.index:
        feats["gpr"] = panel_row.get("gpr")
    return {k: (0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in feats.items()}


def predict_gold(fit: GoldModelFit, x_raw: pd.Series | dict) -> np.ndarray:
    """One-quarter-ahead posterior predictive samples of the gold return (%)."""
    if isinstance(x_raw, dict):
        x_raw = pd.Series(x_raw)
    missing = [c for c in fit.feature_names if c not in x_raw.index]
    if missing:
        raise KeyError(f"Missing features for prediction: {missing}")
    x = np.array(
        [(float(x_raw[c]) - fit.scaler.means[c]) / fit.scaler.sds[c] for c in fit.feature_names]
    )
    p = fit.posterior
    mu = p["alpha"] + p["beta"] @ x
    rng = np.random.default_rng(0)
    return mu + p["sigma"] * rng.standard_t(p["nu"])


def save_gold_model(fit: GoldModelFit, path: str) -> str:
    return save_model(
        path,
        {
            "format_version": 1,
            "model": "v1_gold",
            "feature_names": fit.feature_names,
            "scaler": fit.scaler.to_dict(),
            "posterior": fit.posterior,
            "sign_centers": fit.sign_centers,
            "n_obs": fit.n_obs,
            "metadata": fit.metadata,
        },
    )
