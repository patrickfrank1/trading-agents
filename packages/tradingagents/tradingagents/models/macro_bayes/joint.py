"""V2/V3 — joint quarterly VARX(1) over assets and macro drivers.

Reduced-form implementation of plan milestones M5/M6:

Endogenous block (standardized internally)::

    X_t = c + A X_{t-1} + B U_t + eps_t,   eps_t ~ Student-t(nu, diag(sigma))

- Endog: gold/SPX/Treasury-bond/REIT quarterly returns, change in the 10Y
  real yield, CPI inflation, corporate-profit growth, output gap, NFCI
  change and the credit-spread proxy change (HYG/LQD ratio; FRED's BAML
  OAS series are license-limited to 3 years, plan §3.3).
- Exog: USD return, WTI return, CPI surprise, mortgage-rate change,
  fiscal impulse.
- Feedback loops are *lagged only* (within-quarter slice is a DAG by
  construction of the VARX equations, plan §2.2); stability is enforced by
  rejecting posterior draws whose transition matrix A has spectral radius
  >= 0.99 (plan §2.2, :func:`common.filter_stable_draws`).
- Sign-informed truncated priors on the structurally motivated exogenous
  loadings (§2.6, §2.5): CPI <- oil (+) / USD (-) / surprise (+),
  REITs <- mortgage rate (-), output gap <- fiscal impulse (+),
  gold <- USD (-).

Deviation from the plan's V2 row: the credit/financial-condition *latent
states* are represented by their Chicago-Fed/BofA indicators directly
(NFCI is itself an estimated latent factor); explicit latent layers are
deferred to V4 per the "keep complexity low" directive.

PyMC is imported lazily; simulation from a stored fit needs only NumPy.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import (
    Scaler,
    filter_stable_draws,
    save_model,
)

logger = logging.getLogger(__name__)

ENDOG_VARS = [
    "gold_ret",
    "sp500_ret",
    "treasury_bond_ret",
    "reits_ret",
    "real_yield_chg",
    "cpi_qoq_ann",
    "profits_growth",
    "output_gap",
    "nfci_chg",
    "credit_spread_chg",
]

EXOG_VARS = [
    "usd_ret",
    "wti_oil_ret",
    "inflation_surprise",
    "mortgage_rate_chg",
    "fiscal_impulse",
]

#: structurally motivated exogenous loadings: (endog, exog, center, sign)
#: center in standardized units; ``sign`` in {"+", "-", None}.
SIGN_LINKS = [
    ("cpi_qoq_ann", "inflation_surprise", 0.6, "+"),
    ("cpi_qoq_ann", "wti_oil_ret", 0.25, "+"),
    ("cpi_qoq_ann", "usd_ret", -0.2, "-"),
    ("gold_ret", "usd_ret", -0.4, "-"),
    ("reits_ret", "mortgage_rate_chg", -0.4, "-"),
    ("output_gap", "fiscal_impulse", 0.2, "+"),
]

LINK_PRIOR_SD = 0.5
FREE_PRIOR_SD = 0.2
OFFDIAG_PRIOR_SD = 0.15
DIAG_PRIOR_MU = 0.25
DIAG_PRIOR_SD = 0.25

ASSET_VARS = ["gold_ret", "sp500_ret", "treasury_bond_ret", "reits_ret"]


@dataclass
class JointModelFit:
    endog_vars: list
    exog_vars: list
    scaler_endog: Scaler
    scaler_exog: Scaler
    A: np.ndarray               # (n_kept_draws, K, K), stability-filtered
    B: np.ndarray               # (n_kept_draws, K, L)
    intercept: np.ndarray       # (n_kept_draws, K)
    sigma: np.ndarray           # (n_kept_draws, K)
    nu: np.ndarray              # (n_kept_draws,)
    stability_rejection_rate: float = 0.0
    n_obs: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return self.A.shape[0]

    @classmethod
    def from_payload(cls, payload: dict) -> "JointModelFit":
        from .common import Scaler

        return cls(
            endog_vars=payload["endog_vars"],
            exog_vars=payload["exog_vars"],
            scaler_endog=Scaler.from_dict(payload["scaler_endog"]),
            scaler_exog=Scaler.from_dict(payload["scaler_exog"]),
            A=payload["A"],
            B=payload["B"],
            intercept=payload["intercept"],
            sigma=payload["sigma"],
            nu=payload["nu"],
            stability_rejection_rate=payload.get("stability_rejection_rate", 0.0),
            n_obs=payload.get("n_obs", 0),
            metadata=payload.get("metadata", {}),
        )


def build_joint_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Design frame: endog + exog columns, rows with any missing dropped."""
    cols = ENDOG_VARS + EXOG_VARS
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing required columns: {missing}")
    df = panel[cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna()


def fit_joint_model(
    panel: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 42,
    min_obs: int = 70,
    stability_bound: float = 0.99,
) -> JointModelFit:
    import pymc as pm
    import pytensor.tensor as pt

    df = build_joint_frame(panel)
    if len(df) < min_obs:
        raise RuntimeError(
            f"Joint model needs >= {min_obs} aligned observations, got {len(df)}"
        )

    K, L = len(ENDOG_VARS), len(EXOG_VARS)
    X = df[ENDOG_VARS].to_numpy()
    U = df[EXOG_VARS].to_numpy()

    scaler_endog = Scaler.fit(df[ENDOG_VARS])
    scaler_exog = Scaler.fit(df[EXOG_VARS])
    Xs = scaler_endog.transform(df[ENDOG_VARS]).to_numpy()
    Us = scaler_exog.transform(df[EXOG_VARS]).to_numpy()

    X_prev, X_next = Xs[:-1], Xs[1:]
    U_t = Us[1:]
    T = X_next.shape[0]

    off_mask = 1.0 - np.eye(K)
    link_pos = {(ENDOG_VARS.index(e), EXOG_VARS.index(u)): (mu, sign) for e, u, mu, sign in SIGN_LINKS}

    with pm.Model() as model:
        a_diag = pm.Normal("a_diag", DIAG_PRIOR_MU, DIAG_PRIOR_SD, shape=K)
        a_off = pm.Normal("a_off", 0.0, OFFDIAG_PRIOR_SD, shape=(K, K))
        A = pt.diag(a_diag) + a_off * off_mask

        c = pm.Normal("c", 0.0, 0.3, shape=K)

        free_mask = np.ones((K, L))
        for (i, j) in link_pos:
            free_mask[i, j] = 0.0
        b_free = pm.Normal("b_free", 0.0, FREE_PRIOR_SD, shape=(K, L)) * free_mask
        b_link_terms = []
        for (i, j), (mu, sign) in link_pos.items():
            dist = pm.Normal.dist(mu=mu, sigma=LINK_PRIOR_SD)
            if sign == "+":
                b_link_terms.append((i, j, pm.Truncated(f"b_link_{i}_{j}", dist, lower=0.0)))
            elif sign == "-":
                b_link_terms.append((i, j, pm.Truncated(f"b_link_{i}_{j}", dist, upper=0.0)))
            else:
                b_link_terms.append((i, j, pm.Normal(f"b_link_{i}_{j}", mu, LINK_PRIOR_SD)))
        b_link = pt.zeros((K, L))
        for i, j, v in b_link_terms:
            b_link = pt.set_subtensor(b_link[i, j], v)
        B = b_free + b_link

        sigma_var = pm.HalfNormal("sigma_var", 0.8, shape=K)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)

        mu = c + pt.dot(X_prev, A.T) + pt.dot(U_t, B.T)
        pm.StudentT("x_obs", nu=nu, mu=mu, sigma=sigma_var, observed=X_next)

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
    n_total = int(np.prod(post["a_diag"].shape[:-1]))

    def flat(name):
        return post[name].values.reshape(n_total, *post[name].shape[2:])

    a_diag_d = flat("a_diag")
    a_off_d = flat("a_off") * off_mask
    A_draws = a_diag_d[:, None, :] * np.eye(K)[None] + a_off_d
    A_kept, rejection_rate = filter_stable_draws(A_draws, stability_bound)

    b_link_d = _link_matrix(flat_b=flat, link_pos=link_pos, K=K, L=L)
    B_draws = flat("b_free") + b_link_d

    def subset(arr):
        return arr[: len(A_kept)]

    if len(A_kept) < n_total:
        logger.info(
            "Stability filter rejected %.1f%% of draws (%d kept)",
            100 * rejection_rate,
            len(A_kept),
        )

    fit = JointModelFit(
        endog_vars=list(ENDOG_VARS),
        exog_vars=list(EXOG_VARS),
        scaler_endog=scaler_endog,
        scaler_exog=scaler_exog,
        A=A_kept,
        B=subset(B_draws),
        intercept=subset(flat("c")),
        sigma=subset(flat("sigma_var")),
        nu=subset(flat("nu")),
        stability_rejection_rate=rejection_rate,
        n_obs=int(T),
        metadata={
            "n_draws_total": n_total,
            "stability_bound": stability_bound,
            "sign_links": [f"{e}<-{u}({s})" for e, u, _, s in SIGN_LINKS],
        },
    )
    return fit


def _link_matrix(flat_b, link_pos, K, L):
    b_link = np.zeros((flat_b("nu").shape[0], K, L))
    for (i, j), (_mu, _sign) in link_pos.items():
        b_link[:, i, j] = flat_b(f"b_link_{i}_{j}")
    return b_link


def save_joint_model(fit: JointModelFit, path: str) -> str:
    return save_model(
        path,
        {
            "format_version": 1,
            "model": "joint_v2v3",
            "endog_vars": fit.endog_vars,
            "exog_vars": fit.exog_vars,
            "scaler_endog": fit.scaler_endog.to_dict(),
            "scaler_exog": fit.scaler_exog.to_dict(),
            "A": fit.A,
            "B": fit.B,
            "intercept": fit.intercept,
            "sigma": fit.sigma,
            "nu": fit.nu,
            "stability_rejection_rate": fit.stability_rejection_rate,
            "n_obs": fit.n_obs,
            "metadata": fit.metadata,
        },
    )
