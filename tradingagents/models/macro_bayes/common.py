"""Shared model utilities: persistence, standardization, stability checks."""

import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: posterior draws whose transition matrix violates this spectral radius
#: bound are rejected (stability constraint, plan §2.2).
STABILITY_BOUND = 0.99


class Scaler:
    """Per-column standardizer that round-trips through the model file."""

    def __init__(self, means: dict, sds: dict):
        self.means = means
        self.sds = sds

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        means = {c: float(df[c].mean()) for c in df.columns}
        sds = {c: float(df[c].std(ddof=1)) or 1.0 for c in df.columns}
        return cls(means, sds)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in df.columns:
            out[c] = (df[c] - self.means[c]) / self.sds[c]
        return out

    def inverse_series(self, values: np.ndarray, column: str) -> np.ndarray:
        return values * self.sds[column] + self.means[column]

    def to_dict(self) -> dict:
        return {"means": self.means, "sds": self.sds}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(d["means"], d["sds"])


def spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def is_stable(a: np.ndarray, bound: float = STABILITY_BOUND) -> bool:
    return spectral_radius(a) < bound


def filter_stable_draws(a_draws: np.ndarray, bound: float = STABILITY_BOUND):
    """Reject posterior draws with an unstable transition matrix.

    ``a_draws`` has shape ``(n_draws, K, K)``. Returns the filtered array
    and the rejection rate. Best-practice note: rejection keeps the
    posterior constrained to the stationary region; with shrinkage priors
    and small ``A`` the rejection rate is typically low. If it exceeds
    ~50% the priors should be re-tightened rather than the constraint
    dropped.
    """
    mask = np.array([spectral_radius(a) < bound for a in a_draws])
    kept = a_draws[mask]
    if len(kept) == 0:
        raise RuntimeError(
            "All posterior draws violate the stationarity bound; "
            "the fitted dynamics are explosive. Re-fit with tighter priors."
        )
    return kept, float(1.0 - mask.mean())


def save_model(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
