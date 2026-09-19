"""Causal Bayesian macro models (quarterly, PyMC).

Implements the staged plan in
``assets/bayesian_causal_model_implementation_plan.md``:

- :mod:`.v1_gold` — V1 gold baseline (small Bayesian regression, Student-t).
- :mod:`.joint` — V2/V3 joint VARX(1) over the asset/macro block with a
  financial-conditions latent and stability-constrained transition matrix.
- :mod:`.simulate` — posterior-predictive forward simulation and scenarios.
- :mod:`.validate` — rolling-origin out-of-sample harness vs baselines.

PyMC is imported lazily so the rest of the package (and the test suite)
works without it installed.
"""

MODEL_FORMAT_VERSION = 1
