"""Tests for the causal Bayesian macro models (V1 gold + joint VARX)."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.models.macro_bayes.common import (
    Scaler,
    filter_stable_draws,
    is_stable,
    spectral_radius,
)
from tradingagents.models.macro_bayes.joint import (
    ENDOG_VARS,
    EXOG_VARS,
    JointModelFit,
    build_joint_frame,
)
from tradingagents.models.macro_bayes.simulate import (
    SCENARIOS,
    ASSET_VARS,
    build_last_state_row,
    cumulative_returns,
    render_joint_forecast,
    render_scenario_list,
    simulate_joint,
    summarize_cumulative,
)
from tradingagents.models.macro_bayes.validate import _log_score, _score
from tradingagents.models.macro_bayes.v1_gold import (
    V1_FEATURES,
    GoldModelFit,
    build_v1_features_from_row,
    build_v1_frame,
    predict_gold,
    save_gold_model,
)
from tradingagents.models.macro_bayes.common import load_model


# ---------------------------------------------------------------------------
# helpers: synthetic panels
# ---------------------------------------------------------------------------


def _quarterly_index(periods):
    return pd.period_range("2000-01-01", periods=periods, freq="Q").to_timestamp(how="end").normalize()


def _synthetic_panel(periods=140, seed=11):
    rng = np.random.default_rng(seed)
    idx = _quarterly_index(periods)
    df = pd.DataFrame(index=idx)
    df.index.name = "quarter"

    # stationary-ish macro drivers
    df["real_yield_chg"] = rng.normal(0, 0.35, periods)
    df["usd_ret"] = rng.normal(0.2, 2.5, periods)
    df["wti_oil_ret"] = rng.normal(0.5, 8, periods)
    df["inflation_surprise"] = rng.normal(0, 0.8, periods)
    df["mortgage_rate_chg"] = rng.normal(0, 0.3, periods)
    df["fiscal_impulse"] = rng.normal(0, 1.0, periods)
    df["nfci_chg"] = rng.normal(0, 0.3, periods)
    df["credit_spread_chg"] = rng.normal(0, 3, periods)
    df["cpi_qoq_ann"] = 2.5 + rng.normal(0, 1.2, periods)
    df["profits_growth"] = 6 + rng.normal(0, 8, periods)
    df["output_gap"] = rng.normal(0, 1.5, periods)

    # assets driven by the drivers (so the model has signal to find)
    load = pd.DataFrame({
        "gold_ret": -0.8 * df["real_yield_chg"] - 0.2 * df["usd_ret"] + 0.1 * df["inflation_surprise"],
        "sp500_ret": -1.5 * df["real_yield_chg"] + 0.3 * df["profits_growth"],
        "treasury_bond_ret": -4.0 * df["real_yield_chg"],
        "reits_ret": -1.2 * df["mortgage_rate_chg"] * 10 - 1.0 * df["real_yield_chg"],
    })
    noise = rng.normal(0, 2.5, (periods, 4))
    for i, c in enumerate(load.columns):
        df[c] = load[c].to_numpy() + noise[:, i]
    return df


def _fake_joint_fit(panel, seed=3, n_draws=30):
    rng = np.random.default_rng(seed)
    K, L = len(ENDOG_VARS), len(EXOG_VARS)
    # strongly stable A
    a = rng.normal(0, 0.05, (n_draws, K, K))
    for d in range(n_draws):
        np.fill_diagonal(a[d], rng.uniform(0.1, 0.4))
    scaler_endog = Scaler.fit(panel[ENDOG_VARS])
    scaler_exog = Scaler.fit(panel[EXOG_VARS])
    return JointModelFit(
        endog_vars=list(ENDOG_VARS),
        exog_vars=list(EXOG_VARS),
        scaler_endog=scaler_endog,
        scaler_exog=scaler_exog,
        A=a,
        B=rng.normal(0, 0.05, (n_draws, K, L)),
        intercept=np.zeros((n_draws, K)),
        sigma=np.abs(rng.normal(1.0, 0.1, (n_draws, K))),
        nu=rng.uniform(4, 10, n_draws),
    )


# ---------------------------------------------------------------------------
# common utils
# ---------------------------------------------------------------------------


def test_spectral_radius_and_filter():
    stable = np.diag([0.2, 0.3])
    unstable = np.diag([1.5, 0.2])
    assert spectral_radius(stable) < 0.99
    assert not is_stable(unstable)
    draws = np.stack([stable, unstable, np.diag([0.98, 0.1])])
    kept, rej = filter_stable_draws(draws)
    assert len(kept) == 2
    assert rej == pytest.approx(1 / 3)
    with pytest.raises(RuntimeError):
        filter_stable_draws(np.stack([unstable]))


def test_scaler_roundtrip():
    df = pd.DataFrame({"a": np.arange(50, dtype=float), "b": rng_vals()})
    s = Scaler.fit(df)
    t = s.transform(df)
    assert t["a"].mean() == pytest.approx(0, abs=1e-9)
    back = pd.DataFrame({"a": s.inverse_series(t["a"].to_numpy(), "a")})
    np.testing.assert_allclose(back["a"], df["a"])


def rng_vals():
    return np.random.default_rng(0).normal(10, 3, 50)


# ---------------------------------------------------------------------------
# joint VARX simulation & reporting
# ---------------------------------------------------------------------------


def test_joint_frame_requires_all_columns():
    panel = _synthetic_panel()
    frame = build_joint_frame(panel)
    assert list(frame.columns) == ENDOG_VARS + EXOG_VARS
    bad = panel.drop(columns=["fiscal_impulse"])
    with pytest.raises(KeyError):
        build_joint_frame(bad)


def test_simulate_joint_shapes_and_stability():
    panel = _synthetic_panel()
    fit = _fake_joint_fit(panel)
    row = build_last_state_row(panel)
    paths = simulate_joint(fit, row, horizon=6, scenario="baseline", seed=1)
    assert paths.shape == (fit.n_draws, 6, len(ENDOG_VARS))
    assert np.all(np.isfinite(paths))
    # 12-quarter simulation stays bounded (stability in action)
    paths12 = simulate_joint(fit, row, horizon=12, seed=1)
    assert np.abs(paths12).max() < 50


def test_simulate_joint_scenarios_and_cumulative():
    panel = _synthetic_panel()
    fit = _fake_joint_fit(panel)
    row = build_last_state_row(panel)
    for name in SCENARIOS:
        paths = simulate_joint(fit, row, horizon=4, scenario=name, seed=2)
        assert np.all(np.isfinite(paths))
    with pytest.raises(ValueError):
        simulate_joint(fit, row, scenario="nope")

    cum = cumulative_returns(paths, fit, "gold_ret")
    sd = fit.scaler_endog.sds["gold_ret"]
    mean = fit.scaler_endog.means["gold_ret"]
    manual = paths[:, :, fit.endog_vars.index("gold_ret")].sum(axis=1) * sd + 4 * mean
    np.testing.assert_allclose(cum, manual)


def test_render_joint_forecast_report():
    panel = _synthetic_panel()
    fit = _fake_joint_fit(panel)
    row = build_last_state_row(panel)
    report = render_joint_forecast(fit, row, horizon=8, scenario="risk_off")
    for token in ("Causal Bayesian Macro Model", "risk_off", "Gold", "S&P 500",
                  "Treasury bonds (TLT)", "REITs (VNQ)", "P(positive)"):
        assert token in report
    assert "|" in report and "25–75%" in report
    assert "8 quarters" in report
    assert render_scenario_list().startswith("Available scenarios")


def test_summarize_cumulative():
    s = summarize_cumulative(np.array([-20.0, -5.0, 1.0, 3.0, 12.0]))
    assert s["median"] == 1.0
    assert s["p_positive"] == pytest.approx(0.6)
    assert s["p_below_10"] == pytest.approx(0.2)


def test_all_asset_vars_present_in_endog():
    for v in ASSET_VARS:
        assert v in ENDOG_VARS


# ---------------------------------------------------------------------------
# V1 gold model
# ---------------------------------------------------------------------------


def test_build_v1_frame_shape():
    panel = _synthetic_panel()
    df = build_v1_frame(panel)
    for c in V1_FEATURES:
        assert c in df.columns
    assert "target" in df.columns
    # target is the NEXT quarter's gold return
    gold = panel["gold_ret"]
    np.testing.assert_allclose(df["target"].iloc[:-1], gold.shift(-1).iloc[:-1])
    # feature row t uses only data from row t (gold_ret_lag1 == current row)
    np.testing.assert_allclose(df["gold_ret_lag1"], gold)


def test_build_v1_features_from_row():
    panel = _synthetic_panel()
    row = build_last_state_row(panel)
    feats = build_v1_features_from_row(row)
    assert set(V1_FEATURES) <= set(feats)
    assert feats["gold_ret_lag1"] == pytest.approx(float(panel["gold_ret"].iloc[-1]))
    assert all(np.isfinite(v) for v in feats.values())


@pytest.mark.slow
def test_v1_fit_predict_roundtrip(tmp_path):
    pymc = pytest.importorskip("pymc")
    panel = _synthetic_panel(periods=110)
    from tradingagents.models.macro_bayes.v1_gold import fit_gold_model

    fit = fit_gold_model(panel, draws=60, tune=120, chains=1, seed=5, include_gpr=False, min_obs=60)
    assert fit.n_obs >= 60
    assert fit.n_draws == 60
    # sign priors enforced: real-yield coefficient must be negative
    beta_ry = fit.posterior["beta"][:, fit.feature_names.index("real_yield_chg")]
    assert (beta_ry < 0).mean() > 0.95

    path = save_gold_model(fit, str(tmp_path / "v1_gold.pkl"))
    payload = load_model(path)
    fit2 = GoldModelFit.from_payload(payload)
    row = build_last_state_row(panel)
    preds = predict_gold(fit2, build_v1_features_from_row(row))
    assert preds.shape == (fit2.n_draws,)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# validation harness (metrics only; fitting is covered by the V1 test)
# ---------------------------------------------------------------------------


def test_log_score_and_bucket():
    samples = np.random.default_rng(0).normal(1.0, 2.0, 5000)
    ls = _log_score(samples, 1.0)
    assert np.isfinite(ls) and ls < 0
    bucket = {"log_score": [], "rmse": [], "direction": [], "cov50": [], "cov80": []}
    _score(bucket, samples, realized=1.5)
    assert bucket["log_score"] and len(bucket["rmse"]) == 1
    assert bucket["rmse"][0] == pytest.approx((float(np.mean(samples)) - 1.5) ** 2)
