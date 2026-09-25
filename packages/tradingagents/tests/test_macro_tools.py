"""Tests for counterfactual / sensitivity tooling (asset_regressions + tools)."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.models.macro_bayes.asset_regressions import (
    AssetRegressionFit,
    REGRESSION_SPECS,
    _latest_available,
    counterfactual_samples,
    parse_shocks,
    predict_asset,
    render_counterfactual_report,
    render_sensitivity_table,
)
from tradingagents.models.macro_bayes.common import Scaler


def _fake_fit(target="sp500_ret", label="S&P 500 (SPY)", features=("real_yield_chg", "usd_ret")):
    scaler = Scaler({"real_yield_chg": 0.0, "usd_ret": 0.0}, {"real_yield_chg": 0.3, "usd_ret": 4.0})
    rng = np.random.default_rng(0)
    return AssetRegressionFit(
        target=target, label=label, price_col="sp500",
        features=list(features), scaler=scaler,
        posterior={
            "alpha": np.full(20, 1.0),
            "beta": np.column_stack([
                np.full(20, -1.0),   # real_yield_chg: -1pp per +1sd
                np.full(20, -0.5),   # usd_ret
            ]),
            "sigma": np.full(20, 1.0),
            "nu": np.full(20, 8.0),
        },
        n_obs=100,
    )


def _panel():
    idx = pd.period_range("2020-01-01", periods=8, freq="Q").to_timestamp(how="end").normalize()
    df = pd.DataFrame(index=idx)
    df["real_yield_chg"] = np.linspace(0.1, 0.2, 8)
    df["usd_ret"] = np.linspace(1.0, 2.0, 8)
    df["sp500_ret"] = np.linspace(2.0, 3.0, 8)
    return df


def test_latest_available_handles_nan_and_missing():
    df = _panel()
    df.loc[df.index[-1], "real_yield_chg"] = np.nan
    assert _latest_available(df, "real_yield_chg") == pytest.approx(df["real_yield_chg"].iloc[-2])
    assert _latest_available(df, "nope") == 0.0


def test_parse_shocks_native_sd_and_errors():
    valid = {"real_yield_chg", "usd_ret"}
    sds = {"real_yield_chg": 0.3, "usd_ret": 4.0}
    out = parse_shocks("real_yield_chg:0.27, usd_ret:-1sd", valid, sds)
    assert out["real_yield_chg"] == {"value": 0.27, "unit": "native"}
    assert out["usd_ret"] == {"value": -1.0, "unit": "sd"}
    with pytest.raises(ValueError):
        parse_shocks("fed_funds:0.5", valid, sds)
    with pytest.raises(ValueError):
        parse_shocks("real_yield_chg", valid, sds)
    with pytest.raises(ValueError):
        parse_shocks("real_yield_chg:2.0", valid, sds)      # > 3sd guardrail
    with pytest.raises(ValueError):
        parse_shocks("usd_ret:5sd", valid, sds)
    with pytest.raises(ValueError):
        parse_shocks("", valid, sds)


def test_counterfactual_samples_semantics():
    panel = _panel()
    fits = {"sp500_ret": _fake_fit()}
    shocks = {"real_yield_chg": {"value": 0.27, "unit": "native"}}
    res = counterfactual_samples(fits, panel, shocks)
    r = res["sp500_ret"]
    assert r["baseline"].shape == r["shock"].shape == (20,)

    # baseline zeroes the shocked variable; usd stays at its latest value
    base_feats = {"real_yield_chg": 0.0, "usd_ret": 2.0}
    expected_base = predict_asset(fits["sp500_ret"], base_feats)
    np.testing.assert_allclose(r["baseline"], expected_base)

    # shock: +0.27pp on real yield = +0.9sd -> median effect = -1.0 * 0.9 = -0.9pp
    assert float(np.median(r["shock"]) - np.median(r["baseline"])) == pytest.approx(-0.9, abs=1e-6)
    assert r["affected"] is True

    # sd-unit shock: usd_ret latest(2.0) + 1sd(4.0) = 6.0, baseline no-move(0.0)
    # -> delta = 1.5sd -> median effect -0.5 * 1.5 = -0.75pp
    shocks2 = {"usd_ret": {"value": 1.0, "unit": "sd"}}
    r2 = counterfactual_samples(fits, panel, shocks2)["sp500_ret"]
    assert float(np.median(r2["shock"]) - np.median(r2["baseline"])) == pytest.approx(-0.75, abs=1e-6)


def test_counterfactual_beta_surrogate_row():
    panel = _panel()
    fits = {"sp500_ret": _fake_fit()}
    res = counterfactual_samples(fits, panel, {"real_yield_chg": {"value": 0.3, "unit": "native"}}, stock_beta=1.2)
    assert "stock_beta" in res
    np.testing.assert_allclose(res["stock_beta"]["shock"], res["sp500_ret"]["shock"] * 1.2)
    assert "beta=1.2" in res["stock_beta"]["label"]


def test_unaffected_asset_flagged():
    fit = AssetRegressionFit(
        target="x", label="X", price_col="sp500",
        features=["usd_ret"], scaler=Scaler({"usd_ret": 0.0}, {"usd_ret": 4.0}),
        posterior={"alpha": np.zeros(5), "beta": np.full((5, 1), 0.5),
                   "sigma": np.ones(5), "nu": np.full(5, 8.0)},
    )
    res = counterfactual_samples({"x": fit}, _panel(), {"real_yield_chg": {"value": 0.3, "unit": "native"}})
    assert res["x"]["affected"] is False
    np.testing.assert_allclose(res["x"]["baseline"], res["x"]["shock"])


def test_render_counterfactual_report():
    panel = _panel()
    res = counterfactual_samples({"sp500_ret": _fake_fit()}, panel,
                                 {"real_yield_chg": {"value": 0.27, "unit": "native"}}, stock_beta=1.0)
    out = render_counterfactual_report(res, {"real_yield_chg": {"value": 0.27, "unit": "native"}},
                                       {"sp500": 700.0}, horizon=1, joint_block=None)
    for token in ("Counterfactual", "S&P 500 (SPY)", "Stock (beta=1", "ASSUMPTION", "same quarter"):
        assert token in out


def test_render_sensitivity_table():
    fits = {"sp500_ret": _fake_fit(), "gold_ret": _fake_fit(target="gold_ret", label="Gold",
                                                            features=("real_yield_chg",))}
    out = render_sensitivity_table(fits, gold_fit=None)
    assert "Driver Sensitivities" in out
    assert "real_yield_chg (1sd = 0.30)" in out
    assert out.count("—") >= 1  # gold has no usd_ret loading


def test_all_specs_have_targets_in_panel_columns():
    panel = _panel()
    panel["gold_ret"] = 1.0
    panel["reits_ret"] = 1.0
    panel["treasury_bond_ret"] = 1.0
    panel["mortgage_rate_chg"] = 1.0
    panel["profits_growth"] = 1.0
    panel["nfci_chg"] = 1.0
    panel["inflation_surprise"] = 1.0
    for spec in REGRESSION_SPECS.values():
        for f in spec.features:
            assert f in panel.columns, f"{spec.target} missing feature {f}"


def test_tool_counterfactual_without_regressions(monkeypatch, tmp_path):
    import tradingagents.agents.utils.macro_data_tools as mdt

    bundle = type("B", (), {"data": _panel(), "full": _panel(), "manifest": {}})()
    monkeypatch.setattr(mdt, "load_latest_panel", lambda *a, **k: bundle)
    monkeypatch.setattr(mdt, "model_dir", lambda: str(tmp_path))
    out = mdt.get_macro_causal_forecast_impl(shocks="real_yield_chg:0.27")
    assert "no fitted asset regressions" in out


def test_tool_invalid_shock_message(monkeypatch, tmp_path):
    import tradingagents.agents.utils.macro_data_tools as mdt

    bundle = type("B", (), {"data": _panel(), "full": _panel(), "manifest": {}})()
    monkeypatch.setattr(mdt, "load_latest_panel", lambda *a, **k: bundle)
    monkeypatch.setattr(mdt, "model_dir", lambda: str(tmp_path))
    p = _fake_fit()
    payload = {"target": p.target, "label": p.label, "price_col": p.price_col,
               "features": p.features, "scaler": p.scaler.to_dict(),
               "posterior": p.posterior, "n_obs": p.n_obs}
    pkl = tmp_path / "asset_regressions.pkl"
    from tradingagents.models.macro_bayes.asset_regressions import save_asset_regressions

    save_asset_regressions({"sp500_ret": payload}, str(pkl))
    out = mdt.get_macro_causal_forecast_impl(shocks="bogus_var:1.0")
    assert "Invalid counterfactual specification" in out
