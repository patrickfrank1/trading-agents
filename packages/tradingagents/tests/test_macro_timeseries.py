"""Tests for the quarterly macro panel builder (macro_timeseries)."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows import macro_timeseries as mt
from tradingagents.dataflows.macro_timeseries import (
    PanelBundle,
    _add_asset_returns,
    _add_derived,
    _qoq_annualized,
    _resample_quarterly,
    load_latest_panel,
    publication_lag_quarters,
    real_time_align,
    save_panel,
    _derived_registry,
)


def _quarterly_index(start="2015-01-01", periods=40):
    idx = pd.period_range(start, periods=periods, freq="Q").to_timestamp(how="end").normalize()
    return idx


def test_resample_quarterly_mean_eop_sum():
    daily = pd.Series(
        range(180),
        index=pd.date_range("2020-01-01", periods=180, freq="D"),
        dtype=float,
    )
    mean_q = _resample_quarterly(daily, "mean")
    eop_q = _resample_quarterly(daily, "eop")
    assert len(mean_q) == 2  # Q1 + Q2 of 2020
    assert mean_q.iloc[0] == pytest.approx(np.mean(range(91)))  # Jan 1..Mar 31
    assert eop_q.iloc[0] == 90.0
    with pytest.raises(ValueError):
        _resample_quarterly(daily, "bogus")


def test_qoq_annualized():
    idx = _quarterly_index(periods=3)
    s = pd.Series([100.0, 101.0, 102.01], index=idx)
    out = _qoq_annualized(s)
    assert out.iloc[1] == pytest.approx(1.01 ** 4 - 1, rel=1e-6)


def _fake_panel(periods=40):
    rng = np.random.default_rng(7)
    idx = _quarterly_index(periods=periods)
    df = pd.DataFrame(index=idx)
    df.index.name = "quarter"
    for col in ("real_gdp", "real_potential_gdp", "federal_receipts", "federal_outlays",
                "nominal_gdp", "federal_interest", "gross_federal_debt", "cpi", "cpi_core",
                "corporate_profits", "real_yield_10y", "treasury_10y", "breakeven_10y",
                "mortgage_rate_30y", "nfci", "hy_oas", "fed_balance_sheet",
                "hy_credit", "ig_credit"):
        df[col] = 100 + np.abs(rng.normal(size=periods)).cumsum() * (1 if col != "nominal_gdp" else 3)
    for name in ("gold", "sp500"):
        df[name] = 100 * np.abs(rng.normal(size=periods)).cumsum() + 100
    return df


def test_derived_series_values():
    df = _fake_panel()
    out = _add_derived(df)

    pot = out["real_potential_gdp"]
    np.testing.assert_allclose(out["output_gap"], (out["real_gdp"] - pot) / pot * 100)

    expected_pb = (out["federal_receipts"] - out["federal_outlays"]) / out["nominal_gdp"] * 100
    np.testing.assert_allclose(out["primary_balance_pct_gdp"], expected_pb)

    expected_r = 4.0 * out["federal_interest"] / out["gross_federal_debt"].shift(1) * 100
    np.testing.assert_allclose(out["r_eff"].iloc[1:], expected_r.iloc[1:])

    expected_fi = -out["primary_balance_pct_gdp"].diff()
    np.testing.assert_allclose(out["fiscal_impulse"].iloc[1:], expected_fi.iloc[1:])

    assert "output_gap" in out.columns and "inflation_surprise" in out.columns


def test_asset_returns_added():
    df = _fake_panel()
    out = _add_asset_returns(df)
    for name in ("gold", "sp500"):
        assert f"{name}_ret" in out.columns
        manual = df[name].pct_change() * 100
        np.testing.assert_allclose(out[f"{name}_ret"].iloc[1:], manual.iloc[1:])


def test_real_time_alignment_lags():
    df = _add_derived(_add_asset_returns(_fake_panel(periods=12)))
    rt = real_time_align(df)
    # market data (lag 0) unchanged
    pd.testing.assert_series_equal(rt["gold_ret"], df["gold_ret"])
    # cpi has a 1-month publication lag -> ceil(1/3) = 1 quarter shift
    assert rt["cpi"].iloc[1] == df["cpi"].iloc[0]
    assert np.isnan(rt["cpi"].iloc[0])
    # fiscal series lag 3 months -> 1 quarter
    assert rt["federal_receipts"].iloc[1] == df["federal_receipts"].iloc[0]
    # derived cpi_qoq_ann: parent lag 1 + internal extra lag 1 -> 2 quarters
    assert rt["cpi_qoq_ann"].iloc[3] == df["cpi_qoq_ann"].iloc[1]


def test_publication_lag_registry_complete():
    df = _add_derived(_add_asset_returns(_fake_panel(periods=6)))
    lags = publication_lag_quarters(list(df.columns))
    assert set(lags) == set(df.columns)
    assert all(isinstance(v, int) and v >= 0 for v in lags.values())
    for name in _derived_registry():
        assert name in lags


def test_panel_cache_roundtrip(tmp_path, monkeypatch):
    real_cfg = dict(mt.get_config())
    monkeypatch.setattr(
        mt,
        "get_config",
        lambda: {**real_cfg, "data_cache_dir": str(tmp_path)},
    )
    df = _fake_panel(periods=12)
    bundle = PanelBundle(data=real_time_align(df), full=df, manifest={"n_quarters": 12})
    path = save_panel(bundle)
    assert path and path.startswith(str(tmp_path))

    loaded = load_latest_panel(max_age_days=None)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.data, bundle.data, check_freq=False)
    pd.testing.assert_frame_equal(loaded.full, bundle.full, check_freq=False)
