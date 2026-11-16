"""Forecast evaluation: no look-ahead, sensible benchmarks, correct metrics and tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.demo_data import make_private_credit  # noqa: E402
from core.forecast import default_timing, dm_test, evaluate  # noqa: E402


def ar_data(n=300, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + 0.8 * x[t - 1] + rng.normal()
    return pd.DataFrame({"y": y, "x": x, "z": rng.normal(size=n)})


def row(res, model):
    t = res["table"]
    return t[t["Model"] == model].iloc[0]


def test_future_values_never_change_earlier_forecasts():
    d = ar_data()
    r1 = evaluate(d, "y", ["x", "z"], True, timing="forecast")
    d2 = d.copy()
    d2.iloc[-1, :] = [1e6, -1e6, 1e6]  # wreck the last row only
    r2 = evaluate(d2, "y", ["x", "z"], True, timing="forecast")
    s1, s2 = r1["series"].iloc[:-1], r2["series"].iloc[:-1]
    pd.testing.assert_frame_equal(s1.drop(columns="actual"), s2.drop(columns="actual"))


def test_same_period_drivers_flatter_the_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    d = pd.DataFrame({"y": 2 * x + 0.01 * rng.normal(size=200), "x": x})
    known = evaluate(d, "y", ["x"], True, timing="known")
    fc = evaluate(d, "y", ["x"], True, timing="forecast")
    assert row(known, "Your regression")["RMSE"] < 0.05
    assert row(fc, "Your regression")["RMSE"] > 1.0  # x is unpredictable, so last period's x is useless
    assert "not a forecast you could make in advance" in known["notes"][0]


def test_benchmark_adapts_to_data():
    rng = np.random.default_rng(1)
    level = pd.DataFrame({"y": np.cumsum(rng.normal(size=300)) + 100, "x": rng.normal(size=300)})
    rets = pd.DataFrame({"y": rng.normal(size=300), "x": rng.normal(size=300)})
    r_level = evaluate(level, "y", ["x"], True)
    r_rets = evaluate(rets, "y", ["x"], True)
    assert r_level["benchmark"]["key"] == "RW" and r_rets["benchmark"]["key"] == "MEAN"
    s = r_level["series"]
    prev = level["y"].shift(1).loc[s.index]
    np.testing.assert_allclose(s["Benchmark: last period's value"], prev)
    # expanding historical mean uses only earlier rows (row 0 is lost to the lag)
    s2 = r_rets["series"]
    for idx in s2.index[:5]:
        pos = rets.index.get_loc(idx)
        assert s2.loc[idx, "Benchmark: historical average"] == pytest.approx(rets["y"].iloc[1:pos].mean())


def test_metrics_and_improvement_formula():
    d = ar_data()
    r = evaluate(d, "y", ["x", "z"], True)
    s = r["series"]
    b = r["table"].iloc[0]
    for _, m in r["table"].iloc[1:].iterrows():
        e = s["actual"] - s[m["Model"]]
        assert m["MAE"] == pytest.approx(np.mean(np.abs(e)))
        assert m["RMSE"] == pytest.approx(np.sqrt(np.mean(e ** 2)))
        assert m["RMSE improvement (%)"] == pytest.approx((1 - m["RMSE"] / b["RMSE"]) * 100)
        assert m["Out-of-sample R²"] == pytest.approx(1 - (m["RMSE"] / b["RMSE"]) ** 2)


def test_persistent_outcome_alternative_wins():
    r = evaluate(ar_data(600), "y", ["x", "z"], True)
    assert r["best"] == "Your regression + last period's outcome"
    assert r["verdict"].tone == "good"


def test_unpredictable_outcome_does_not_beat_benchmark():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"y": rng.normal(size=400), "x": rng.normal(size=400), "z": rng.normal(size=400)})
    r = evaluate(d, "y", ["x", "z"], True)
    assert r["verdict"].tone in ("problem", "caution")
    assert "Yes" not in r["table"]["Beats benchmark?"].tolist()


def test_dm_matches_manual():
    rng = np.random.default_rng(2)
    e1, e2 = rng.normal(size=80), 1.3 * rng.normal(size=80)
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    stat = d.mean() / np.sqrt(d.var() / T) * np.sqrt((T - 1) / T)
    s, p = dm_test(e1, e2)
    assert s == pytest.approx(stat) and p == pytest.approx(2 * stats.t.sf(abs(stat), T - 1))
    assert np.isnan(dm_test(e1, e1)[1])


def test_already_lagged_columns_not_lagged_twice():
    d = ar_data()
    d["lag1(x)"] = d["x"].shift(1)
    d = d.dropna()
    a = evaluate(d, "y", ["lag1(x)"], True, timing="forecast")
    b = evaluate(d, "y", ["lag1(x)"], True, timing="known")
    np.testing.assert_allclose(a["series"]["Your regression"], b["series"]["Your regression"])
    assert default_timing(["lag1(x)"]) == "known" and default_timing(["x"]) == "forecast"


def test_model_b_only_when_drivers_differ():
    d = ar_data()
    same = evaluate(d, "y", ["x", "z"], True, b_xs=["z", "x"])
    diff = evaluate(d, "y", ["x", "z"], True, b_xs=["x"])
    assert "Model B" not in same["table"]["Model"].tolist()
    assert "Model B" in diff["table"]["Model"].tolist()


def test_cross_section_every_row_predicted_once():
    df = make_private_credit().dropna()
    r = evaluate(df, "spread_bps", ["leverage_x", "log_ebitda", "ltv_pct"], False, unit="basis points")
    assert r["n_test"] == len(df) and r["benchmark"]["key"] == "MEAN"
    assert r["timing"] is None
    assert row(r, "Your regression")["RMSE"] < r["table"].iloc[0]["RMSE"]


def test_fixed_estimation_matches_single_holdout():
    n = 100
    d = pd.DataFrame({"x": np.arange(n, dtype=float)})
    d["y"] = d["x"] * 2.0
    d.loc[80:, "y"] += 1000
    r = evaluate(d, "y", ["x"], True, timing="known", refit=False)
    assert row(r, "Your regression")["RMSE"] == pytest.approx(1000, rel=1e-6)


def test_too_little_data_is_reported():
    d = ar_data(12)
    assert not evaluate(d, "y", ["x", "z"], True)["ok"]
