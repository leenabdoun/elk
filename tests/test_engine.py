"""Correctness checks: ELK's numbers vs certified / independent reference calculations."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import diagnostics as D  # noqa: E402
from core.data_io import infer_frequency, make_dummies, read_table  # noqa: E402
from core.demo_data import make_private_credit  # noqa: E402
from core.market_data import _compound, align_by_period, attach_rf, parse_kf_csv  # noqa: E402
from core.models import ModelSpec, compare, fit_ols, out_of_sample, scenario  # noqa: E402
from core.suitability import assess  # noqa: E402

# NIST StRD Longley certified values (https://www.itl.nist.gov/div898/strd/lls/data/Longley.shtml)
LONGLEY = {
    "const": (-3482258.63459582, 890420.383607373),
    "GNPDEFL": (15.0618722713733, 84.9149257747669),
    "GNP": (-0.358191792925910e-01, 0.334910077722432e-01),
    "UNEMP": (-2.02022980381683, 0.488399681651699),
    "ARMED": (-1.03322686717359, 0.214274163161675),
    "POP": (-0.511041056535807e-01, 0.226073200069370),
    "YEAR": (1829.15146461355, 455.478499142212),
}


def longley():
    from statsmodels.datasets import longley as L
    d = L.load_pandas().data
    return d.rename(columns={"TOTEMP": "y"})


def test_ols_matches_nist_longley():
    d = longley()
    xs = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
    f = fit_ols(d, ModelSpec("y", xs))
    for k, (b, se) in LONGLEY.items():
        assert f.coef.loc[k, "estimate"] == pytest.approx(b, rel=1e-8)
        assert f.coef.loc[k, "std_error"] == pytest.approx(se, rel=1e-8)
    assert f.stats["r2"] == pytest.approx(0.995479004577296, rel=1e-10)
    assert f.stats["rmse"] == pytest.approx(304.854073561965, rel=1e-9)
    assert f.stats["f_stat"] == pytest.approx(330.285339234588, rel=1e-8)


def test_ols_matches_matrix_algebra():
    df = make_private_credit().dropna()
    xs = ["leverage_x", "log_ebitda", "ltv_pct", "sponsor_backed"]
    f = fit_ols(df, ModelSpec("spread_bps", xs))
    X = np.column_stack([np.ones(len(df)), df[xs].to_numpy()])
    y = df["spread_bps"].to_numpy()
    b = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    np.testing.assert_allclose(f.coef["estimate"].to_numpy(), b, rtol=1e-9)
    np.testing.assert_allclose(f.coef["std_error"].to_numpy(), se, rtol=1e-9)
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), len(y) - X.shape[1])
    np.testing.assert_allclose(f.coef["p_value"].to_numpy(), p, rtol=1e-7, atol=1e-14)
    # ANOVA identity
    a = f.anova.set_index("source")
    assert a.loc["Regression", "SS"] + a.loc["Residual", "SS"] == pytest.approx(a.loc["Total", "SS"], rel=1e-10)


def test_hc3_matches_manual_and_coefficients_unchanged():
    df = make_private_credit().dropna()
    xs = ["leverage_x", "log_ebitda"]
    a = fit_ols(df, ModelSpec("spread_bps", xs))
    h = fit_ols(df, ModelSpec("spread_bps", xs, "HC3"))
    np.testing.assert_allclose(a.coef["estimate"], h.coef["estimate"], rtol=1e-12)
    X = np.column_stack([np.ones(len(df)), df[xs].to_numpy()])
    XtXi = np.linalg.inv(X.T @ X)
    e = a.res.resid.to_numpy()
    hii = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = X.T @ (X * (e ** 2 / (1 - hii) ** 2)[:, None])
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    np.testing.assert_allclose(h.coef["std_error"].to_numpy(), se, rtol=1e-8)


def test_durbin_watson_manual():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=200)})
    d["y"] = 1 + 2 * d["x"] + rng.normal(size=200)
    f = fit_ols(d, ModelSpec("y", ["x"]))
    e = f.res.resid.to_numpy()
    dw = np.sum(np.diff(e) ** 2) / np.sum(e ** 2)
    assert D.run_core(f, 0.05, True, "monthly")["dw"]["stat"] == pytest.approx(dw, rel=1e-12)


def test_breusch_pagan_manual_koenker():
    df = make_private_credit().dropna()
    xs = ["leverage_x", "log_ebitda"]
    f = fit_ols(df, ModelSpec("spread_bps", xs))
    e2 = f.res.resid.to_numpy() ** 2
    X = np.column_stack([np.ones(len(df)), df[xs].to_numpy()])
    g = np.linalg.lstsq(X, e2, rcond=None)[0]
    r2 = 1 - np.sum((e2 - X @ g) ** 2) / np.sum((e2 - e2.mean()) ** 2)
    lm = len(e2) * r2  # Koenker (studentised) version
    assert D.run_core(f, 0.05, False, None)["bp"]["stat"] == pytest.approx(lm, rel=1e-8)


def test_vif_manual():
    df = make_private_credit().dropna()
    xs = ["leverage_x", "interest_coverage_x", "log_ebitda"]
    f = fit_ols(df, ModelSpec("spread_bps", xs))
    v = D.vif_table(f).set_index("variable")["VIF"]
    for x in xs:
        others = [o for o in xs if o != x]
        Z = np.column_stack([np.ones(len(df)), df[others].to_numpy()])
        g = np.linalg.lstsq(Z, df[x].to_numpy(), rcond=None)[0]
        r = df[x].to_numpy() - Z @ g
        r2 = 1 - r @ r / np.sum((df[x] - df[x].mean()) ** 2)
        assert v[x] == pytest.approx(1 / (1 - r2), rel=1e-8)


def test_demo_b_planted_problems_are_detected():
    df = make_private_credit()
    xs = ["leverage_x", "interest_coverage_x", "log_ebitda", "ltv_pct", "sponsor_backed"]
    s = assess(df, "spread_bps", xs, None, "loan_id")
    assert not s.blocked and s.metrics["n_complete"] == 294
    f = fit_ols(s.frame, ModelSpec("spread_bps", xs))
    d = D.run_core(f, 0.05, False, None)
    assert d["bp"]["level"] == "warn"
    assert D.vif_table(f)["VIF"].max() > 10
    assert len(D.influence_table(f, s.labels)[1]) > 0


def test_binary_outcome_blocked_and_constant_blocked():
    df = make_private_credit()
    assert assess(df, "default_flag", ["leverage_x"]).blocked
    d = df.assign(k=1.0)
    assert assess(d, "spread_bps", ["leverage_x", "k"]).blocked


def test_perfect_collinearity_and_dummy_trap_blocked():
    df = make_private_credit()
    d, made = make_dummies(df, "seniority", "First Lien")
    d["seniority_First_Lien"] = (d["seniority"] == "First Lien").astype(float)
    s = assess(d, "spread_bps", made + ["seniority_First_Lien"])
    assert s.blocked
    d["lev2"] = 2 * d["leverage_x"]
    assert assess(d, "spread_bps", ["leverage_x", "lev2"]).blocked


def test_too_few_rows_blocked():
    d = pd.DataFrame({"y": [1.0, 2.0, 3.0], "a": [1.0, 0.0, 2.0], "b": [0.0, 1.0, 5.0]})
    assert assess(d, "y", ["a", "b"]).blocked


def test_nested_f_test_matches_manual():
    df = make_private_credit().dropna()
    r = fit_ols(df, ModelSpec("spread_bps", ["leverage_x"]))
    u = fit_ols(df, ModelSpec("spread_bps", ["leverage_x", "log_ebitda", "ltv_pct"], label="Model B"))
    out = compare(r, u, 0.05)
    q = 2
    F = ((r.stats["ssr"] - u.stats["ssr"]) / q) / (u.stats["ssr"] / u.stats["df_resid"])
    t = out["tests"][0]
    assert t["name"] == "Nested-model F-test"
    assert t["stat"] == pytest.approx(F, rel=1e-10)
    assert t["p_value"] == pytest.approx(stats.f.sf(F, q, u.stats["df_resid"]), rel=1e-8)


def test_different_samples_warned():
    df = make_private_credit()
    a = fit_ols(df.dropna(subset=["leverage_x"]), ModelSpec("spread_bps", ["leverage_x"]))
    b = fit_ols(df.dropna(subset=["leverage_x", "ltv_pct"]), ModelSpec("spread_bps", ["leverage_x", "ltv_pct"]))
    out = compare(a, b, 0.05)
    assert not out["same_sample"] and out["warnings"] and not out["tests"]


def test_chronological_holdout_no_shuffle():
    n = 100
    d = pd.DataFrame({"x": np.arange(n, dtype=float)})
    d["y"] = d["x"] * 2.0
    d.loc[80:, "y"] += 1000  # break only in the test window
    res = out_of_sample(d, [ModelSpec("y", ["x"])], is_ts=True, holdout=0.2)
    assert res["rows"][0]["RMSE"] == pytest.approx(1000, rel=1e-6)
    assert res["rows"][0]["test observations"] == 20


def test_scenario_extrapolation_flag():
    df = make_private_credit().dropna()
    f = fit_ols(df, ModelSpec("spread_bps", ["leverage_x", "log_ebitda"]))
    r = scenario(f, {"leverage_x": 20.0, "log_ebitda": 3.0}, 0.05)
    assert r["outside"] == ["leverage_x"] and r["pi"][0] < r["mean"] < r["pi"][1]


KF_SAMPLE = """This file was created by CMPT_ME_BEME_RETS using the 202508 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
192609,    0.38,   -1.36,    0.05,    0.23

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.04,   -3.54,    3.12
"""


def test_ken_french_parser():
    d = parse_kf_csv(KF_SAMPLE)
    assert list(d.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(d) == 3
    assert d.index[0] == pd.Timestamp("1926-07-31")
    assert d.loc["1926-08-31", "HML"] == pytest.approx(0.0381)


def test_compound_rebuilds_market_from_total_return():
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    m = pd.DataFrame({"Mkt-RF": [0.01, 0.02, -0.01, 0.03, 0.0, 0.01], "RF": [0.004] * 6}, index=idx)
    q = _compound(m, "quarterly")
    mkt = (1 + m["Mkt-RF"] + m["RF"]).iloc[:3].prod() - 1
    rf = (1.004 ** 3) - 1
    assert q["Mkt-RF"].iloc[0] == pytest.approx(mkt - rf, rel=1e-12)
    assert q["RF"].iloc[0] == pytest.approx(rf, rel=1e-12)


def test_period_alignment_and_rf_attach():
    a = pd.DataFrame({"r": [0.01, 0.02]}, index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    b = pd.DataFrame({"f": [0.1, 0.2]}, index=pd.to_datetime(["2024-01-30", "2024-02-28"]))  # last trading days
    j = align_by_period(a, b, "monthly")
    assert len(j) == 2 and j["f"].tolist() == [0.1, 0.2]
    rf = pd.Series([0.003, 0.004], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    got = attach_rf(pd.Series(pd.to_datetime(["2024-02-15", "2024-01-03", "2023-12-01"])), rf, "monthly")
    assert got.iloc[0] == 0.004 and got.iloc[1] == 0.003 and np.isnan(got.iloc[2])


def test_frequency_detection():
    for f, rng in [("monthly", pd.date_range("2020-01-31", periods=24, freq="ME")),
                   ("weekly", pd.date_range("2020-01-03", periods=24, freq="W-FRI")),
                   ("daily", pd.bdate_range("2020-01-01", periods=50)),
                   ("quarterly", pd.date_range("2020-03-31", periods=12, freq="QE"))]:
        assert infer_frequency(pd.Series(rng))[0] == f


def test_read_csv_with_percent_text(tmp_path):
    raw = "Date,Fund,Mkt\n2024-01-31,1.5%,2%\n2024-02-29,-0.5%,1%\n2024-03-31,\"1,200\",3%\n"
    d = read_table(raw.encode(), "x.csv")
    s = assess(d, "Fund", ["Mkt"], "Date")
    assert s.frame["Fund"].tolist() == [1.5, -0.5, 1200.0]
    assert any("% signs" in i.title for i in s.issues)


def test_leakage_blocked():
    df = make_private_credit().dropna()
    df["spread_copy"] = df["spread_bps"]
    df["spread_after_fee"] = df["spread_bps"] - 25
    assert assess(df, "spread_bps", ["leverage_x", "spread_copy"]).blocked
    assert assess(df, "spread_bps", ["leverage_x", "spread_after_fee"]).blocked
    assert not assess(df, "spread_bps", ["leverage_x", "log_ebitda"]).blocked
