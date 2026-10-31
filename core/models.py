"""OLS estimation, robust inference, model comparison, out-of-sample checks and scenarios."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

COV_LABELS = {
    "nonrobust": "Classical (assumes constant variance, no autocorrelation)",
    "HC1": "Heteroskedasticity-robust (HC1 / White)",
    "HC3": "Heteroskedasticity-robust (HC3, better in small samples)",
    "HAC": "HAC / Newey–West (heteroskedasticity + autocorrelation)",
}


def newey_west_lags(n: int) -> int:
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


@dataclass
class ModelSpec:
    y: str
    xs: list[str]
    cov_type: str = "nonrobust"
    hac_lags: int | None = None
    exclude_rows: list = field(default_factory=list)  # index labels excluded as a sensitivity check
    label: str = "Model A"
    note: str = ""

    def describe(self) -> str:
        s = f"{self.y} ~ const + " + " + ".join(self.xs)
        return s


@dataclass
class FitResult:
    spec: ModelSpec
    res: object  # statsmodels RegressionResults
    frame: pd.DataFrame
    coef: pd.DataFrame
    stats: dict
    anova: pd.DataFrame


def fit_ols(frame: pd.DataFrame, spec: ModelSpec) -> FitResult:
    d = frame.drop(index=[i for i in spec.exclude_rows if i in frame.index])
    y = d[spec.y].astype(float)
    X = sm.add_constant(d[spec.xs].astype(float), has_constant="add")
    model = sm.OLS(y, X)
    # t / F reference distributions for every covariance type (small-sample convention)
    if spec.cov_type == "nonrobust":
        res = model.fit()
    elif spec.cov_type == "HAC":
        spec.hac_lags = int(spec.hac_lags or newey_west_lags(len(d)))
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": spec.hac_lags, "use_correction": True}, use_t=True)
    else:
        res = model.fit(cov_type=spec.cov_type, use_t=True)
    ci = res.conf_int()
    coef = pd.DataFrame({
        "estimate": res.params, "std_error": res.bse, "t_stat": res.tvalues, "p_value": res.pvalues,
        "ci_low": ci[0], "ci_high": ci[1]})
    coef.index.name = "term"
    n = int(res.nobs)
    k = int(res.df_model)
    ssr = float(res.ssr)
    tss = float(res.centered_tss)
    ess = tss - ssr
    df_r = float(res.df_resid)
    st = {
        "n": n, "k": k, "df_resid": df_r, "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj),
        "f_stat": float(np.squeeze(res.fvalue)) if res.fvalue is not None else np.nan,
        "f_pvalue": float(np.squeeze(res.f_pvalue)) if res.f_pvalue is not None else np.nan,
        "aic": float(res.aic), "bic": float(res.bic), "llf": float(res.llf),
        "rmse": float(np.sqrt(ssr / df_r)) if df_r > 0 else np.nan,
        "ssr": ssr, "ess": ess, "tss": tss, "cov_type": spec.cov_type, "hac_lags": spec.hac_lags,
        "f_label": "F-test (all slopes = 0)" if spec.cov_type == "nonrobust" else "Robust Wald F-test (all slopes = 0)",
    }
    anova = pd.DataFrame({
        "source": ["Regression", "Residual", "Total"],
        "SS": [ess, ssr, tss], "df": [k, df_r, n - 1],
        "MS": [ess / k if k else np.nan, ssr / df_r if df_r else np.nan, np.nan],
    })
    cl_f = (ess / k) / (ssr / df_r) if k and df_r else np.nan
    anova["F"] = [cl_f, np.nan, np.nan]
    anova["p_value"] = [stats.f.sf(cl_f, k, df_r) if k and df_r else np.nan, np.nan, np.nan]
    return FitResult(spec, res, d, coef, st, anova)


# ----------------------------------------------------------------------------- comparison
def same_sample(a: FitResult, b: FitResult) -> bool:
    return a.frame.index.equals(b.frame.index)


def nested_relation(a: FitResult, b: FitResult):
    """Return (restricted, unrestricted) if one model's regressors are a strict subset of the other's."""
    sa, sb = set(a.spec.xs), set(b.spec.xs)
    if sa < sb:
        return a, b
    if sb < sa:
        return b, a
    return None


def compare(a: FitResult, b: FitResult, alpha: float) -> dict:
    out = {"same_y": a.spec.y == b.spec.y, "same_sample": same_sample(a, b), "warnings": [], "tests": []}
    if not out["same_y"]:
        out["warnings"].append("The models explain different outcomes, so R², AIC and BIC are not comparable.")
    if not out["same_sample"]:
        out["warnings"].append(
            f"The models use different observations (A: {a.stats['n']}, B: {b.stats['n']}). AIC, BIC and F-tests "
            "are only valid on identical samples; differences may reflect the sample, not the specification.")
    rows = []
    for key, lab, better in [("adj_r2", "Adjusted R²", "higher"), ("aic", "AIC", "lower"), ("bic", "BIC", "lower"),
                             ("rmse", "Residual std. error", "lower"), ("r2", "R²", "higher"), ("n", "Observations", None),
                             ("k", "Slope parameters", None)]:
        va, vb = a.stats[key], b.stats[key]
        pref = ""
        if better and out["same_y"] and out["same_sample"] and key not in ("r2",):
            if np.isfinite(va) and np.isfinite(vb) and not np.isclose(va, vb):
                pref = "A" if (va > vb) == (better == "higher") else "B"
        rows.append({"metric": lab + (f" ({better} is better)" if better and key != "r2" else ""), "Model A": va,
                     "Model B": vb, "difference (B − A)": vb - va, "favours": pref})
    out["metrics"] = pd.DataFrame(rows)

    terms = list(dict.fromkeys(list(a.coef.index) + list(b.coef.index)))
    ct = pd.DataFrame(index=terms)
    for lab, m in (("A", a), ("B", b)):
        ct[f"coef {lab}"] = m.coef["estimate"]
        ct[f"SE {lab}"] = m.coef["std_error"]
        ct[f"p {lab}"] = m.coef["p_value"]
    ct["coef change"] = ct["coef B"] - ct["coef A"]
    ct["SE change"] = ct["SE B"] - ct["SE A"]
    ct.index.name = "term"
    out["coef_table"] = ct

    if set(a.spec.xs) == set(b.spec.xs) and out["same_sample"]:
        out["nested_note"] = ("Both models use the same variables and observations, so the coefficients are identical; "
                              "only the standard errors, t-statistics and p-values differ (an inference adjustment). "
                              "AIC, BIC and R² are therefore the same.")
        return out
    nest = nested_relation(a, b)
    if nest and out["same_y"] and out["same_sample"]:
        r, u = nest
        extra = sorted(set(u.spec.xs) - set(r.spec.xs))
        robust = next((m.spec for m in (u, r) if m.spec.cov_type != "nonrobust"), None)
        if robust is None:
            f, p, dfd = u.res.compare_f_test(r.res)
            out["tests"].append({
                "name": "Nested-model F-test", "stat": float(f), "p_value": float(p), "df": f"({int(dfd)}, {int(u.stats['df_resid'])})",
                "null": f"The extra variable(s) {', '.join(extra)} all have zero coefficients.",
                "reading": (f"Reject at {alpha:.0%}: the extra variables jointly add explanatory power."
                            if p < alpha else f"Do not reject at {alpha:.0%}: no evidence the extra variables jointly add explanatory power."),
                "caveat": "Valid when classical assumptions hold (constant variance, no autocorrelation)."})
        else:
            uu = fit_ols(u.frame, ModelSpec(u.spec.y, list(u.spec.xs), robust.cov_type, robust.hac_lags))
            R = np.zeros((len(extra), len(uu.res.params)))
            for i, t in enumerate(extra):
                R[i, list(uu.res.params.index).index(t)] = 1
            w = uu.res.wald_test(R, use_f=True, scalar=True)
            p = float(np.squeeze(w.pvalue))
            out["tests"].append({
                "name": f"Robust Wald test of the extra variables ({robust.cov_type})", "stat": float(np.squeeze(w.statistic)),
                "p_value": p, "df": f"({len(extra)}, {int(uu.stats['df_resid'])})",
                "null": f"The extra variable(s) {', '.join(extra)} all have zero coefficients.",
                "reading": (f"Reject at {alpha:.0%}: the extra variables jointly add explanatory power." if p < alpha
                            else f"Do not reject at {alpha:.0%}: no evidence the extra variables jointly add explanatory power."),
                "caveat": "A robust covariance is in use, so the classical nested F-test is replaced by a Wald test that "
                          "uses the larger model's robust covariance."})
    elif nest is None and set(a.spec.xs) != set(b.spec.xs):
        out["nested_note"] = ("The models are not nested (neither model's variables are a subset of the other's), "
                              "so no joint F-test is shown. Compare using AIC/BIC and out-of-sample error.")
    return out


def out_of_sample(frame: pd.DataFrame, specs: list[ModelSpec], is_ts: bool, holdout: float = 0.2, folds: int = 5,
                  seed: int = 11) -> dict:
    """Chronological holdout for time series (no shuffling); k-fold CV for cross-sectional data."""
    res = {"method": "", "rows": []}
    n = len(frame)
    if is_ts:
        cut = int(np.floor(n * (1 - holdout)))
        train, test = frame.iloc[:cut], frame.iloc[cut:]
        res["method"] = (f"Chronological holdout: estimated on the first {len(train)} observations, "
                         f"tested on the last {len(test)} (no shuffling).")
        splits = [(train, test)]
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        parts = np.array_split(order, folds)
        splits = [(frame.iloc[np.setdiff1d(order, p)], frame.iloc[p]) for p in parts]
        res["method"] = f"{folds}-fold cross-validation (random folds, fixed seed {seed}); data are cross-sectional."
    for s in specs:
        errs = []
        for tr, te in splits:
            if len(tr) <= len(s.xs) + 2 or len(te) == 0:
                continue
            Xtr = sm.add_constant(tr[s.xs], has_constant="add")
            Xte = sm.add_constant(te[s.xs], has_constant="add")
            m = sm.OLS(tr[s.y], Xtr).fit()
            errs.append(te[s.y].to_numpy() - m.predict(Xte).to_numpy())
        if not errs:
            res["rows"].append({"model": s.label, "RMSE": np.nan, "MAE": np.nan})
            continue
        e = np.concatenate(errs)
        res["rows"].append({"model": s.label, "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
                            "test observations": int(len(e))})
    res["table"] = pd.DataFrame(res["rows"])
    return res


# ----------------------------------------------------------------------------- scenario
def scenario(fit: FitResult, values: dict, alpha: float) -> dict:
    xs = fit.spec.xs
    row = pd.DataFrame([[1.0] + [float(values[x]) for x in xs]], columns=["const"] + xs)
    pred = fit.res.get_prediction(row)
    sf = pred.summary_frame(alpha=alpha)
    X = fit.frame[xs]
    outside = [x for x in xs if values[x] < X[x].min() or values[x] > X[x].max()]
    # hidden extrapolation: leverage of the scenario point vs the in-sample maximum
    Xm = sm.add_constant(X, has_constant="add").to_numpy()
    try:
        XtX_inv = np.linalg.pinv(Xm.T @ Xm)
        r0 = row.to_numpy(dtype=float)[0]
        h0 = float(r0 @ XtX_inv @ r0)
        hmax = float(np.max(np.einsum("ij,jk,ik->i", Xm, XtX_inv, Xm)))
    except Exception:
        h0, hmax = np.nan, np.nan
    return {"mean": float(sf["mean"].iloc[0]), "ci": (float(sf["mean_ci_lower"].iloc[0]), float(sf["mean_ci_upper"].iloc[0])),
            "pi": (float(sf["obs_ci_lower"].iloc[0]), float(sf["obs_ci_upper"].iloc[0])),
            "outside": outside, "leverage": h0, "max_leverage": hmax,
            "hidden_extrapolation": (not outside) and np.isfinite(h0) and h0 > hmax}


def summarize_change(a: FitResult, b: FitResult, alpha: float) -> dict:
    """Plain-language list of what differs between Model A and Model B."""
    inference_only = set(a.spec.xs) == set(b.spec.xs) and a.frame.index.equals(b.frame.index)
    common = [t for t in b.coef.index if t in a.coef.index]
    lines = []
    if inference_only:
        lines.append("**Coefficients: unchanged.** Same variables and observations, so OLS gives identical estimates, "
                     "R², AIC and residual diagnostics. This is expected for a standard-error adjustment.")
        se = (b.coef.loc[common, "std_error"] / a.coef.loc[common, "std_error"] - 1) * 100
        top = se.abs().sort_values(ascending=False).head(3)
        lines.append("**Standard errors: changed.** Largest moves: " + ", ".join(
            f"{'intercept' if t == 'const' else t} {se[t]:+.0f}%" for t in top.index) + ".")
        if b.spec.cov_type in ("HC1", "HC3"):
            lines.append("**Diagnostics: unchanged — as expected.** HC errors do not remove heteroskedasticity, so "
                         "Breusch–Pagan still flags it. They make the standard errors and p-values valid *despite* it; "
                         "the question HC answers is whether your conclusions survive.")
        elif b.spec.cov_type == "HAC":
            lines.append("**Diagnostics: unchanged — as expected.** HAC errors do not remove autocorrelation or "
                         "heteroskedasticity, so the tests still flag them. HAC makes the standard errors and p-values "
                         "valid *despite* them.")
    else:
        bc = b.coef.loc[common, "estimate"]
        ac = a.coef.loc[common, "estimate"]
        pct = ((bc - ac) / ac.abs().replace(0, np.nan) * 100).drop(index="const", errors="ignore")
        top = pct.abs().sort_values(ascending=False).head(3)
        if len(top):
            lines.append("**Coefficients: changed.** Largest moves: " + ", ".join(
                f"{t} {ac[t]:.4g} → {bc[t]:.4g} ({pct[t]:+.0f}%)" for t in top.index) + ".")
        flipped = [t for t in pct.index if np.sign(ac[t]) != np.sign(bc[t])]
        if flipped:
            lines.append(f"**Sign flipped:** {', '.join(flipped)}.")
        dropped = sorted(set(a.spec.xs) - set(b.spec.xs))
        added = sorted(set(b.spec.xs) - set(a.spec.xs))
        if dropped or added:
            lines.append("**Variables:** " + "; ".join(filter(None, [f"removed {', '.join(dropped)}" if dropped else "",
                                                                       f"added {', '.join(added)}" if added else ""])) + ".")
        if b.spec.exclude_rows:
            lines.append(f"**Observations:** {len(b.spec.exclude_rows)} flagged rows excluded (n {a.stats['n']} → {b.stats['n']}).")
        lines.append(f"**Fit:** adjusted R² {a.stats['adj_r2']:.3f} → {b.stats['adj_r2']:.3f}.")
    gained = [t for t in common if a.coef.loc[t, "p_value"] >= alpha > b.coef.loc[t, "p_value"]]
    lost = [t for t in common if a.coef.loc[t, "p_value"] < alpha <= b.coef.loc[t, "p_value"]]
    nm = lambda ts: ", ".join("intercept" if t == "const" else t for t in ts)  # noqa: E731
    if gained or lost:
        lines.append(f"**Significance at {alpha:.0%} changed:** "
                     + "; ".join(filter(None, [f"now significant: {nm(gained)}" if gained else "",
                                               f"no longer significant: {nm(lost)}" if lost else ""])) + ".")
    else:
        lines.append(f"**Significance at {alpha:.0%}: no change** — the same variables are significant in both models"
                     + (" (the conclusion is robust to this adjustment)." if inference_only else "."))
    return {"inference_only": inference_only, "lines": lines}
