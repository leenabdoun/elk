"""Diagnostic tests. Each returns a plain dict that states what is tested, the null, and the limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_arch, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import acf, adfuller

DEFAULT_LAGS = {"daily": 5, "weekly": 4, "monthly": 12, "quarterly": 4, "annual": 2}

EXPLAIN = {
    "dw": {"what": "Whether each residual is correlated with the one before it (first-order autocorrelation).",
           "null": "No first-order autocorrelation (statistic near 2).",
           "read": "Values well below 2 suggest positive autocorrelation; well above 2 suggest negative. "
                   "Roughly 1.5–2.5 is usually unremarkable.",
           "limits": "An indicator, not a formal p-value test here. Only meaningful when rows are in time order. "
                     "Not valid with a lagged dependent variable."},
    "bg": {"what": "Serial correlation in residuals at several lags at once.",
           "null": "No autocorrelation in residuals up to the chosen lag.",
           "read": "A small p-value means residuals are serially correlated, so classical standard errors are "
                   "likely too small. HAC (Newey–West) standard errors are the usual inference remedy.",
           "limits": "Requires ordered, equally spaced observations. Result depends on lag choice."},
    "bp": {"what": "Whether residual variance changes with the explanatory variables (heteroskedasticity).",
           "null": "Residual variance is constant (homoskedastic).",
           "read": "A small p-value means variance is not constant: coefficients are still unbiased, but classical "
                   "standard errors and p-values are unreliable. Heteroskedasticity-robust (HC) errors address inference.",
           "limits": "Detects variance linked linearly to the regressors; other patterns may be missed. "
                     "Sensitive to non-normal residuals (studentised version used)."},
    "jb": {"what": "Whether residuals look normally distributed (skewness and kurtosis).",
           "null": "Residuals are normally distributed.",
           "read": "Rejection matters most in small samples, where t- and F-tests rely on normality. "
                   "In large samples OLS inference is approximately valid without it.",
           "limits": "Very sensitive in large samples (tiny departures are flagged); fat tails are normal for returns."},
    "reset": {"what": "Whether the linear form misses curvature (Ramsey RESET).",
              "null": "The model's functional form is adequate (no omitted powers of fitted values).",
              "read": "A small p-value suggests nonlinearity or missing interactions; consider transformations.",
              "limits": "Says something is missing, not what."},
    "vif": {"what": "How much each coefficient's variance is inflated by correlation with the other regressors.",
            "null": "Not a hypothesis test.",
            "read": "Rule of thumb: above 5 is notable, above 10 is high. High VIF makes individual coefficients "
                    "imprecise and unstable, even when the model as a whole fits well.",
            "limits": "An indicator, not proof the model is invalid. Predictions can still be fine. "
                      "Dropping a variable to cut VIF can introduce omitted-variable bias."},
    "influence": {"what": "Observations that are unusual (studentised residual) or that move the estimates a lot (Cook's distance).",
                  "null": "Not a hypothesis test.",
                  "read": "Flags: |studentised residual| > 3, or Cook's distance > 4/n. Review these rows: data errors, "
                          "genuine extreme events, or a sign the model is missing something.",
                  "limits": "ELK never deletes data. Excluding rows is only offered as a labelled sensitivity check."},
    "adf": {"what": "Whether a series has a unit root (behaves like a random walk, e.g. price levels).",
            "null": "The series has a unit root (non-stationary).",
            "read": "A small p-value rejects the unit root, supporting stationarity. Regressing one non-stationary "
                    "series on another can produce spurious relationships.",
            "limits": "Low power in short samples. Returns are usually stationary; price levels usually are not. "
                      "Result depends on the constant/trend and lag choices."},
    "arch": {"what": "Volatility clustering: whether large residuals tend to follow large residuals (ARCH effects).",
             "null": "No ARCH effects up to the chosen lag.",
             "read": "A small p-value indicates time-varying volatility, common in financial returns. "
                     "HAC/robust errors help inference; volatility models (GARCH) are a future module.",
             "limits": "Needs ordered data and a reasonable sample (dozens of observations or more)."},
}


def _verdict(p, alpha, bad_if_reject=True):
    if p is None or not np.isfinite(p):
        return "na"
    return ("warn" if bad_if_reject else "ok") if p < alpha else ("ok" if bad_if_reject else "warn")


def run_core(fit, alpha: float, is_ts: bool, freq: str | None, bg_lags: int | None = None,
             arch_lags: int | None = None) -> dict:
    res = fit.res
    resid = res.resid
    exog = res.model.exog
    n = int(res.nobs)
    out = {}

    jb, jbp, skew, kurt = jarque_bera(resid)
    out["jb"] = dict(name="Jarque–Bera normality", stat=float(jb), p=float(jbp), level=_verdict(jbp, alpha),
                     extra=f"skewness {skew:.2f}, kurtosis {kurt:.2f} (normal = 3)", key="jb")

    try:
        lm, lmp, fval, fp = het_breuschpagan(resid, exog, robust=True)
        out["bp"] = dict(name="Breusch–Pagan heteroskedasticity", stat=float(lm), p=float(lmp),
                         level=_verdict(lmp, alpha), extra=f"LM statistic, {exog.shape[1] - 1} df", key="bp")
    except Exception as e:
        out["bp"] = dict(name="Breusch–Pagan heteroskedasticity", stat=np.nan, p=np.nan, level="na", extra=str(e), key="bp")

    try:
        r = linear_reset(res, power=3, use_f=True)
        out["reset"] = dict(name="Ramsey RESET functional form", stat=float(np.squeeze(r.fvalue)),
                            p=float(np.squeeze(r.pvalue)), level=_verdict(float(np.squeeze(r.pvalue)), alpha),
                            extra="adds squared and cubed fitted values", key="reset")
    except Exception as e:
        out["reset"] = dict(name="Ramsey RESET functional form", stat=np.nan, p=np.nan, level="na", extra=str(e), key="reset")

    if is_ts:
        dw = float(durbin_watson(resid))
        lvl = "ok" if 1.5 <= dw <= 2.5 else "warn"
        out["dw"] = dict(name="Durbin–Watson", stat=dw, p=np.nan, level=lvl,
                         extra="≈2 none · <2 positive · >2 negative autocorrelation", key="dw")
        lags = bg_lags or min(DEFAULT_LAGS.get(freq or "", 4), max(1, n // 5))
        try:
            lm, lmp, f, fp = acorr_breusch_godfrey(res, nlags=lags, result_object=False)
            out["bg"] = dict(name="Breusch–Godfrey serial correlation", stat=float(lm), p=float(lmp),
                             level=_verdict(lmp, alpha), extra=f"{lags} lag(s)", key="bg", lags=lags)
        except Exception as e:
            out["bg"] = dict(name="Breusch–Godfrey serial correlation", stat=np.nan, p=np.nan, level="na", extra=str(e), key="bg")
        alags = arch_lags or min(DEFAULT_LAGS.get(freq or "", 4), max(1, n // 10))
        if n >= 30:
            try:
                lm, lmp, f, fp = het_arch(resid, nlags=alags, result_object=False)
                out["arch"] = dict(name="ARCH LM volatility clustering", stat=float(lm), p=float(lmp),
                                   level=_verdict(lmp, alpha), extra=f"{alags} lag(s)", key="arch", lags=alags)
            except Exception as e:
                out["arch"] = dict(name="ARCH LM volatility clustering", stat=np.nan, p=np.nan, level="na", extra=str(e), key="arch")
        else:
            out["arch"] = dict(name="ARCH LM volatility clustering", stat=np.nan, p=np.nan, level="na",
                               extra="needs at least 30 observations", key="arch")
    return out


def vif_table(fit) -> pd.DataFrame:
    xs = fit.spec.xs
    if len(xs) < 2:
        return pd.DataFrame(columns=["variable", "VIF", "level"])
    X = fit.res.model.exog
    rows = []
    for i, name in enumerate(fit.res.model.exog_names):
        if name == "const":
            continue
        v = float(variance_inflation_factor(X, i))
        rows.append({"variable": name, "VIF": v, "level": "high" if v > 10 else ("moderate" if v > 5 else "low")})
    return pd.DataFrame(rows)


def influence_table(fit, labels: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    inf = OLSInfluence(fit.res)
    n = int(fit.res.nobs)
    df = pd.DataFrame({
        "row": labels.reindex(fit.frame.index).values if labels is not None else fit.frame.index.astype(str),
        "fitted": fit.res.fittedvalues.values, "actual": fit.frame[fit.spec.y].values,
        "residual": fit.res.resid.values,
        "studentized_resid": inf.resid_studentized_external,
        "leverage": inf.hat_matrix_diag,
        "cooks_d": inf.cooks_distance[0],
    }, index=fit.frame.index)
    thr = 4.0 / n
    df["flag"] = np.where((df["studentized_resid"].abs() > 3) & (df["cooks_d"] > thr), "outlier + influential",
                          np.where(df["studentized_resid"].abs() > 3, "outlier",
                                   np.where(df["cooks_d"] > thr, "influential", "")))
    flagged = df[df["flag"] != ""].sort_values("cooks_d", ascending=False)
    return df, flagged


def adf_test(series: pd.Series, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> dict:
    s = pd.Series(series).dropna()
    if len(s) < 20:
        return {"stat": np.nan, "p": np.nan, "lags": None, "nobs": len(s), "crit": {}, "note": "needs at least 20 observations"}
    stat, p, lags, nobs, crit, *_ = adfuller(s, regression=regression, autolag=autolag, maxlag=maxlag)
    return {"stat": float(stat), "p": float(p), "lags": int(lags), "nobs": int(nobs), "crit": crit, "note": ""}


def residual_acf(fit, nlags: int = 20):
    r = fit.res.resid
    nlags = min(nlags, len(r) // 2 - 1)
    vals = acf(r, nlags=nlags, fft=True)
    band = 1.96 / np.sqrt(len(r))
    return np.arange(len(vals)), vals, band
