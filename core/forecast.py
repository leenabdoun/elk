"""Out-of-sample forecast evaluation: does the model forecast better than a simple benchmark?

This is kept separate from robustness checks (HC/HAC standard errors, excluded rows). Those change how far the
p-values can be trusted, not the predictions, so they have nothing to add to a forecasting comparison.

What is compared, adapted to the data:
- Naive benchmark. Time series: the last observed value (when the outcome behaves like a slowly moving level) or the
  historical average (when it moves around a mean, e.g. returns). The rule is applied to the training data only.
  Cross-sectional data: the training average.
- Your regression (Model A's drivers).
- One automatic alternative. Time series: your regression plus last period's outcome. Otherwise (or if that lag is
  already a driver): your regression keeping only the drivers that are clearly useful in the training data.
- Model B, when it uses a different set of drivers.

No look-ahead: time series are split chronologically, every model is re-estimated using only the rows before each
forecast (expanding window), the benchmark rule and the driver selection are re-made inside each training window, and in
"forecast" timing each driver enters with its previous-period value. Cross-sectional data use k-fold cross-validation.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .findings import Finding, amount
from .insights import p_eq

YLAG = "__ylag"
TIMING = {
    "forecast": "Not known yet: use last period's driver values (a real forecast)",
    "known": "Known in advance: use same-period driver values (tests the relationship on unseen data)",
}
LAG_NAME = re.compile(r"^(lag\d*|l\d+)[(_.]", re.I)


def already_lagged(col: str) -> bool:
    """Columns ELK created with 'Previous row's value' (lag1(x)) or named like lag_x are already known in advance."""
    return bool(LAG_NAME.match(col))


def default_timing(xs: list[str]) -> str:
    return "known" if xs and all(already_lagged(x) for x in xs) else "forecast"


# ----------------------------------------------------------------------------- small OLS helpers
def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def _pred(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return b[0] + X @ b[1:]


def _select(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Indices of drivers with p < alpha in an OLS fit on this training window (classical standard errors)."""
    Z = np.column_stack([np.ones(len(X)), X])
    n, k = Z.shape
    if n <= k + 1:
        return np.arange(X.shape[1])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    s2 = e @ e / (n - k)
    se = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(Z.T @ Z)), 0, None))[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(b[1:] / se), n - k)
    return np.flatnonzero(np.nan_to_num(p, nan=1.0) < alpha)


def dm_test(e_model: np.ndarray, e_ref: np.ndarray) -> tuple[float, float]:
    """Diebold–Mariano test of equal squared-error accuracy, one-step forecasts, Harvey–Leybourne–Newbold correction.

    Negative statistic = the model is more accurate than the reference. Returns (stat, two-sided p)."""
    d = np.asarray(e_model) ** 2 - np.asarray(e_ref) ** 2
    T = len(d)
    if T < 5 or np.allclose(d, 0):
        return np.nan, np.nan
    v = d.var()
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / T) * np.sqrt((T - 1) / T)
    return float(stat), float(2 * stats.t.sf(abs(stat), T - 1))


# ----------------------------------------------------------------------------- engine
def _candidates(xs, b_xs, is_ts, has_ylag):
    c = [{"key": "A", "label": "Your regression", "xs": list(xs), "select": False,
          "uses": "Your drivers: " + ", ".join(xs)}]
    if is_ts and not has_ylag:
        c.append({"key": "AR", "label": "Your regression + last period's outcome", "xs": list(xs) + [YLAG],
                  "select": False, "uses": "Your drivers plus the outcome's previous value"})
    elif len(xs) >= 2:
        c.append({"key": "SEL", "label": "Your regression, clearly useful drivers only", "xs": list(xs), "select": True,
                  "uses": "Only drivers significant in each training window"})
    if b_xs and set(b_xs) != set(xs):
        c.append({"key": "B", "label": "Model B", "xs": list(b_xs), "select": False,
                  "uses": "Model B's drivers: " + ", ".join(b_xs)})
    return c


def _has_ylag(frame, y, xs) -> bool:
    lag = frame[y].shift(1)
    for x in xs:
        if x.lower() in (f"lag1({y})".lower(), f"lag({y})".lower()):
            return True
        both = lag.notna() & frame[x].notna()
        if both.sum() > 5 and np.allclose(frame.loc[both, x], lag[both]):
            return True
    return False


def evaluate(frame: pd.DataFrame, y: str, xs: list[str], is_ts: bool, alpha: float = 0.05, b_xs: list[str] | None = None,
             timing: str = "forecast", holdout: float = 0.2, refit: bool = True, folds: int = 5, seed: int = 11,
             unit: str = "", extra_notes: list[str] | None = None) -> dict:
    notes = list(extra_notes or [])
    b_xs = [x for x in (b_xs or []) if x in frame.columns]
    cols = list(dict.fromkeys(list(xs) + b_xs))
    has_ylag = _has_ylag(frame, y, xs) if is_ts else False
    cands = _candidates(xs, b_xs, is_ts, has_ylag)

    # ---- build the working frame (time series: shift drivers so only past information is used)
    W = pd.DataFrame(index=frame.index)
    W[y] = frame[y].astype(float)
    shifted = []
    for c in cols:
        if is_ts and timing == "forecast" and not already_lagged(c):
            W[c] = frame[c].astype(float).shift(1)
            shifted.append(c)
        else:
            W[c] = frame[c].astype(float)
    if is_ts:
        W[YLAG] = frame[y].astype(float).shift(1)
    W = W.dropna()
    n = len(W)
    yv = W[y].to_numpy()
    kmax = max(len(c["xs"]) for c in cands) + 1

    # ---- splits
    if is_ts:
        cut = int(np.floor(n * (1 - holdout)))
        if cut < kmax + 10 or n - cut < 5:
            return {"ok": False, "error": f"Not enough observations for a forecast test ({n} usable rows). ELK needs at "
                                          f"least {kmax + 10} to estimate the models and 5 to test them."}
        test_pos = np.arange(cut, n)
        ac = pd.Series(yv[:cut]).autocorr(1)
        level = bool(np.isfinite(ac) and ac > 0.9)
        bench = {"key": "RW" if level else "MEAN",
                 "label": "Benchmark: last period's value" if level else "Benchmark: historical average",
                 "uses": "No model: next value = latest value" if level else "No model: average of all values so far",
                 "why": (f"the outcome behaves like a slowly moving level (correlation with its own previous value is "
                         f"{ac:.2f} in the training period), so last period's value is the natural no-skill forecast"
                         if level else
                         f"the outcome moves around an average rather than drifting (correlation with its own previous "
                         f"value is {ac:.2f} in the training period), so the running average is the natural no-skill "
                         "forecast")}
        how = ("re-estimated before every forecast using only the rows up to that point (expanding window)" if refit
               else f"estimated once on the first {cut} rows")
        method = (f"Chronological split: first {cut} rows for training, last {n - cut} for testing (no shuffling). "
                  f"Models are {how}. One-step-ahead forecasts.")
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        parts = np.array_split(order, folds)
        if n - len(parts[0]) < kmax + 10:
            return {"ok": False, "error": f"Not enough observations for cross-validation ({n} usable rows)."}
        bench = {"key": "MEAN", "label": "Benchmark: training average", "uses": "No model: average of the training rows",
                 "why": "the data are not time-ordered, so the average outcome is the natural no-skill prediction"}
        method = (f"{folds}-fold cross-validation (random folds, fixed seed {seed}): each row is predicted by models "
                  "estimated without it. Time order does not apply to cross-sectional data.")

    # ---- predictions
    preds = {c["key"]: np.full(n, np.nan) for c in cands}
    preds[bench["key"]] = np.full(n, np.nan)
    Xall = {c["key"]: W[c["xs"]].to_numpy() for c in cands}

    def fit_predict(c, tr, te):
        X = Xall[c["key"]]
        Xtr, Xte = X[tr], X[te]
        if c["select"]:
            keep = _select(Xtr, yv[tr], alpha)
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        return _pred(_fit(Xtr, yv[tr]), Xte)

    if is_ts:
        if refit:
            for t in test_pos:
                tr = np.arange(t)
                for c in cands:
                    preds[c["key"]][t] = fit_predict(c, tr, np.array([t]))[0]
                preds[bench["key"]][t] = yv[:t].mean() if bench["key"] == "MEAN" else W[YLAG].iloc[t]
        else:
            tr = np.arange(cut)
            for c in cands:
                preds[c["key"]][test_pos] = fit_predict(c, tr, test_pos)
            preds[bench["key"]][test_pos] = yv[:cut].mean() if bench["key"] == "MEAN" else W[YLAG].iloc[test_pos]
    else:
        for p in parts:
            tr = np.setdiff1d(order, p)
            for c in cands:
                preds[c["key"]][p] = fit_predict(c, tr, p)
            preds[bench["key"]][p] = yv[tr].mean()
        test_pos = np.arange(n)

    # drop alternatives that turned out identical to an earlier model
    kept = []
    for c in cands:
        dup = next((k for k in kept if np.allclose(preds[c["key"]][test_pos], preds[k["key"]][test_pos])), None)
        if dup:
            notes.append(f"'{c['label']}' gave the same forecasts as '{dup['label']}' "
                         + ("(every useful driver was kept in each window)" if c["select"] else "") + ", so it is not shown.")
        else:
            kept.append(c)
    cands = kept

    # ---- metrics
    act = yv[test_pos]
    err = {k: act - v[test_pos] for k, v in preds.items() if k in [c["key"] for c in cands] + [bench["key"]]}
    eb = err[bench["key"]]
    mae_b, rmse_b = float(np.mean(np.abs(eb))), float(np.sqrt(np.mean(eb ** 2)))
    rows = [{"Model": bench["label"], "What it uses": bench["uses"], "MAE": mae_b, "RMSE": rmse_b,
             "MAE improvement (%)": 0.0, "RMSE improvement (%)": 0.0, "Out-of-sample R²": 0.0,
             "p-value vs benchmark": np.nan, "Beats benchmark?": "—", "key": bench["key"]}]
    for c in cands:
        e = err[c["key"]]
        mae, rmse = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))
        _, p = dm_test(e, eb)
        imp = 1 - rmse / rmse_b if rmse_b > 0 else np.nan
        verdict = ("Yes" if imp > 0 and np.isfinite(p) and p < alpha else
                   "Slightly — could be chance" if imp > 0 else "No")
        rows.append({"Model": c["label"], "What it uses": c["uses"], "MAE": mae, "RMSE": rmse,
                     "MAE improvement (%)": (1 - mae / mae_b) * 100 if mae_b > 0 else np.nan,
                     "RMSE improvement (%)": imp * 100, "Out-of-sample R²": 1 - (rmse / rmse_b) ** 2 if rmse_b > 0 else np.nan,
                     "p-value vs benchmark": p, "Beats benchmark?": verdict, "key": c["key"]})
    table = pd.DataFrame(rows)

    # ---- notes on design and data
    if is_ts and timing == "forecast" and shifted:
        notes.insert(0, "Each driver enters with its value from the previous period, so every forecast uses only "
                        "information available at the time.")
    if is_ts and timing == "known":
        notes.insert(0, "These results assume the drivers' values for each forecast period are already known (for example "
                        "a factor model that explains returns with same-month factors). That tests whether the "
                        "relationship holds on unseen data, but it is not a forecast you could make in advance. Switch "
                        "to 'Not known yet' for a real forecast.")
    if len(test_pos) < 20 and is_ts:
        notes.append(f"Only {len(test_pos)} test periods. Treat the comparison as indicative.")
    notes.append("Drivers and any adjustments chosen after looking at the full-sample results have seen the test "
                 "period, so this check is slightly optimistic for them.")

    verdict, best_label = _verdict(table, cands, err, bench, alpha, unit, is_ts, timing, len(test_pos))
    series = pd.DataFrame({"actual": act, **{r["Model"]: preds[r["key"]][test_pos] for r in rows}}, index=W.index[test_pos])
    return {"ok": True, "method": method, "benchmark": bench, "table": table.drop(columns="key"), "keys": table["key"].tolist(),
            "verdict": verdict, "best": best_label, "notes": notes, "timing": timing if is_ts else None, "n_test": int(len(test_pos)),
            "series": series, "is_ts": is_ts}


# ----------------------------------------------------------------------------- plain English
def _verdict(table, cands, err, bench, alpha, unit, is_ts, timing, n_test) -> Finding:
    models = table[table["key"] != bench["key"]]
    top = models.sort_values("RMSE").iloc[0]
    a_row = models[models["key"] == "A"].iloc[0]
    # prefer the original regression unless an alternative is clearly (statistically) better than it
    pa = np.nan
    if top["key"] != "A":
        _, pa = dm_test(err[top["key"]], err["A"])
    clear_alt = top["key"] != "A" and np.isfinite(pa) and pa < alpha and top["RMSE"] < a_row["RMSE"]
    best = top if clear_alt else a_row
    b_row = table.iloc[0]
    imp = best["RMSE improvement (%)"]
    p = best["p-value vs benchmark"]
    amt = lambda v: amount(v, unit)  # noqa: E731
    bname = b_row["Model"].replace("Benchmark: ", "").lower()
    period = "test periods" if is_ts else "rows"
    verb = "forecasts" if is_ts else "predicts"
    body = (f"On the {n_test} {period}, '{best['Model']}' missed by {amt(best['MAE'])} on average, against "
            f"{amt(b_row['MAE'])} for the benchmark ({bname}). "
            f"Its RMSE is {abs(imp):.1f}% {'lower' if imp > 0 else 'higher'} than the benchmark's"
            + (f" (Diebold–Mariano {p_eq(p)})." if np.isfinite(p) else "."))
    points = []
    if clear_alt:
        points.append(f"It is clearly more accurate than your regression ({p_eq(pa)}).")
    elif top["key"] != "A":
        points.append(f"'{top['Model']}' has a slightly lower RMSE than your regression, but the difference is too small "
                      f"to rule out chance ({p_eq(pa)}), so ELK sticks with your regression.")
    by_mae = models.sort_values("MAE").iloc[0]
    if by_mae["Model"] != top["Model"]:
        points.append(f"On MAE, '{by_mae['Model']}' does best instead. RMSE punishes large misses more, so "
                      f"'{top['Model']}' handles the big moves better while '{by_mae['Model']}' is closer in typical cases.")
    points.append(f"The benchmark is the {bname} because {bench['why']}.")

    reason = {
        "A": "Your drivers most likely carry information the benchmark doesn't"
             + (", and the alternative versions don't clearly improve on them." if len(cands) > 1 else "."),
        "AR": "The outcome is persistent: its last value carries information your drivers don't, so including it "
              "tracks the series more closely.",
        "SEL": "Dropping drivers that weren't clearly useful in the training data most likely cut noise from the "
               "estimates. The full model may be over-fitting a little.",
        "B": "Model B's set of drivers most likely captures the outcome better than the original set.",
    }[best["key"]]

    if imp > 0 and np.isfinite(p) and p < alpha:
        title = f"'{best['Model']}' {verb} better than a simple benchmark ({imp:.1f}% lower RMSE)"
        why, tone = reason, "good"
        nxt = ["Check the gain is stable: try a different test share." if is_ts else
               "Check the gain holds on new records as they arrive.",
               "Judge models for prediction by this test, not by in-sample R²."]
    elif imp > 0:
        title = f"'{best['Model']}' is only slightly more accurate than a simple benchmark, which could be chance"
        why = ("The improvement is small relative to how much the errors vary from one observation to the next, so luck "
               "can't be ruled out. If the gain is real, the likely reason is this: " + reason[0].lower() + reason[1:])
        tone = "caution"
        nxt = ["Treat forecasts from this model with caution.",
               "A longer test period, or more data, would give a clearer answer."]
    else:
        title = f"None of the models {verb} better than a simple benchmark"
        if bench["key"] == "RW":
            why = ("The outcome moves like a level, so last period's value is already a strong forecast. The drivers "
                   "don't add enough to beat it.")
        else:
            why = ("Many outcomes, returns especially, are hard to predict. Estimated coefficients add noise, so a model can fit "
                   "the past well and still forecast worse than the plain average.")
        why += " This does not undo the relationship found in-sample. It means the model shouldn't be relied on to forecast."
        tone = "problem"
        nxt = ["Use the model to describe exposures or relationships, not to forecast.",
               "If forecasting is the goal, try drivers that are known in advance (leading indicators)."]
    if is_ts and timing == "known":
        title = "With drivers known in advance: " + title[0].lower() + title[1:] if not title.startswith("'") else \
            "With drivers known in advance: " + title
        nxt.insert(0, "These results assume drivers are known in advance. Switch timing to 'Not known yet' for a real forecast.")
    return Finding(title, body, why, nxt, tone=tone, points=points), best["Model"]
