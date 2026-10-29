"""Data Suitability Assessment: descriptive warnings, blocking issues only where maths fails."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_io import infer_frequency, parse_dates, try_numeric


@dataclass
class Issue:
    level: str  # "block" | "warn" | "info"
    title: str
    detail: str


@dataclass
class Suitability:
    issues: list[Issue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    frame: pd.DataFrame | None = None  # complete-case modelling frame (index = row label)
    labels: pd.Series | None = None  # date or id per row, for display
    dates: pd.Series | None = None

    @property
    def blocked(self) -> bool:
        return any(i.level == "block" for i in self.issues)

    def add(self, level, title, detail):
        self.issues.append(Issue(level, title, detail))


def assess(df: pd.DataFrame, y: str, xs: list[str], date_col: str | None = None,
           id_col: str | None = None, freq: str | None = None) -> Suitability:
    out = Suitability()
    if not y or not xs:
        out.add("block", "Variables not selected", "Choose one outcome (Y) and at least one explanatory variable (X).")
        return out
    if y in xs:
        out.add("block", "Outcome also used as explanatory variable", f"'{y}' cannot be both Y and an X.")
        return out

    cols = [y] + list(xs)
    work = pd.DataFrame(index=df.index)
    n_raw = len(df)

    # numeric conversion, invalid values
    for c in cols:
        conv, info = try_numeric(df[c])
        if info["unparseable"]:
            out.add("warn", f"Non-numeric values in '{c}'",
                    f"{info['unparseable']} cell(s) could not be read as numbers and are treated as missing.")
        if info["had_percent_sign"]:
            out.add("info", f"'{c}' contains % signs",
                    "Values were read as the number shown (e.g. '5%' → 5). Confirm the unit label for this column.")
        inf = int(np.isinf(conv).sum())
        if inf:
            out.add("warn", f"Infinite values in '{c}'", f"{inf} infinite value(s) are treated as missing.")
            conv = conv.replace([np.inf, -np.inf], np.nan)
        work[c] = conv

    # dates
    dates = None
    if date_col:
        dates = parse_dates(df[date_col])
        bad = int(dates.isna().sum() - df[date_col].isna().sum())
        if bad > 0:
            out.add("warn", "Unreadable dates", f"{bad} date value(s) could not be parsed; those rows are excluded.")
        work["__date"] = dates
    if id_col:
        work["__id"] = df[id_col].astype(str)

    # missingness
    miss = work[cols].isna().sum()
    out.metrics["missing_by_column"] = miss.to_dict()
    keep_cols = cols + (["__date"] if date_col else [])
    complete = work.dropna(subset=keep_cols)
    n = len(complete)
    out.metrics["n_raw"] = n_raw
    out.metrics["n_complete"] = n
    out.metrics["n_dropped_missing"] = n_raw - n
    if n_raw - n > 0:
        worst = miss.sort_values(ascending=False)
        worst = ", ".join(f"{k} ({v})" for k, v in worst[worst > 0].head(4).items())
        out.add("warn" if (n_raw - n) / max(1, n_raw) > 0.1 else "info",
                f"{n_raw - n} row(s) have missing values in the modelled columns",
                f"Complete-case analysis will use {n} of {n_raw} rows. Most missing: {worst}. "
                "No values are filled in or interpolated.")

    # sort chronologically, duplicates
    is_ts = bool(date_col)
    if is_ts:
        complete = complete.sort_values("__date", kind="stable")
        dup_ts = int(complete["__date"].duplicated().sum())
        if dup_ts:
            out.add("warn", "Duplicate timestamps",
                    f"{dup_ts} row(s) share a date with another row. If this is several entities per date, the data is "
                    "a panel, and time-series tests (Durbin–Watson, Breusch–Godfrey, ADF, ARCH) are not valid as ordered.")
        out.metrics["duplicate_dates"] = dup_ts
        f, gap = infer_frequency(complete["__date"])
        out.metrics["detected_frequency"] = f
        out.metrics["median_gap_days"] = gap
        out.metrics["start"] = complete["__date"].min()
        out.metrics["end"] = complete["__date"].max()
        if freq and f not in (freq, "irregular"):
            out.add("warn", "Frequency mismatch", f"You selected {freq} data but the dates look {f}.")
        if f == "irregular":
            out.add("warn", "Irregular date spacing",
                    "Dates are not evenly spaced. Lag-based tests assume equally spaced observations.")
    dup_rows = int(complete.duplicated(subset=cols).sum())
    out.metrics["duplicate_rows"] = dup_rows
    if dup_rows:
        out.add("warn", "Duplicate observations",
                f"{dup_rows} row(s) have identical values across all modelled columns. Check for double entry.")

    # variability
    var_rows = []
    for c in cols:
        s = complete[c]
        sd = float(s.std()) if n > 1 else 0.0
        nun = int(s.nunique())
        var_rows.append({"variable": c, "mean": s.mean(), "std": sd, "min": s.min(), "max": s.max(), "unique": nun})
        if nun <= 1:
            out.add("block", f"'{c}' is constant", "A column with a single value cannot be estimated (it duplicates the intercept).")
        elif nun == 2 and c == y:
            out.add("block", "Binary outcome — OLS is not supported",
                    f"'{y}' takes only two values. Yes/no outcomes such as defaults need logistic or survival models "
                    "(planned future module). A linear probability model can produce probabilities below 0 or above 1 "
                    "and its standard errors are invalid without correction.")
        elif nun < 5 and c == y:
            out.add("warn", "Outcome has very few distinct values",
                    f"'{y}' has only {nun} distinct values; it may be ordinal or categorical rather than continuous.")
    out.metrics["variability"] = pd.DataFrame(var_rows)

    # parameters vs observations
    k = len(xs) + 1
    out.metrics["k_params"] = k
    out.metrics["df_resid"] = n - k
    ratio = n / k if k else 0
    out.metrics["obs_per_param"] = ratio
    if n - k < 1:
        out.add("block", "Too few observations",
                f"{n} usable rows for {k} parameters leaves no residual degrees of freedom. Add data or remove variables.")
    elif n - k < 10:
        out.add("warn", "Very few residual degrees of freedom",
                f"Only {n - k} residual degrees of freedom. Standard errors and tests will be highly unstable.")
    elif ratio < 10:
        out.add("warn", "Low observations per parameter",
                f"{ratio:.1f} observations per estimated parameter. Estimates may be imprecise and sensitive to individual points.")

    # perfect collinearity
    if n > k and not any(i.title.endswith("is constant") for i in out.issues):
        X = np.column_stack([np.ones(n), complete[xs].to_numpy(dtype=float)])
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            culprits = _find_collinear(complete[xs])
            out.add("block", "Perfect collinearity",
                    "At least one explanatory variable is an exact linear combination of others"
                    + (f" ({culprits})" if culprits else "") +
                    ". The regression cannot separate their effects. Remove one of them "
                    "(for dummy variables, leave one category out as the reference).")
        else:
            Z = X[:, 1:]
            sd = Z.std(0)
            Z = (Z - Z.mean(0)) / np.where(sd == 0, 1, sd)
            out.metrics["condition_number"] = float(np.linalg.cond(Z)) if len(xs) > 1 else 1.0

    # classification & test appropriateness
    out.metrics["structure"] = "time series" if is_ts else "cross-sectional"
    if is_ts:
        f = out.metrics.get("detected_frequency")
        out.add("info", "Time-series data",
                f"Rows are sorted by date ({f}). Serial-correlation tests, ADF, ARCH and chronological holdout are available.")
        if f in ("quarterly", "annual") and n < 40:
            out.add("warn", "Short low-frequency series",
                    f"{n} {f} observations. Inference is limited; appraisal-based or smoothed series (e.g. private-market NAVs) "
                    "typically understate volatility and overstate autocorrelation.")
        if n > 3:
            r1 = complete[y].autocorr(1)
            out.metrics["y_lag1_autocorr"] = r1
            if r1 is not None and np.isfinite(r1) and r1 > 0.5:
                out.add("warn", "Outcome is highly autocorrelated",
                        f"Lag-1 autocorrelation of '{y}' is {r1:.2f}. This can indicate price levels rather than returns, "
                        "or return smoothing (common in appraisal-based NAVs and illiquid funds).")
    else:
        out.add("info", "Cross-sectional data",
                "No date column selected, so row order is treated as arbitrary. Durbin–Watson, Breusch–Godfrey, ADF and ARCH "
                "are not meaningful and are hidden; heteroskedasticity, collinearity and influence checks apply.")

    out.frame = complete[cols].astype(float)
    out.dates = complete["__date"] if is_ts else None
    if is_ts:
        out.labels = complete["__date"].dt.strftime("%Y-%m-%d")
    elif id_col:
        out.labels = complete["__id"]
    else:
        out.labels = pd.Series((complete.index + 1).astype(str), index=complete.index).radd("row ")
    return out


def _find_collinear(X: pd.DataFrame) -> str:
    cols = list(X.columns)
    corr = X.corr().abs()
    pairs = [f"{a} ~ {b}" for i, a in enumerate(cols) for b in cols[i + 1:] if corr.loc[a, b] > 0.99999]
    if pairs:
        return "; ".join(pairs[:3])
    # dummy trap: a set summing to constant
    binary = [c for c in cols if set(np.unique(X[c].dropna())) <= {0.0, 1.0}]
    if len(binary) >= 2 and (X[binary].sum(axis=1).nunique() == 1):
        return "dummy variables " + ", ".join(binary) + " sum to a constant"
    return ""
