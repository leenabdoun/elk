"""Reading analyst files, profiling columns, detecting dates and frequency."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FREQ_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}
FREQ_PANDAS = {"daily": "D", "weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "annual": "Y"}

_NUM_CLEAN = re.compile(r"[,\s$€£]")


def list_sheets(file_bytes: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl") as xf:
        return list(xf.sheet_names)


def read_table(file_bytes: bytes, filename: str, sheet: str | None = None) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet or 0, engine="openpyxl")
    elif name.endswith((".csv", ".txt")):
        raw = file_bytes.decode("utf-8-sig", errors="replace")
        sample = raw[:5000]
        sep = ";" if sample.count(";") > sample.count(",") else ","
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    else:
        raise ValueError("Unsupported file type. Please upload an Excel (.xlsx) or CSV file.")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if unnamed and len(unnamed) == len(df.columns):
        raise ValueError("No column headers were found. The first row of the sheet should contain column names.")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {', '.join(dupes)}. Please give each column a unique name.")
    return df.reset_index(drop=True)


def try_numeric(s: pd.Series) -> tuple[pd.Series, dict]:
    """Convert a column to numbers where it is safe; report what happened. Never guesses units."""
    info = {"had_percent_sign": False, "unparseable": 0, "converted_from_text": False}
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float), info
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float), info
    txt = s.astype("string").str.strip()
    info["had_percent_sign"] = bool(txt.str.contains("%", na=False).any())
    cleaned = txt.str.replace("%", "", regex=False).str.replace(_NUM_CLEAN, "", regex=True)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (1.2) accounting negatives
    out = pd.to_numeric(cleaned, errors="coerce").astype(float)
    nonblank = txt.notna() & (txt != "")
    info["unparseable"] = int((out.isna() & nonblank).sum())
    info["converted_from_text"] = True
    return out, info


def looks_like_date(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        if re.search(r"date|month|period|time", name, re.I):
            v = s.dropna()
            if len(v) and v.between(190001, 21001231).all() and (v % 1 == 0).all():
                return True
        return False
    v = s.dropna().astype(str).head(50)
    if v.empty:
        return False
    parsed = pd.to_datetime(v, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9 and v.str.contains(r"[-/.]|\d{4}", regex=True).mean() > 0.9


def parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype("Int64").astype(str)
        if v.str.len().eq(6).all():
            return pd.to_datetime(v, format="%Y%m") + pd.offsets.MonthEnd(0)
        return pd.to_datetime(v, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce", format="mixed")


def infer_frequency(dates: pd.Series) -> tuple[str, float | None]:
    d = pd.Series(pd.to_datetime(dates)).dropna().sort_values().drop_duplicates()
    if len(d) < 3:
        return "irregular", None
    gap = float(d.diff().dt.days.median())
    if gap <= 4:
        f = "daily"
    elif gap <= 9:
        f = "weekly"
    elif 25 <= gap <= 35:
        f = "monthly"
    elif 80 <= gap <= 100:
        f = "quarterly"
    elif 330 <= gap <= 400:
        f = "annual"
    else:
        f = "irregular"
    return f, gap


@dataclass
class ColumnProfile:
    name: str
    kind: str  # numeric | text-numeric | categorical | date | id-like | empty
    dtype: str
    missing: int
    unique: int
    example: str
    note: str = ""


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        uniq = int(s.nunique(dropna=True))
        ex = "" if s.dropna().empty else str(s.dropna().iloc[0])[:30]
        note = ""
        if s.dropna().empty:
            kind = "empty"
        elif looks_like_date(s, c):
            kind = "date"
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            kind = "numeric"
            if uniq == 2:
                note = "binary (0/1)"
            elif uniq == n and re.search(r"(^|_)id$|^id|code", c, re.I):
                kind = "id-like"
        else:
            conv, info = try_numeric(s)
            ok = conv.notna().sum() / max(1, s.notna().sum())
            if ok > 0.95:
                kind = "text-numeric"
                note = "stored as text; contains % signs" if info["had_percent_sign"] else "stored as text"
            elif uniq == n:
                kind = "id-like"
            elif uniq <= max(20, 0.05 * n):
                kind = "categorical"
                note = f"{uniq} categories"
            else:
                kind = "id-like" if uniq > 0.5 * n else "categorical"
        rows.append(ColumnProfile(c, kind, str(s.dtype), miss, uniq, ex, note).__dict__)
    out = pd.DataFrame(rows)
    out["missing_pct"] = (out["missing"] / max(1, n) * 100).round(1)
    return out


def unit_hint(s: pd.Series) -> str:
    """Describe the scale of a numeric series without deciding the unit for the analyst."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return ""
    med = float(v.abs().median())
    if med < 0.2 and v.abs().max() < 3:
        return f"typical magnitude {med:.4f} — looks like decimals (0.01 = 1%)"
    if med < 20 and v.abs().max() < 300:
        return f"typical magnitude {med:.2f} — could be percent (1.0 = 1%)"
    return f"typical magnitude {med:,.1f}"


def make_dummies(df: pd.DataFrame, col: str, reference: str) -> tuple[pd.DataFrame, list[str]]:
    cats = [c for c in pd.Series(df[col].dropna().unique()).astype(str).sort_values() if c != str(reference)]
    new_cols = []
    out = df.copy()
    base = out[col].astype("string")
    for c in cats:
        safe = re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_")
        name = f"{col}_{safe}"
        out[name] = np.where(base.isna(), np.nan, (base == c).astype(float))
        new_cols.append(name)
    return out, new_cols
