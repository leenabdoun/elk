"""Best-effort table detection for ordinary analyst workbooks and CSVs.

ELK reads every sheet as a raw grid, then finds the most plausible table: it skips title rows, notes, blank
rows/columns and merged headings, detects the header row, and recognises sideways tables (periods across columns).
Nothing is converted or rescaled here — cell values are kept exactly as stored. When more than one interpretation
is plausible the result says so (`needs_confirmation`) and the UI asks the analyst before any model is run.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_HEADER_SCAN = 40
_PERIOD_RE = re.compile(r"^(?:(?:FY|CY)\s*'?\d{2,4}|Q[1-4][\s\-/]*'?\d{2,4}|\d{4}[\s\-/]*Q[1-4]|H[12][\s\-/]*\d{4}|"
                        r"\d{4}|\d{4}[\-/]\d{1,2}([\-/]\d{1,2})?|\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|"
                        r"[A-Za-z]{3,9}[\s\-']*\d{2,4})$", re.I)


@dataclass
class ImportResult:
    df: pd.DataFrame
    sheet: str | None
    header_row: int | None          # 1-based row number in the source sheet (None = no header found)
    first_row: int                  # 1-based source row of the first data row
    last_row: int
    first_col: int                  # 1-based source column numbers
    last_col: int
    transposed: bool = False
    notes: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    question: str = ""
    sheets: list[dict] = field(default_factory=list)  # every sheet's best candidate, for the chooser

    @property
    def location(self) -> str:
        rng = f"{col_letter(self.first_col)}{self.first_row}:{col_letter(self.last_col)}{self.last_row}"
        hdr = f"header in row {self.header_row}" if self.header_row else "no header row found"
        where = f"sheet '{self.sheet}', " if self.sheet else ""
        return f"{where}{hdr}, data in {rng}" + (" (sideways table, transposed)" if self.transposed else "")


def col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ----------------------------------------------------------------------------- raw grids
def read_grids(file_bytes: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """Every sheet as a raw object grid (no header interpretation). CSV → one grid named ''."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        out = {}
        with pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl") as xf:
            for sh in xf.sheet_names:
                g = pd.read_excel(xf, sheet_name=sh, header=None, dtype=object)
                out[sh] = g
        return out
    if name.endswith((".csv", ".txt")):
        text = file_bytes.decode("utf-8-sig", errors="replace")
        sample = "\n".join(text.splitlines()[:50])
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ";" if sample.count(";") > sample.count(",") else ","
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        width = max((len(r) for r in rows), default=0)
        rows = [r + [""] * (width - len(r)) for r in rows]
        g = pd.DataFrame(rows, dtype=object).replace({"": None})
        return {"": g}
    raise ValueError("Unsupported file type. Please upload an Excel (.xlsx) or CSV file.")


# ----------------------------------------------------------------------------- cell classification
def _kind(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "empty"
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year") and hasattr(v, "month") and not isinstance(v, str):
        return "date"
    if isinstance(v, (bool, np.bool_)):
        return "num"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return "num"
    s = str(v).strip()
    if s == "":
        return "empty"
    t = re.sub(r"[,\s$€£%]", "", s)
    t = re.sub(r"^\((.*)\)$", r"-\1", t)
    try:
        float(t)
        return "num"
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", s):
        return "date"
    return "text"


def _is_period_label(v) -> bool:
    k = _kind(v)
    if k == "date":
        return True
    s = str(v).strip()
    if k == "num":
        try:
            f = float(s)
            return f.is_integer() and 1900 <= f <= 2100
        except ValueError:
            return False
    return bool(_PERIOD_RE.match(s))


# ----------------------------------------------------------------------------- detection on one grid
def _trim(grid: pd.DataFrame) -> pd.DataFrame:
    kinds = grid.map(_kind)
    keep_r = (kinds != "empty").any(axis=1)
    keep_c = (kinds != "empty").any(axis=0)
    return grid.loc[keep_r, keep_c]


def _score_header(kinds: pd.DataFrame, r: int, raw: pd.DataFrame | None = None) -> float:
    row = kinds.iloc[r]
    filled = row[row != "empty"]
    width = kinds.shape[1]
    if len(filled) < max(2, 0.5 * width):
        return -1
    text_share = (filled == "text").mean()
    if text_share < 0.5 and raw is not None:
        # sideways tables: a row of periods (2019, 2020, … or dates) over text row labels
        vals = [v for v, k in zip(raw.iloc[r].tolist()[1:], row.tolist()[1:]) if k != "empty"]
        labels = kinds.iloc[r + 1: r + 16, 0]
        if len(vals) >= 3 and np.mean([_is_period_label(v) for v in vals]) >= 0.8 and (labels == "text").mean() >= 0.8:
            text_share = 1.0
    if text_share < 0.5:
        return -1
    below = kinds.iloc[r + 1: r + 16]
    if below.empty:
        return -1
    cols = filled.index
    body = below[cols]
    body_filled = (body != "empty").mean().mean()
    body_valueish = body.isin(["num", "date"]).mean().mean()
    return text_share * 2 + body_valueish * 2 + body_filled + 0.5 * (len(filled) / width)


def detect_table(grid: pd.DataFrame, sheet: str | None, header_row: int | None = None,
                 first_row: int | None = None, last_row: int | None = None, transpose: bool | None = None
                 ) -> ImportResult | None:
    """Find the main table in a raw grid. Row/column numbers in overrides and results are 1-based source positions."""
    g = _trim(grid)
    if g.empty or g.shape[0] < 2 or g.shape[1] < 1:
        return None
    rows = list(g.index)       # 0-based source row positions
    cols = list(g.columns)     # 0-based source col positions
    kinds = g.map(_kind).reset_index(drop=True)
    kinds.columns = range(kinds.shape[1])
    notes = []

    # ---- header row
    if header_row is not None:
        if header_row - 1 not in rows:
            raise ValueError(f"Row {header_row} is empty or outside the sheet.")
        h = rows.index(header_row - 1)
    else:
        rawg = g.reset_index(drop=True)
        scores = [(_score_header(kinds, r, rawg), r) for r in range(min(MAX_HEADER_SCAN, len(rows) - 1))]
        best, h = max(scores) if scores else (-1, 0)
        if best < 2.0:  # no convincing text header → numbers only
            h = None
    # ---- data block
    start = (h + 1) if h is not None else 0
    if first_row is not None:
        start = max(start, rows.index(first_row - 1) if first_row - 1 in rows else start)
    end = len(rows) - 1
    if last_row is not None:
        cand = [i for i, r in enumerate(rows) if r <= last_row - 1]
        end = cand[-1] if cand else end
    else:
        # stop at the first gap of 2+ blank source rows, then trim trailing notes
        for i in range(start + 1, end + 1):
            if rows[i] - rows[i - 1] > 2:
                end = i - 1
                notes.append(f"Stopped at a blank gap after source row {rows[i - 1] + 1}; content below was ignored.")
                break
        while end > start:
            rk = kinds.iloc[end]
            filled = rk[rk != "empty"]
            if len(filled) <= 1 and (filled == "text").all() and kinds.shape[1] > 1:
                end -= 1
                continue
            break
        dropped_notes = len(rows) - 1 - end
        if dropped_notes > 0 and not any(n.startswith("Stopped") for n in notes):
            notes.append(f"{dropped_notes} note/footer row(s) below the table were ignored.")
    if h is not None and h > 0:
        notes.append(f"{h} title/heading row(s) above the header were skipped.")

    body = g.iloc[start: end + 1]
    if h is not None:
        names = [_name(v, j) for j, v in enumerate(g.iloc[h].tolist())]
        # a merged/group heading directly above often leaves the first header cell blank – fine, named generically
    else:
        names = [f"Column {j + 1}" for j in range(g.shape[1])]
        notes.append("No header row was recognised; columns were given generic names.")
    names = _dedupe(names)
    df = pd.DataFrame(body.to_numpy(), columns=names)
    # drop columns that are empty inside the table body
    empty_cols = [c for c in df.columns if df[c].map(_kind).eq("empty").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    df = df.reset_index(drop=True)

    # ---- orientation
    is_sideways = _looks_sideways(g, h) if transpose is None else transpose
    if is_sideways:
        df = _transpose(g, h, start, end)
        notes.append("Periods run across the columns, so the table was turned sideways: each period is now a row.")

    df = _coerce_types(df)
    res = ImportResult(df=df, sheet=sheet, header_row=(rows[h] + 1) if h is not None else None,
                       first_row=rows[start] + 1, last_row=rows[end] + 1, first_col=cols[0] + 1, last_col=cols[-1] + 1,
                       transposed=bool(is_sideways), notes=notes)
    if h is None:
        res.needs_confirmation = True
        res.question = "ELK could not find a row of column names. Check the preview, or set the header row under Advanced import."
    return res


def _name(v, j) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return f"Column {j + 1}"
    if hasattr(v, "strftime") and not isinstance(v, str):
        return v.strftime("%Y-%m-%d")
    s = re.sub(r"\s+", " ", str(v)).strip()
    if isinstance(v, float) and v.is_integer():
        s = str(int(v))
    return s


def _dedupe(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 1
            out.append(n)
    return out


def _looks_sideways(g: pd.DataFrame, h: int | None) -> bool:
    if h is None or g.shape[1] < 4:
        return False
    hdr = g.iloc[h].tolist()[1:]
    hdr = [v for v in hdr if _kind(v) != "empty"]
    if len(hdr) < 3:
        return False
    periods = np.mean([_is_period_label(v) for v in hdr])
    first_col = [v for v in g.iloc[h + 1:, 0].tolist() if _kind(v) != "empty"]
    labels_text = np.mean([_kind(v) == "text" for v in first_col]) if first_col else 0
    # a normal table can have a date header *column*, never date-like column *names* across the top
    return periods >= 0.8 and labels_text >= 0.8


def _transpose(g, h, start, end) -> pd.DataFrame:
    block = g.iloc[[h] + list(range(start, end + 1))]
    labels = [_name(v, i) for i, v in enumerate(block.iloc[1:, 0].tolist())]
    periods = block.iloc[0, 1:].tolist()
    values = block.iloc[1:, 1:].T.to_numpy()
    df = pd.DataFrame(values, columns=_dedupe(labels))
    df.insert(0, "Period", [_period_value(p) for p in periods])
    df = df[[c for c in df.columns if not df[c].map(_kind).eq("empty").all()]]
    return df.dropna(how="all", subset=[c for c in df.columns if c != "Period"]).reset_index(drop=True)


def _period_value(v):
    if _kind(v) == "date":
        return pd.to_datetime(v, errors="coerce", format="mixed")
    s = str(v).strip()
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return pd.Timestamp(int(v), 12, 31)
    m = re.match(r"^(?:FY|CY)\s*'?(\d{2,4})$", s, re.I)
    if m:
        y = int(m.group(1))
        return pd.Timestamp(y + 2000 if y < 100 else y, 12, 31)
    m = re.match(r"^Q([1-4])[\s\-/]*'?(\d{2,4})$|^(\d{4})[\s\-/]*Q([1-4])$", s, re.I)
    if m:
        q, y = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(4)), int(m.group(3)))
        y = y + 2000 if y < 100 else y
        return pd.Period(f"{y}Q{q}", "Q").to_timestamp(how="end").normalize()
    d = pd.to_datetime(s, errors="coerce", format="mixed")
    return d if pd.notna(d) else s


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Store pure-number columns as numbers and date columns as dates. Text with %, commas or brackets is left
    as text exactly as typed (ELK converts it later, visibly, and never rescales)."""
    out = df.copy()
    for c in out.columns:
        k = out[c].map(_kind)
        filled = k[k != "empty"]
        if filled.empty:
            continue
        raw = out[c]
        if (filled == "num").all() and all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_))
                                           for v in raw[k != "empty"]):
            out[c] = pd.to_numeric(raw, errors="coerce")
        elif (filled == "num").all():
            plain = raw[k != "empty"].astype(str).str.strip()
            if plain.str.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?").all():
                out[c] = pd.to_numeric(raw, errors="coerce")
            else:
                out[c] = raw.astype(object).where(k != "empty", None)
        elif (filled == "date").all():
            out[c] = pd.to_datetime(raw, errors="coerce", format="mixed")
        else:
            out[c] = raw.astype(object).where(k != "empty", None)
    return out


def _numberish(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    k = s.map(_kind)
    k = k[k != "empty"]
    return len(k) > 0 and (k == "num").mean() >= 0.8


# ----------------------------------------------------------------------------- whole file
def import_file(file_bytes: bytes, filename: str, sheet: str | None = None, **overrides) -> ImportResult:
    grids = read_grids(file_bytes, filename)
    cands = []
    for sh, g in grids.items():
        try:
            r = detect_table(g, sh or None)
        except Exception:
            r = None
        if r is None:
            cands.append({"sheet": sh, "rows": 0, "cols": 0, "numeric_cols": 0, "usable": False})
            continue
        num = sum(_numberish(r.df[c]) for c in r.df.columns)
        cands.append({"sheet": sh, "rows": len(r.df), "cols": r.df.shape[1], "numeric_cols": int(num),
                      "usable": len(r.df) >= 3 and num >= 2, "location": r.location})
    usable = [c for c in cands if c["usable"]]
    auto = sheet is None
    if sheet is None:
        if not usable:
            raise ValueError("ELK could not find a table with at least two numeric columns and three rows in this file. "
                             "Check that the data has one observation per row (or per column), or use Advanced import "
                             "to point ELK at the right rows.")
        usable.sort(key=lambda c: (c["rows"] * c["numeric_cols"]), reverse=True)
        sheet = usable[0]["sheet"]
    if sheet not in grids:
        raise ValueError(f"Sheet '{sheet}' not found.")
    res = detect_table(grids[sheet], sheet or None, **overrides)
    if res is None:
        raise ValueError(f"No table found on sheet '{sheet}'.")
    res.sheets = cands
    if len(usable) > 1 and auto and not overrides:
        res.needs_confirmation = True
        others = ", ".join(f"'{c['sheet']}'" for c in usable if c["sheet"] != sheet)
        res.question = (f"This workbook has data tables on {len(usable)} sheets. ELK picked '{sheet}' (the largest). "
                        f"Other candidates: {others}. Confirm or choose another sheet.")
    return res
