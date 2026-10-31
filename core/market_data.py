"""Public market data: Kenneth French factors + risk-free rate, Yahoo ETF prices, FRED T-bills.

Every function tries the live source first, then falls back to the snapshot saved in data/snapshot
(refreshed by a scheduled GitHub Action). Provenance is returned with every result.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import FREQ_PERIODS_PER_YEAR

SNAPSHOT_DIR = Path(os.environ.get("ELK_SNAPSHOT_DIR") or Path(__file__).resolve().parent.parent / "data" / "snapshot")
KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
KF_PAGE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
KF_FILES = {
    ("ff3", "monthly"): "F-F_Research_Data_Factors_CSV.zip",
    ("ff3", "weekly"): "F-F_Research_Data_Factors_weekly_CSV.zip",
    ("ff3", "daily"): "F-F_Research_Data_Factors_daily_CSV.zip",
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    ("ff5", "daily"): "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ("mom", "monthly"): "F-F_Momentum_Factor_CSV.zip",
    ("mom", "daily"): "F-F_Momentum_Factor_daily_CSV.zip",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FACTOR_MODELS = {
    "CAPM (market only)": ["Mkt-RF"],
    "Fama–French 3": ["Mkt-RF", "SMB", "HML"],
    "Fama–French 3 + Momentum": ["Mkt-RF", "SMB", "HML", "Mom"],
    "Fama–French 5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "Fama–French 5 + Momentum": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}
FACTOR_DESCRIPTIONS = {
    "Mkt-RF": "Market excess return: value-weighted US equity market minus the 1-month T-bill",
    "SMB": "Small minus big: small-cap minus large-cap stocks",
    "HML": "High minus low: value (high book-to-market) minus growth stocks",
    "RMW": "Robust minus weak: high-profitability minus low-profitability stocks",
    "CMA": "Conservative minus aggressive: low-investment minus high-investment firms",
    "Mom": "Momentum: prior 12-month winners minus losers (skipping the latest month)",
    "RF": "Risk-free rate: 1-month US Treasury bill return for the period",
}
FACTOR_SHORT = {"Mkt-RF": "market excess return", "SMB": "small-cap factor", "HML": "value factor",
                "RMW": "profitability factor", "CMA": "conservative-investment factor", "Mom": "momentum factor",
                "RF": "risk-free rate"}
DEFAULT_TICKERS = ["SPY", "IWM", "IWD", "IWF", "QUAL", "XLK"]
TICKER_NOTES = {
    "SPY": "S&P 500 — broad US large-cap market",
    "IWM": "Russell 2000 — US small caps (expect positive SMB)",
    "IWD": "Russell 1000 Value (expect positive HML)",
    "IWF": "Russell 1000 Growth (expect negative HML)",
    "QUAL": "MSCI USA Quality (expect positive RMW)",
    "XLK": "Technology Select Sector — concentrated sector exposure",
}


# ----------------------------------------------------------------------------- Ken French
def parse_kf_csv(text: str) -> pd.DataFrame:
    """Parse the first data block of a Ken French CSV (values in percent → decimals)."""
    lines = text.splitlines()
    header, rows = None, []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if header is None:
            if len(parts) > 1 and parts[0] == "" and all(re.search(r"[A-Za-z]", p) for p in parts[1:] if p):
                header = [p for p in parts[1:] if p]
            continue
        if parts and re.fullmatch(r"\d{6}|\d{8}", parts[0] or ""):
            rows.append(parts[: len(header) + 1])
        elif rows:
            break  # end of first (period) block; annual block follows
    if header is None or not rows:
        raise ValueError("Unrecognised Ken French file format")
    df = pd.DataFrame(rows, columns=["code"] + header)
    code = df.pop("code")
    if code.str.len().iloc[0] == 6:
        idx = pd.to_datetime(code, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        idx = pd.to_datetime(code, format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.index = pd.DatetimeIndex(idx, name="Date")
    df.columns = [c.strip().replace("Mom   ", "Mom") for c in df.columns]
    df = df.rename(columns={c: "Mom" for c in df.columns if c.lower().startswith("mom")})
    df = df.replace([-99.99 / 100, -999 / 100], np.nan)
    return df


def fetch_kf(name: str, freq: str, timeout=20) -> pd.DataFrame:
    import requests

    fn = KF_FILES[(name, freq)]
    r = requests.get(KF_BASE + fn, timeout=timeout, headers={"User-Agent": "ELK research app"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        inner = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]
        text = z.read(inner).decode("latin-1")
    return parse_kf_csv(text)


def _snap_path(key: str) -> Path:
    return SNAPSHOT_DIR / f"{key}.csv"


def _snapshot_meta() -> dict:
    p = SNAPSHOT_DIR / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _read_snap(key: str) -> pd.DataFrame | None:
    p = _snap_path(key)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def get_kf(name: str, freq: str, allow_live=True) -> tuple[pd.DataFrame, dict]:
    key = f"kf_{name}_{freq}"
    err = None
    if allow_live:
        try:
            df = fetch_kf(name, freq)
            return df, {"source": "Kenneth R. French Data Library (live)", "url": KF_PAGE,
                        "file": KF_FILES[(name, freq)], "retrieved": _now(), "mode": "live"}
        except Exception as e:  # network, format
            err = f"{type(e).__name__}: {e}"
    df = _read_snap(key)
    if df is None:
        raise RuntimeError(f"Could not reach the Ken French library ({err}) and no saved snapshot is available.")
    meta = _snapshot_meta().get(key, {})
    return df, {"source": "Kenneth R. French Data Library (saved snapshot)", "url": KF_PAGE,
                "file": KF_FILES[(name, freq)], "retrieved": meta.get("retrieved", "unknown"),
                "mode": "snapshot", "live_error": err}


def get_factors(model: str, freq: str, allow_live=True) -> tuple[pd.DataFrame, list[dict]]:
    """Factor returns + RF at the requested frequency (daily/weekly/monthly/quarterly/annual)."""
    want = FACTOR_MODELS[model]
    base_freq = freq if freq in ("daily", "monthly", "weekly") else "monthly"
    five = any(f in want for f in ("RMW", "CMA"))
    mom = "Mom" in want
    if base_freq == "weekly" and (five or mom):
        # weekly files exist only for FF3; build weekly by compounding daily
        df, prov = get_factors(model, "daily", allow_live)
        return _compound(df, "weekly"), [dict(p, note="weekly built by compounding daily factors") for p in prov]
    provs = []
    df, p = get_kf("ff5" if five else "ff3", base_freq, allow_live)
    provs.append(p)
    if mom:
        m, p2 = get_kf("mom", base_freq, allow_live)
        provs.append(p2)
        df = df.join(m[["Mom"]], how="inner")
    df = df[want + ["RF"]].dropna()
    if freq in ("quarterly", "annual"):
        df = _compound(df, freq)
        provs = [dict(p, note=f"{freq} built by compounding monthly returns") for p in provs]
    return df, provs


def _compound(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Compound period returns to a lower frequency. Mkt-RF is rebuilt from compounded Mkt and RF;
    long–short factors are compounded directly (a standard approximation)."""
    rule = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "annual": "YE"}[freq]
    d = df.copy()
    if "Mkt-RF" in d:
        d["__mkt"] = d["Mkt-RF"] + d["RF"]
    g = (1 + d).resample(rule).prod(min_count=1) - 1
    counts = d["RF"].resample(rule).count()
    g = g[counts > 0]
    if "Mkt-RF" in g:
        g["Mkt-RF"] = g.pop("__mkt") - g["RF"]
    # drop an incomplete final period
    last = df.index.max()
    if freq != "weekly" and last < g.index.max() - pd.Timedelta(days=5):
        g = g.iloc[:-1]
    g.index.name = "Date"
    return g[[c for c in df.columns]]


# ----------------------------------------------------------------------------- Yahoo
def fetch_prices(tickers: list[str], start: str = "1993-01-01") -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    if data is None or data.empty:
        raise RuntimeError("Yahoo returned no data")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    return close.dropna(how="all")


def get_prices(tickers: list[str], allow_live=True) -> tuple[pd.DataFrame, dict, list[str]]:
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    err = None
    if allow_live:
        try:
            px = fetch_prices(tickers)
            missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
            if len(missing) < len(tickers):
                return px[[t for t in tickers if t not in missing]], {
                    "source": "Yahoo Finance via yfinance (live; adjusted close incl. dividends)",
                    "url": "https://finance.yahoo.com", "retrieved": _now(), "mode": "live"}, missing
            err = "no data for any ticker"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    snap = _read_snap("etf_prices_daily")
    if snap is None:
        raise RuntimeError(f"Could not reach Yahoo Finance ({err}) and no saved snapshot is available.")
    have = [t for t in tickers if t in snap.columns]
    missing = [t for t in tickers if t not in snap.columns]
    if not have:
        raise RuntimeError(f"Yahoo Finance unavailable ({err}); the saved snapshot only contains: {', '.join(snap.columns)}.")
    meta = _snapshot_meta().get("etf_prices_daily", {})
    return snap[have], {"source": "Yahoo Finance via yfinance (saved snapshot; adjusted close incl. dividends)",
                        "url": "https://finance.yahoo.com", "retrieved": meta.get("retrieved", "unknown"),
                        "mode": "snapshot", "live_error": err}, missing


def prices_to_returns(px: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "daily":
        p = px
    else:
        rule = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "annual": "YE"}[freq]
        p = px.resample(rule).last()
        # drop incomplete current period
        if px.index.max() < p.index.max() - pd.Timedelta(days=3 if freq == "weekly" else 5):
            p = p.iloc[:-1]
    r = p.pct_change(fill_method=None)
    r.index.name = "Date"
    return r.iloc[1:]


def align_by_period(a: pd.DataFrame, b: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Join two date-indexed frames on calendar period (handles month-end vs last-trading-day)."""
    if freq == "daily":
        return a.join(b, how="inner")
    pf = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "annual": "Y"}[freq]
    a2, b2 = a.copy(), b.copy()
    a2.index = a2.index.to_period(pf)
    b2.index = b2.index.to_period(pf)
    a2 = a2[~a2.index.duplicated(keep="last")]
    b2 = b2[~b2.index.duplicated(keep="last")]
    j = a2.join(b2, how="inner")
    j.index = j.index.to_timestamp(how="end").normalize()
    j.index.name = "Date"
    return j


def build_demo_a(tickers, model, freq, start, end, allow_live=True):
    px, p_prov, missing = get_prices(tickers, allow_live)
    rets = prices_to_returns(px, freq)
    fac, f_prov = get_factors(model, freq, allow_live)
    df = align_by_period(rets, fac, freq)
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df = df.reset_index()
    provenance = {"prices": p_prov, "factors": f_prov, "missing_tickers": missing,
                  "transformations": [
                      f"Prices resampled to {freq} (last available price in each period)" if freq != "daily" else "Daily prices",
                      "Simple returns computed as P_t / P_(t-1) − 1 from dividend-adjusted closes",
                      "Ken French factors converted from percent to decimals",
                      "ETF returns and factors joined on calendar period (inner join)"]}
    return df, provenance


# ----------------------------------------------------------------------------- Risk-free
def fetch_fred(series_id: str, timeout=20) -> pd.Series:
    import requests

    r = requests.get(FRED_URL.format(sid=series_id), timeout=timeout, headers={"User-Agent": "ELK research app"})
    r.raise_for_status()
    s = pd.read_csv(io.StringIO(r.text))
    s.columns = ["Date", series_id]
    s["Date"] = pd.to_datetime(s["Date"])
    s[series_id] = pd.to_numeric(s[series_id], errors="coerce")
    return s.set_index("Date")[series_id].dropna()


def get_risk_free(freq: str, source: str = "Ken French (1-month T-bill)", allow_live=True) -> tuple[pd.Series, dict]:
    """Per-period risk-free return (decimal) at the requested frequency."""
    if source.startswith("Ken French"):
        if freq in ("daily", "weekly", "monthly"):
            df, prov = get_kf("ff3", freq, allow_live)
            rf = df["RF"]
        else:
            df, prov = get_kf("ff3", "monthly", allow_live)
            rf = (1 + df["RF"]).resample({"quarterly": "QE", "annual": "YE"}[freq]).prod() - 1
            prov = dict(prov, note=f"monthly RF compounded to {freq}")
        return rf.rename("RF"), dict(prov, series="RF (1-month T-bill return)")
    # FRED 3-month T-bill secondary market rate (annualised %, discount basis)
    err = None
    s = None
    if allow_live:
        try:
            s = fetch_fred("DTB3")
            prov = {"source": "FRED, Federal Reserve Bank of St. Louis (live)", "series": "DTB3 3-month T-bill",
                    "url": "https://fred.stlouisfed.org/series/DTB3", "retrieved": _now(), "mode": "live"}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    if s is None:
        snap = _read_snap("fred_dtb3")
        if snap is None:
            raise RuntimeError(f"Could not reach FRED ({err}) and no saved snapshot is available.")
        s = snap.iloc[:, 0]
        prov = {"source": "FRED (saved snapshot)", "series": "DTB3 3-month T-bill", "mode": "snapshot",
                "url": "https://fred.stlouisfed.org/series/DTB3",
                "retrieved": _snapshot_meta().get("fred_dtb3", {}).get("retrieved", "unknown"), "live_error": err}
    ppy = FREQ_PERIODS_PER_YEAR[freq]
    if freq != "daily":
        rule = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "annual": "YE"}[freq]
        s = s.resample(rule).mean()
    rf = (1 + s / 100.0) ** (1.0 / ppy) - 1
    prov = dict(prov, note=f"annualised % rate averaged per period and converted to a per-period return: (1+r)^(1/{ppy})−1")
    return rf.dropna().rename("RF"), prov


def attach_rf(dates: pd.Series, rf: pd.Series, freq: str) -> pd.Series:
    """Match a per-period RF series to arbitrary dates by calendar period."""
    d = pd.to_datetime(dates)
    if freq == "daily":
        m = rf.reindex(d.values)
        return pd.Series(m.values, index=dates.index)
    pf = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "annual": "Y"}[freq]
    r = rf.copy()
    r.index = r.index.to_period(pf)
    r = r[~r.index.duplicated(keep="last")]
    per = d.dt.to_period(pf)
    return pd.Series(r.reindex(per).values, index=dates.index)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
