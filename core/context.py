"""Best-guess units and plain labels for variables. Guesses are only ever *pre-selections* the analyst confirms."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

FACTOR_NAMES = {"mkt-rf", "smb", "hml", "rmw", "cma", "mom", "umd", "rf"}


def guess_unit(name: str, s: pd.Series) -> str:
    n = name.lower()
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return "unitless / other"
    uniq = set(np.unique(v))
    if uniq <= {0.0, 1.0}:
        return "0/1 indicator"
    med = float(v.abs().median())
    mx = float(v.abs().max())
    tokens = set(re.split(r"[^a-z0-9]+", n))
    if "bps" in tokens or "bp" in tokens or n.endswith("_bps"):
        return "basis points"
    if n.startswith("log") or "log" in tokens or n.startswith("ln_"):
        return "log value"
    if n in FACTOR_NAMES or tokens & {"return", "returns", "ret", "excess", "alpha", "ytd", "irr"}:
        return "decimal return (0.01 = 1%)" if med < 0.2 and mx < 3 else "percent return (1 = 1%)"
    if n.endswith("_x") or tokens & {"leverage", "coverage", "multiple", "multiples", "turns", "dscr", "icr"}:
        return "multiple (x)"
    if tokens & {"pct", "percent", "share", "rate", "yield", "margin", "ltv", "occupancy", "growth", "utilisation", "utilization"}:
        return "ratio (0–1)" if mx <= 1.5 else "percent (1 = 1%)"
    if tokens & {"musd", "usdm", "mm", "mn", "millions", "usd_m"} or re.search(r"(usd|\$)_?m\b|_m$|musd", n):
        return "USD millions"
    if tokens & {"usd", "price", "nav", "revenue", "sales", "ebitda", "noi", "rent", "capex", "debt", "cash"}:
        return "USD"
    if tokens & {"year", "yr"} and v.between(1900, 2100).all():
        return "unitless / other"
    return "unitless / other"


def label(name: str, descriptions: dict) -> str:
    d = (descriptions or {}).get(name, "")
    return f"{name} ({d})" if d else name
