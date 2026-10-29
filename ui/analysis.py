"""Cached diagnostic computations shared by several pages."""
from __future__ import annotations

from core import diagnostics as D
from ui.common import fit_cache, ss


def get_diag(fit):
    lags = ss.get("diag_lags", {})
    key = ("diag", lags.get("bg"), lags.get("arch"), ss.spec["alpha"])
    c = fit_cache(fit)
    if key not in c:
        c[key] = D.run_core(fit, ss.spec["alpha"], ss.spec["is_ts"], ss.spec.get("freq"), lags.get("bg"), lags.get("arch"))
    return c[key]


def get_vif(fit):
    c = fit_cache(fit)
    if "vif" not in c:
        c["vif"] = D.vif_table(fit)
    return c["vif"]


def get_influence(fit):
    c = fit_cache(fit)
    if "infl" not in c:
        c["infl"] = D.influence_table(fit, ss.suit.labels)
    return c["infl"]


def diag_rows(fit) -> list[dict]:
    d = get_diag(fit)
    rows = []
    for k in ["dw", "bg", "bp", "arch", "jb", "reset"]:
        if k in d:
            t = d[k]
            rows.append({"Test": t["name"], "Statistic": t["stat"], "p-value": t["p"],
                         "Result": {"ok": "no problem detected", "warn": "flagged", "na": "not available"}[t["level"]],
                         "Detail": t.get("extra", "")})
    v = get_vif(fit)
    if len(v):
        rows.append({"Test": "Max VIF", "Statistic": float(v["VIF"].max()), "p-value": float("nan"),
                     "Result": "flagged" if v["VIF"].max() > 10 else ("moderate" if v["VIF"].max() > 5 else "no problem detected"),
                     "Detail": v.sort_values("VIF").iloc[-1]["variable"]})
    _, fl = get_influence(fit)
    rows.append({"Test": "Influential / outlying observations", "Statistic": float(len(fl)), "p-value": float("nan"),
                 "Result": "flagged" if len(fl) else "no problem detected", "Detail": "count; |stud. resid| > 3 or Cook's D > 4/n"})
    return rows
