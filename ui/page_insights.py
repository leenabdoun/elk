"""Step 5 — Insights & Export."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from core.insights import build_narrative, fmt, in_unit
from core.models import COV_LABELS, compare, scenario
from core.report import excel_report, html_report
from ui.analysis import diag_rows, get_diag, get_influence, get_vif
from ui.common import intercept_label, log, nav_buttons, need_model, page_header, ss

KIND = {"fact": ("Measured", "#4F7A5B"), "warning": ("Warning", "#B07D3E"), "judgment": ("Judgment", "#8A939B")}


def _ctx():
    sp = ss.spec
    return {"alpha": sp["alpha"], "units": ss.units, "alpha_ok": sp["alpha_ok"], "alpha_reason": sp["alpha_reason"],
            "is_ts": sp["is_ts"], "ppy": sp.get("ppy")}


def report_context(fit, secs):
    sp = ss.spec
    d = ss.data
    prov = list(d["provenance"]) + [("Transformation", t) for t in d.get("transforms", [])]
    m = ss.suit.metrics
    if sp["is_ts"] and m.get("start") is not None:
        prov.append(("Sample window used", f"{m['start']:%Y-%m-%d} to {m['end']:%Y-%m-%d} ({m.get('detected_frequency')})"))
    prov.append(("Observations used", f"{fit.stats['n']} of {m.get('n_raw')} rows"))
    spec = {"Outcome (Y)": f"{sp['y']} [{ss.units.get(sp['y'], 'unitless')}]",
            "Explanatory variables": ", ".join(f"{x} [{ss.units.get(x, 'unitless')}]" for x in fit.spec.xs),
            "Structure": "time series" + (f" ({sp['freq']}, date column '{sp['date_col']}')" if sp["is_ts"] else "") if sp["is_ts"]
            else "cross-sectional",
            "Standard errors": COV_LABELS[fit.spec.cov_type] + (f" ({fit.spec.hac_lags} lags)" if fit.spec.cov_type == "HAC" else ""),
            "Significance level": f"{sp['alpha']:.0%}",
            "Intercept interpretation": "alpha" if sp["alpha_ok"] else f"intercept (not alpha: {sp['alpha_reason']})",
            "Reported model": "Model B (alternative)" if fit is ss.fitB else "Model A (original)"}
    if fit.spec.exclude_rows:
        spec["Excluded rows (sensitivity)"] = str(len(fit.spec.exclude_rows))
    _, flagged = get_influence(fit)
    cmp_ = None
    if ss.fitB is not None:
        A, B = ss.fitA, ss.fitB
        cmp_ = {"a_desc": f"{A.spec.describe()} · {COV_LABELS[A.spec.cov_type]} · n={A.stats['n']}",
                "b_desc": f"{B.spec.describe()} · {COV_LABELS[B.spec.cov_type]}"
                          + (f" ({B.spec.hac_lags} lags)" if B.spec.cov_type == "HAC" else "")
                          + f" · n={B.stats['n']}" + (f" · {len(B.spec.exclude_rows)} rows excluded" if B.spec.exclude_rows else ""),
                "justification": (ss.b_cfg or {}).get("justification", ""), "result": compare(A, B, sp["alpha"]),
                "oos": ss.oos}
    return {"title": d["name"], "provenance": prov, "spec": spec, "suit": ss.suit, "fit": fit, "diag": diag_rows(fit),
            "vif": get_vif(fit), "flagged": flagged, "comparison": cmp_, "narrative": secs, "notes": ss.notes,
            "log": ss.log, "dates": ss.suit.dates, "alpha": sp["alpha"], "synthetic": d.get("synthetic"),
            "intercept_label": intercept_label(), "labels": ss.suit.labels}


def render():
    page_header("Insights & Export", "A structured reading of the results that separates what was measured, what the "
                                     "diagnostics warn about, and what remains analyst judgment.")
    if not need_model():
        return
    fit = ss.fitB if (ss.report_model == "B" and ss.fitB is not None) else ss.fitA
    sp = ss.spec
    d = get_diag(fit)
    vif = get_vif(fit)
    _, flagged = get_influence(fit)
    secs = build_narrative(fit, d, vif, flagged, ss.suit, _ctx())
    st.caption(f"Describing **{'Model B' if fit is ss.fitB else 'Model A'}**: {fit.spec.describe()} · "
               f"{COV_LABELS[fit.spec.cov_type]}. Change this at the bottom of Compare & Repair.")
    st.markdown(" ".join(f"<span class='elk-chip' style='color:{c};border-color:{c}'>{n}</span>" for n, c in KIND.values())
                + "<span style='opacity:.7;font-size:.85rem'> Measured = computed from the data · Warning = a diagnostic "
                  "concern · Judgment = interpretation that needs the analyst</span>", unsafe_allow_html=True)
    st.write("")
    letters = "abcde"
    for i, sec in enumerate(secs):
        with st.container(border=True):
            st.markdown(f"**({letters[i]}) {sec['title']}**")
            for kind, text in sec["items"]:
                name, color = KIND[kind]
                st.markdown(f"<div style='margin:.35rem 0'><span class='elk-kind' style='color:{color}'>{name}</span>{text}</div>",
                            unsafe_allow_html=True)

    # scenario ------------------------------------------------------------------
    st.markdown("##### Scenario: what does the model imply for chosen inputs?")
    with st.container(border=True):
        st.caption("Plug in values for each explanatory variable. This applies the estimated relationship — it is **not a "
                   "forecast** unless the model has been validated out of sample. Defaults are sample means.")
        X = fit.frame[fit.spec.xs]
        cols = st.columns(min(4, len(fit.spec.xs)))
        vals = {}
        for i, x in enumerate(fit.spec.xs):
            lo, hi, mu = float(X[x].min()), float(X[x].max()), float(X[x].mean())
            if set(np.unique(X[x])) <= {0.0, 1.0}:
                vals[x] = float(cols[i % len(cols)].selectbox(x, [0, 1], index=int(mu >= 0.5), key=f"scn_{fit.spec.label}_{x}",
                                                              help=f"0/1 indicator; {mu:.0%} of observations are 1"))
                continue
            mu = float(f"{mu:.4g}")
            step = float(10 ** np.floor(np.log10(max(abs(hi - lo), 1e-9)) - 2))
            vals[x] = cols[i % len(cols)].number_input(x, value=mu, step=step, format="%.6g", key=f"scn_{fit.spec.label}_{x}",
                                                       help=f"Sample range {fmt(lo)} to {fmt(hi)}")
        r = scenario(fit, vals, sp["alpha"])
        uy = ss.units.get(sp["y"], "unitless / other")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Predicted {sp['y']}", in_unit(r["mean"], uy))
        c2.metric(f"{1 - sp['alpha']:.0%} CI for the average", f"{in_unit(r['ci'][0], uy)} to {in_unit(r['ci'][1], uy)}",
                  help="Uncertainty about the average outcome for these inputs.")
        c3.metric(f"{1 - sp['alpha']:.0%} prediction interval", f"{in_unit(r['pi'][0], uy)} to {in_unit(r['pi'][1], uy)}",
                  help="Range for a single new observation — much wider, because it includes residual noise.")
        if r["outside"]:
            st.warning(f"**Extrapolation:** {', '.join(r['outside'])} outside the range observed in the sample. The model has "
                       "no evidence about this region; treat the result as illustrative.", icon=":material/warning:")
        elif r["hidden_extrapolation"]:
            st.warning("**Unusual combination:** each input is within its own observed range, but the combination is unlike "
                       "any observation in the sample (hidden extrapolation).", icon=":material/warning:")

    # notes & export -----------------------------------------------------------
    st.markdown("##### Analyst notes")
    ss.notes = st.text_area("Your judgment, context and caveats — included in the report", value=ss.notes, height=110,
                            label_visibility="collapsed",
                            placeholder="e.g. Sample covers a single rate cycle; results should be refreshed after …")

    st.markdown("##### Export")
    ctx = report_context(fit, secs)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ss.data["name"]).strip("_")[:40]
    c1, c2, c3 = st.columns(3)
    try:
        html = html_report(ctx)
        c1.download_button("Report (HTML · print to PDF)", html.encode("utf-8"), f"ELK_report_{slug}.html", "text/html",
                           type="primary", icon=":material/description:", width="stretch",
                           on_click=lambda: log("Report exported", "HTML"))
    except Exception as e:
        c1.error(f"Report failed: {e}")
    try:
        xl = excel_report(ctx)
        c2.download_button("Results workbook (Excel)", xl, f"ELK_results_{slug}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/table_view:", width="stretch", on_click=lambda: log("Results exported", "Excel"))
    except Exception as e:
        c2.error(f"Workbook failed: {e}")
    data = fit.frame.copy()
    data.insert(0, "row", ss.suit.labels.reindex(data.index).values)
    c3.download_button("Data used (CSV)", data.to_csv(index=False).encode(), f"ELK_data_used_{slug}.csv", "text/csv",
                       icon=":material/download:", width="stretch")
    st.caption("Open the HTML report in your browser and use Print → Save as PDF for a PDF copy. The report records the "
               "data source, sample, variables, adjustments, diagnostics and the audit trail below.")
    with st.expander(f"Audit trail ({len(ss.log)} steps)"):
        st.dataframe(pd.DataFrame(ss.log), hide_index=True, width="stretch")
    nav_buttons(3, None)
