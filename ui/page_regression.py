"""Step 2 — Regression Lab: key findings first, then a plain driver summary and charts, statistics on demand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.findings import driver_table, nm, takeaways
from core.insights import fmt, fmt_p, in_unit
from core.models import ModelSpec, fit_ols
from ui.analysis import get_diag, get_influence, get_vif
from ui.common import (ACCENT, LINE, POINT, coef_table, finding_card, intercept_label, log, nav_buttons, need_model,
                       page_header, show, ss, style_fig)


def ensure_model_a():
    """Fit (or refit) Model A once the analyst has run the analysis and the current setup is valid.
    Any change to data or variables changes the key, so stale results are never shown."""
    sp, suit = ss.spec, ss.suit
    if not ss.get("analysis_ran") or not ss.get("spec_ready") or suit is None or suit.blocked or not sp.get("xs"):
        if ss.fitA is not None and ss.get("analysis_ran") and not ss.get("spec_ready"):
            ss.fitA = None
            ss.fitB = None
        return None
    key = (sp["y"], tuple(sp["xs"]), sp["date_col"], sp["id_col"], len(suit.frame), ss.data["name"],
           len(ss.data.get("transforms", [])), sp.get("freq"))
    if ss.fitA is None or getattr(ss.fitA, "key", None) != key:
        refit = ss.fitA is not None
        fit = fit_ols(suit.frame, ModelSpec(sp["y"], sp["xs"], "nonrobust", label="Model A"))
        fit.key = key
        ss.fitA = fit
        ss.fitB = None
        ss.b_cfg = None
        ss.oos = None
        ss.report_model = "A"
        log("Model A estimated", f"{fit.spec.describe()} · n = {fit.stats['n']}")
        if refit:
            st.toast("Results updated for your new selection.", icon=":material/refresh:")
    return ss.fitA


def actual_vs_fitted(fit, dates, title="Model vs actual"):
    y = fit.frame[fit.spec.y]
    fv = fit.res.fittedvalues
    fig = go.Figure()
    if dates is not None:
        d = dates.reindex(fit.frame.index)
        fig.add_scatter(x=d, y=y, name="Actual", mode="lines", line=dict(color=POINT, width=1.4))
        fig.add_scatter(x=d, y=fv, name="Model", mode="lines", line=dict(color=ACCENT, width=1.6))
    else:
        fig.add_scatter(x=fv, y=y, mode="markers", name="Observations", marker=dict(color=POINT, size=6, opacity=.7),
                        text=ss.suit.labels.reindex(fit.frame.index), hovertemplate="%{text}<br>model %{x:.4g}<br>actual %{y:.4g}")
        lo, hi = float(min(fv.min(), y.min())), float(max(fv.max(), y.max()))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                        line=dict(color=ACCENT, width=1.4, dash="dash"))
        fig.update_xaxes(title="What the model expects")
        fig.update_yaxes(title="Actual")
    return style_fig(fig, 360, title)


def importance_chart(rows, y_label):
    r = [x for x in rows if x["typical"] is not None] or rows
    r = sorted(r, key=lambda x: x["importance"] if np.isfinite(x["importance"]) else 0)
    colors = [ACCENT if x["evidence"] in ("strong evidence", "moderate evidence") else "#C2C8CE" for x in r]
    fig = go.Figure(go.Bar(x=[x["importance"] for x in r], y=[x["label"] for x in r], orientation="h",
                           marker=dict(color=colors), text=[x["evidence"] for x in r], textposition="auto",
                           hovertemplate="%{y}: a typical change moves the outcome by %{x:.0f}% of its usual variation<extra></extra>"))
    fig.update_xaxes(title=f"% of the usual variation in {y_label}")
    return style_fig(fig, max(240, 60 + 42 * len(r)), "How much each driver matters")


def render():
    page_header("Regression Lab", "What the data says about your question — in plain terms first, with the full "
                                  "statistics one click away.")
    if not need_model():
        return
    fit = ss.fitA
    sp = ss.spec
    a = sp["alpha"]
    desc = ss.descriptions
    if sp.get("question"):
        st.markdown(f":material/help_outline: **Your question:** {sp['question']}")
    if ss.data.get("synthetic"):
        st.caption(":material/science: Synthetic demonstration data.")

    d, v = get_diag(fit), get_vif(fit)
    _, flagged = get_influence(fit)
    tk = takeaways(fit, d, v, flagged, ss.suit, sp, ss.units, desc)
    st.markdown("##### Key takeaways")
    cols = st.columns(2)
    for i, f in enumerate(tk):
        with cols[i % 2]:
            finding_card(f)

    st.markdown("##### Driver by driver")
    rows = driver_table(fit, sp, ss.units, desc)
    tbl = pd.DataFrame([{"Driver": r["label"], "What the data shows": r["effect"],
                         "A typical change in the driver moves the outcome by": r["typical"] or "— (yes/no variable)",
                         "Evidence": r["evidence"]} for r in rows]).set_index("Driver")
    st.table(tbl)
    st.caption(f"Each effect holds the other drivers fixed. 'Evidence' reflects how confident the data lets us be "
               f"(using a {a:.0%} significance threshold); it says nothing about whether the effect is large enough "
               "to matter for the investment — judge that from the size.")

    left, right = st.columns(2)
    with left:
        show(importance_chart(rows, nm(sp["y"], desc)), "imp")
    with right:
        show(actual_vs_fitted(fit, ss.suit.dates), "avf")

    with st.expander("See statistical details", icon=":material/table_view:"):
        s = fit.stats
        c = st.columns(5)
        c[0].metric("Observations", f"{s['n']:,}", help=f"Residual degrees of freedom: {s['df_resid']:.0f}")
        c[1].metric("R²", f"{s['r2']:.3f}", help="Share of the variation in Y explained by the model, in sample.")
        c[2].metric("Adjusted R²", f"{s['adj_r2']:.3f}", help="R² penalised for the number of drivers.")
        c[3].metric("F-test p-value", fmt_p(s["f_pvalue"]), help="Null hypothesis: all slope coefficients are zero.")
        c[4].metric("Residual std. error", fmt(s["rmse"]), help="Typical size of a residual, in Y's units.")
        st.markdown(f"<div class='elk-eq'>{sp['y']} = {'α' if sp['alpha_ok'] else 'β₀'} + "
                    + " + ".join(f"β<sub>{i + 1}</sub>·{x}" for i, x in enumerate(sp["xs"])) + " + ε</div>",
                    unsafe_allow_html=True)
        coef_table(fit.coef, a)
        st.caption(f"Classical standard errors; robust alternatives are in Compare & Adjust. {intercept_label()} = the "
                   f"model's {sp['y']} when every driver is zero"
                   + ("" if sp["alpha_ok"] else f" (not read as alpha: {sp['alpha_reason']})") + ".")
        if sp["alpha_ok"]:
            r = fit.coef.loc["const"]
            ppy = sp.get("ppy") or 12
            st.caption(f"Alpha per period {in_unit(r['estimate'], ss.units.get(sp['y']))}; simple annualised "
                       f"(×{ppy}) {in_unit(r['estimate'] * ppy, ss.units.get(sp['y']))}; p-value {fmt_p(r['p_value'])}.")
        st.dataframe(fit.anova, hide_index=True, width="stretch",
                     column_config={"SS": st.column_config.NumberColumn(format="%.6g"),
                                    "MS": st.column_config.NumberColumn(format="%.6g"),
                                    "F": st.column_config.NumberColumn(format="%.4f"),
                                    "p_value": st.column_config.NumberColumn("p-value", format="%.4g")})
    nav_buttons(0, 2, "Model checks")
