"""Step 2 — Regression Lab: OLS estimates, fit, ANOVA and plain-English reading."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.insights import build_narrative, fmt, fmt_p, in_unit
from core.models import ModelSpec, fit_ols
from ui.common import (ACCENT, LINE, POINT, coef_table, go_to, intercept_label, log, nav_buttons, page_header, show,
                       ss, style_fig)


def ensure_model_a():
    """(Re)estimate the baseline model when the specification has changed."""
    sp, suit = ss.spec, ss.suit
    if suit is None or suit.blocked or not sp.get("xs"):
        return None
    key = (sp["y"], tuple(sp["xs"]), sp["date_col"], sp["id_col"], len(suit.frame), ss.data["name"],
           len(ss.data.get("transforms", [])))
    if ss.fitA is None or getattr(ss.fitA, "key", None) != key:
        fit = fit_ols(suit.frame, ModelSpec(sp["y"], sp["xs"], "nonrobust", label="Model A"))
        fit.key = key
        ss.fitA = fit
        ss.fitB = None
        ss.b_cfg = None
        ss.oos = None
        log("Model A estimated", f"{fit.spec.describe()} · n = {fit.stats['n']} · classical SEs")
    return ss.fitA


def actual_vs_fitted(fit, dates, title="Actual vs fitted"):
    y = fit.frame[fit.spec.y]
    fv = fit.res.fittedvalues
    fig = go.Figure()
    if dates is not None:
        d = dates.reindex(fit.frame.index)
        fig.add_scatter(x=d, y=y, name="Actual", mode="lines", line=dict(color=POINT, width=1.4))
        fig.add_scatter(x=d, y=fv, name="Fitted", mode="lines", line=dict(color=ACCENT, width=1.6))
    else:
        fig.add_scatter(x=fv, y=y, mode="markers", name="Observations", marker=dict(color=POINT, size=6, opacity=.7),
                        text=ss.suit.labels.reindex(fit.frame.index), hovertemplate="%{text}<br>fitted %{x:.4g}<br>actual %{y:.4g}")
        lo, hi = float(min(fv.min(), y.min())), float(max(fv.max(), y.max()))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45° line (perfect fit)",
                        line=dict(color=ACCENT, width=1.4, dash="dash"))
        fig.update_xaxes(title="Fitted")
        fig.update_yaxes(title="Actual")
    return style_fig(fig, 360, title)


def simple_scatter(fit):
    x = fit.spec.xs[0]
    d = fit.frame
    fig = go.Figure()
    fig.add_scatter(x=d[x], y=d[fit.spec.y], mode="markers", name="Observations",
                    marker=dict(color=POINT, size=6, opacity=.7))
    xx = np.linspace(d[x].min(), d[x].max(), 50)
    b0, b1 = fit.coef.loc["const", "estimate"], fit.coef.loc[x, "estimate"]
    fig.add_scatter(x=xx, y=b0 + b1 * xx, mode="lines", name="Fitted line", line=dict(color=ACCENT, width=2))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=fit.spec.y)
    return style_fig(fig, 360, f"{fit.spec.y} vs {x}")


def render():
    page_header("Regression Lab", "Ordinary least squares with an intercept. Coefficients describe associations in this "
                                  "sample — not causal effects.")
    fit = ensure_model_a()
    if fit is None:
        st.info("Choose variables and pass the suitability check first.", icon=":material/info:")
        if st.button("Go to Data & Suitability", type="primary"):
            go_to(0)
        return
    sp = ss.spec
    a = sp["alpha"]
    st.markdown(f"<div class='elk-eq'>{sp['y']} = {'α' if sp['alpha_ok'] else 'β₀'} + "
                + " + ".join(f"β<sub>{i + 1}</sub>·{x}" for i, x in enumerate(sp["xs"])) + " + ε</div>",
                unsafe_allow_html=True)
    if ss.data.get("synthetic"):
        st.caption(":material/science: Synthetic demonstration data.")
    st.write("")
    s = fit.stats
    c = st.columns(5)
    c[0].metric("Observations", f"{s['n']:,}", help=f"Residual degrees of freedom: {s['df_resid']:.0f}")
    c[1].metric("R²", f"{s['r2']:.3f}", help="Share of the variation in Y explained by the model, in sample.")
    c[2].metric("Adjusted R²", f"{s['adj_r2']:.3f}", help="R² penalised for the number of explanatory variables.")
    c[3].metric("F-test p-value", fmt_p(s["f_pvalue"]), help="Null hypothesis: all slope coefficients are zero.")
    c[4].metric("Residual std. error", fmt(s["rmse"]), help="Typical size of a residual, in Y's units.")

    if sp["alpha_ok"]:
        r = fit.coef.loc["const"]
        uy = ss.units.get(sp["y"])
        ppy = sp.get("ppy") or 12
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Alpha per period", in_unit(r["estimate"], uy))
            c2.metric("Alpha, annualised (×%d)" % ppy, in_unit(r["estimate"] * ppy, uy),
                      help="Simple annualisation: per-period alpha × periods per year.")
            c3.metric("Alpha p-value", fmt_p(r["p_value"]),
                      help=f"{'Significant' if r['p_value'] < a else 'Not significant'} at the {a:.0%} level.")
            st.caption("Alpha is the average excess return not explained by the factors. A significant alpha in one sample "
                       "is not proof of skill; check diagnostics and robust standard errors before relying on it.")

    st.markdown("##### Coefficients")
    coef_table(fit.coef, a)
    st.caption(f"Classical standard errors. Robust alternatives are available in **Compare & Repair**. "
               f"{intercept_label()} = predicted {sp['y']} when every X is zero.")

    ctx = {"alpha": a, "units": ss.units, "alpha_ok": sp["alpha_ok"], "alpha_reason": sp["alpha_reason"],
           "is_ts": sp["is_ts"], "ppy": sp.get("ppy")}
    secs = build_narrative(fit, {}, None, None, ss.suit, ctx)
    with st.container(border=True):
        st.markdown("##### Reading the estimates")
        for kind, text in secs[0]["items"]:
            st.markdown(f"- {text}")

    left, right = st.columns(2)
    with left:
        show(actual_vs_fitted(fit, ss.suit.dates), "avf")
    with right:
        if len(sp["xs"]) == 1:
            show(simple_scatter(fit), "sc")
        else:
            t = fit.coef.drop(index="const")
            fig = go.Figure()
            fig.add_scatter(x=t["estimate"], y=t.index, mode="markers", marker=dict(color=LINE, size=9),
                            error_x=dict(type="data", symmetric=False, array=t["ci_high"] - t["estimate"],
                                         arrayminus=t["estimate"] - t["ci_low"], color=POINT), name="Estimate")
            fig.add_vline(x=0, line=dict(color=ACCENT, dash="dash"))
            fig.update_yaxes(autorange="reversed")
            show(style_fig(fig, 360, f"Slope estimates with {1 - a:.0%} confidence intervals"), "ci")
            st.caption("Slopes are in different units, so compare each interval with zero rather than with each other.")

    with st.expander("ANOVA table and full statistics"):
        st.dataframe(fit.anova, hide_index=True, width="stretch",
                     column_config={"SS": st.column_config.NumberColumn(format="%.6g"),
                                    "MS": st.column_config.NumberColumn(format="%.6g"),
                                    "F": st.column_config.NumberColumn(format="%.4f"),
                                    "p_value": st.column_config.NumberColumn("p-value", format="%.4g")})
        st.write({k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()})
    nav_buttons(0, 2, "Check diagnostics")
