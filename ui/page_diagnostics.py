"""Step 3 — Diagnostic Center."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from core import diagnostics as D
from core.insights import fmt, fmt_p
from ui.analysis import get_diag, get_influence, get_vif
from ui.common import (ACCENT, LEVEL_ICON, LEVEL_WORD, LINE, POINT, chip, explain, nav_buttons, need_model, page_header,
                       show, ss, style_fig)


def _test_card(t, alpha):
    with st.container(border=True):
        st.markdown(f"{chip(LEVEL_WORD[t['level']], t['level'])}<br><b>{t['name']}</b>", unsafe_allow_html=True)
        stat = fmt(t["stat"]) if np.isfinite(t["stat"]) else "—"
        ptxt = f" · p-value **{fmt_p(t['p'])}**" if np.isfinite(t["p"]) else ""
        st.markdown(f"Statistic **{stat}**{ptxt}  \n<span style='opacity:.7'>{t.get('extra', '')}</span>",
                    unsafe_allow_html=True)
        explain(t["key"])


def residual_figs(fit, dates):
    r = fit.res.resid
    fv = fit.res.fittedvalues
    lab = ss.suit.labels.reindex(fit.frame.index)
    z = (r - r.mean()) / r.std(ddof=1)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Residuals vs fitted", "Normal Q–Q", "Scale–location",
                                                        "Residual distribution"), vertical_spacing=.14, horizontal_spacing=.08)
    fig.add_scatter(x=fv, y=r, mode="markers", marker=dict(color=POINT, size=5, opacity=.7), text=lab,
                    hovertemplate="%{text}<br>fitted %{x:.4g}<br>resid %{y:.4g}", row=1, col=1, showlegend=False)
    fig.add_hline(y=0, line=dict(color=ACCENT, dash="dash"), row=1, col=1)
    order = np.argsort(z.values)
    q = stats.norm.ppf((np.arange(1, len(z) + 1) - .5) / len(z))
    fig.add_scatter(x=q, y=z.values[order], mode="markers", marker=dict(color=POINT, size=5, opacity=.7),
                    text=lab.values[order], hovertemplate="%{text}", row=1, col=2, showlegend=False)
    fig.add_scatter(x=[q.min(), q.max()], y=[q.min(), q.max()], mode="lines", line=dict(color=ACCENT, dash="dash"),
                    row=1, col=2, showlegend=False)
    fig.add_scatter(x=fv, y=np.sqrt(np.abs(z)), mode="markers", marker=dict(color=POINT, size=5, opacity=.7), text=lab,
                    hovertemplate="%{text}", row=2, col=1, showlegend=False)
    fig.add_histogram(x=z, nbinsx=40, marker=dict(color=POINT), row=2, col=2, showlegend=False)
    xs = np.linspace(z.min(), z.max(), 100)
    binw = (z.max() - z.min()) / 40
    fig.add_scatter(x=xs, y=stats.norm.pdf(xs) * len(z) * binw, mode="lines", line=dict(color=ACCENT), row=2, col=2,
                    showlegend=False)
    fig.update_xaxes(title_text="Fitted", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical quantile", row=1, col=2)
    fig.update_xaxes(title_text="Fitted", row=2, col=1)
    fig.update_xaxes(title_text="Standardised residual", row=2, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="√|std. residual|", row=2, col=1)
    return style_fig(fig, 660, top=36)


def ts_figs(fit, dates):
    r = fit.res.resid
    d = dates.reindex(fit.frame.index)
    lags, vals, band = D.residual_acf(fit, 24)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals over time", "Residual autocorrelation (ACF)"),
                        horizontal_spacing=.08)
    fig.add_scatter(x=d, y=r, mode="lines", line=dict(color=POINT, width=1.2), row=1, col=1, showlegend=False)
    fig.add_hline(y=0, line=dict(color=ACCENT, dash="dash"), row=1, col=1)
    fig.add_bar(x=lags[1:], y=vals[1:], marker=dict(color=LINE), row=1, col=2, showlegend=False)
    for b in (band, -band):
        fig.add_hline(y=b, line=dict(color=ACCENT, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    return style_fig(fig, 340, top=36)


def render():
    page_header("Diagnostics", "Checks on whether the regression's assumptions hold. Each test explains what it checks, its "
                               "null hypothesis, and its limits. Visual evidence complements formal tests.")
    if not need_model():
        return
    fit = ss.fitA
    sp = ss.spec
    alpha = sp["alpha"]
    lags = ss.setdefault("diag_lags", {})
    d = get_diag(fit)
    vif = get_vif(fit)
    full, flagged = get_influence(fit)

    # summary strip
    chips = []
    for k in ["bp", "bg", "dw", "arch", "jb", "reset"]:
        if k in d:
            chips.append(chip(d[k]["name"].split(" ")[0], d[k]["level"]))
    if len(vif):
        mx = vif["VIF"].max()
        chips.append(chip(f"VIF max {mx:.1f}", "warn" if mx > 10 else "ok" if mx <= 5 else "warn"))
    chips.append(chip(f"{len(flagged)} influential", "warn" if len(flagged) else "ok"))
    st.markdown("<div style='line-height:2.1'>" + " ".join(chips) + "</div>", unsafe_allow_html=True)
    st.caption(f"Significance level {alpha:.0%}. Amber = flagged for attention, green = no problem detected at this level "
               "(which does not prove the assumption holds).")
    if not sp["is_ts"]:
        st.caption(":material/info: Cross-sectional data: Durbin–Watson, Breusch–Godfrey, ARCH and ADF are not shown because "
                   "row order has no meaning.")

    tabs = st.tabs(["Tests", "Residual plots", "Collinearity", "Influential observations"]
                   + (["Stationarity (ADF)"] if sp["is_ts"] else []))
    with tabs[0]:
        groups = [("Variance", ["bp"]), ("Autocorrelation & volatility", ["dw", "bg", "arch"] if sp["is_ts"] else []),
                  ("Distribution & form", ["jb", "reset"])]
        for title, keys in groups:
            if not keys:
                continue
            st.markdown(f"###### {title}")
            cols = st.columns(len(keys))
            for col, k in zip(cols, keys):
                with col:
                    _test_card(d[k], alpha)
        if sp["is_ts"]:
            with st.popover("Lag settings", icon=":material/tune:"):
                n = fit.stats["n"]
                bg = st.number_input("Breusch–Godfrey lags", 1, max(1, n // 3), int(d["bg"].get("lags") or 1))
                ar = st.number_input("ARCH LM lags", 1, max(1, n // 3), int(d.get("arch", {}).get("lags") or 1))
                if st.button("Apply", type="primary"):
                    ss.diag_lags = {"bg": int(bg), "arch": int(ar)}
                    st.rerun()

    with tabs[1]:
        show(residual_figs(fit, ss.suit.dates), "resid4")
        if sp["is_ts"]:
            show(ts_figs(fit, ss.suit.dates), "residts")
        st.caption("Look for: a funnel shape in residuals vs fitted (changing variance), curvature (missing nonlinearity), "
                   "points far off the Q–Q line (fat tails or outliers), and slow-decaying ACF bars (autocorrelation).")

    with tabs[2]:
        if len(vif) == 0:
            st.info("Collinearity applies only with two or more explanatory variables.")
        else:
            c1, c2 = st.columns([1, 1.3])
            with c1:
                st.dataframe(vif, hide_index=True, width="stretch",
                             column_config={"VIF": st.column_config.ProgressColumn("VIF", format="%.2f", min_value=0,
                                                                                   max_value=float(max(12, vif["VIF"].max())))})
                explain("vif")
            with c2:
                corr = fit.frame[sp["xs"]].corr()
                fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                           colorscale=[[0, "#4A6684"], [.5, "#F0F1F3"], [1, "#B07D3E"]],
                                           text=np.round(corr.values, 2), texttemplate="%{text}", showscale=False))
                fig.update_yaxes(autorange="reversed")
                show(style_fig(fig, 360, "Correlation between explanatory variables"), "corr")

    with tabs[3]:
        n = fit.stats["n"]
        fig = go.Figure()
        size = 6 + 30 * np.sqrt(full["cooks_d"] / max(full["cooks_d"].max(), 1e-12))
        colors = np.where(full["flag"] != "", ACCENT, POINT)
        fig.add_scatter(x=full["leverage"], y=full["studentized_resid"], mode="markers", text=full["row"],
                        marker=dict(size=size, color=colors, opacity=.75, line=dict(width=0)),
                        hovertemplate="%{text}<br>leverage %{x:.3f}<br>stud. resid %{y:.2f}")
        for yv in (3, -3):
            fig.add_hline(y=yv, line=dict(color=ACCENT, dash="dot"))
        fig.update_xaxes(title="Leverage (hat value)")
        fig.update_yaxes(title="Studentised residual")
        show(style_fig(fig, 380, "Influence plot (bubble size = Cook's distance)"), "infl")
        if len(flagged):
            st.markdown(f"**{len(flagged)} flagged observation(s)** — review, do not delete automatically. "
                        f"Cook's distance threshold 4/n = {4 / n:.4f}.")
            st.dataframe(flagged[["row", "actual", "fitted", "residual", "studentized_resid", "leverage", "cooks_d", "flag"]],
                         hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4g") for c in
                                        ["actual", "fitted", "residual", "studentized_resid", "leverage", "cooks_d"]})
        else:
            st.success("No observations exceed the outlier or influence thresholds.", icon=":material/check_circle:")
        explain("influence")

    if sp["is_ts"]:
        with tabs[4]:
            st.markdown("Unit-root test on selected series. Returns are usually stationary; price or NAV levels usually are not.")
            c1, c2, c3 = st.columns([2, 1, 1])
            series = c1.multiselect("Series", [sp["y"]] + sp["xs"], default=[sp["y"]], key="adf_series")
            reg = c2.selectbox("Deterministic terms", ["c", "ct", "n"],
                               format_func={"c": "constant", "ct": "constant + trend", "n": "none"}.get, key="adf_reg")
            lagmode = c3.selectbox("Lag selection", ["AIC (automatic)", "BIC (automatic)", "fixed"], key="adf_lag")
            fixed = None
            if lagmode == "fixed":
                fixed = st.number_input("Lags", 0, 24, 1, key="adf_fixed")
            rows = []
            for s in series:
                r = D.adf_test(fit.frame[s], reg, None if lagmode == "fixed" else lagmode[:3], fixed)
                rows.append({"series": s, "ADF statistic": r["stat"], "p-value": r["p"], "lags used": r["lags"],
                             "obs": r["nobs"], "5% critical": r["crit"].get("5%", np.nan),
                             "reading": ("rejects unit root → stationary" if r["p"] < alpha else
                                         "cannot reject unit root → possibly non-stationary") if np.isfinite(r["p"]) else r["note"]})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                             column_config={"ADF statistic": st.column_config.NumberColumn(format="%.3f"),
                                            "p-value": st.column_config.NumberColumn(format="%.4f"),
                                            "5% critical": st.column_config.NumberColumn(format="%.3f")})
            explain("adf")

    nav_buttons(1, 3, "Compare & repair")
