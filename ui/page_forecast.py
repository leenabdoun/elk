"""Compare & Adjust → Forecast accuracy: does the model predict unseen data better than a simple benchmark?"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.forecast import TIMING, default_timing, evaluate
from ui.common import ACCENT, LINE, POINT, finding_card, log, show, ss, style_fig


def _frame(A, B):
    """Rows usable by every compared model, in date order. Never drops rows flagged with hindsight."""
    frame = ss.suit.frame
    need = list(dict.fromkeys(list(A.spec.xs) + (list(B.spec.xs) if B is not None else [])))
    if set(need) - set(frame.columns):
        from core.suitability import assess
        sp = ss.spec
        s2 = assess(ss.data["df"], sp["y"], need, sp["date_col"], sp["id_col"], sp["freq"])
        if s2.blocked:
            return None
        frame = s2.frame
    return frame


def _data_notes(cols) -> list[str]:
    """Look-ahead warnings from the data preparation steps (time series only)."""
    out = []
    for t in ss.data.get("transforms", []):
        name = t.split(":")[0]
        if name in cols and "column median" in t:
            out.append(f"'{name}' had gaps filled with the median of the whole column, which uses values from the test "
                       "period. Prefer 'Previous value (time order)' for time series.")
    if (ss.suit.metrics.get("duplicate_dates") or 0) > 0:
        out.append("Several rows share a date (panel data). The chronological split treats them as one series, so "
                   "treat these results with caution.")
    return out


def render_forecast():
    A, B, sp = ss.fitA, ss.fitB, ss.spec
    st.caption("Does the model predict values it has not seen better than a simple no-skill benchmark? ELK hides part "
               "of the data, estimates each model on the rest, and scores the predictions. Robust standard errors "
               "(HC/HAC) don't appear here: they change p-values, not predictions.")
    is_ts = sp["is_ts"]
    b_xs = list(B.spec.xs) if B is not None and set(B.spec.xs) != set(A.spec.xs) else None
    with st.container(border=True):
        if is_ts:
            c1, c2 = st.columns([1.5, 1])
            ss.setdefault("fc_timing", default_timing(A.spec.xs))
            timing = c1.radio("When you make a forecast, are the drivers' values already known?", list(TIMING),
                              format_func=TIMING.get, key="fc_timing",
                              help="Most drivers (factor returns, economic data) are only known after the period ends. "
                                   "Using same-period values then flatters the model. Columns already lagged (e.g. "
                                   "lag1(x)) are never lagged twice.")
            hold = c2.slider("Test period: most recent share of the data", 0.1, 0.4, 0.2, 0.05, key="oos_hold")
            refit = c2.toggle("Re-estimate before every forecast", True, key="fc_refit",
                              help="On: each forecast uses a model estimated on all data up to that point (expanding "
                                   "window). Off: one model estimated on the training period only.")
        else:
            timing, hold, refit = "known", 0.2, True
            st.caption("Cross-sectional data: 5-fold cross-validation with a fixed seed. Each row is predicted by models "
                       "that never saw it.")

    key = (id(A), id(B), tuple(b_xs or []), timing, hold, refit, sp["alpha"])
    if not ss.oos or ss.oos.get("key") != key:
        frame = _frame(A, B)
        if frame is None:
            st.error("The rows needed for Model B's drivers could not be assembled.")
            return
        cols = list(dict.fromkeys(list(A.spec.xs) + (b_xs or [])))
        with st.spinner("Scoring forecasts…"):
            res = evaluate(frame, sp["y"], list(A.spec.xs), is_ts, sp["alpha"], b_xs, timing, hold, refit,
                           unit=ss.units.get(sp["y"], ""), extra_notes=_data_notes(cols) if is_ts else None)
        res["key"] = key
        if B is not None and not b_xs:
            what = (f"only changes the standard errors ({B.spec.cov_type})" if not B.spec.exclude_rows else
                    "only excludes rows that were flagged using the full sample, which a real forecast couldn't do")
            res.setdefault("notes", []).insert(0, f"Model B uses the same drivers as your regression. It {what}, so it "
                                                  "isn't a separate forecasting model. Its job is the robustness check.")
        ss.oos = res
        if res.get("ok"):
            log("Forecast comparison", res["method"] + (f" Timing: {TIMING[timing]}." if is_ts else ""))
    res = ss.oos
    if not res.get("ok"):
        st.warning(res.get("error", "The forecast comparison could not run."), icon=":material/warning:")
        return

    finding_card(res["verdict"])
    t = res["table"]
    nf = st.column_config.NumberColumn
    st.dataframe(t[["Model", "MAE", "RMSE", "MAE improvement (%)", "RMSE improvement (%)", "Beats benchmark?"]],
                 hide_index=True, width="stretch", column_config={
                     "MAE": nf("MAE", format="%.4g", help="Average size of a miss, in the outcome's units. Lower is better."),
                     "RMSE": nf("RMSE", format="%.4g", help="Like MAE, but large misses count more. Lower is better."),
                     "MAE improvement (%)": nf("MAE vs benchmark", format="%+.1f%%",
                                               help="How much smaller the average miss is than the benchmark's. "
                                                    "Positive = better than the benchmark."),
                     "RMSE improvement (%)": nf("RMSE vs benchmark", format="%+.1f%%",
                                                help="How much smaller the RMSE is than the benchmark's. Positive = better."),
                     "Beats benchmark?": st.column_config.TextColumn(
                         "Beats benchmark?", help=f"'Yes' only when the gain passes a Diebold–Mariano test at "
                                                  f"{sp['alpha']:.0%}, so it is unlikely to be luck.")})
    for n in res["notes"][:2]:
        st.caption(n)

    with st.expander("See how the forecast test works — method, tests and forecasts vs actual", icon=":material/table_view:"):
        st.markdown(f"**Method.** {res['method']}")
        if is_ts:
            st.markdown(f"**Driver timing.** {TIMING[res['timing']]}.")
        st.dataframe(t[["Model", "What it uses", "Out-of-sample R²", "p-value vs benchmark"]], hide_index=True,
                     width="stretch", column_config={
                         "Out-of-sample R²": nf(format="%.3f", help="1 − (model MSE ÷ benchmark MSE). Above 0 = better "
                                                                     "than the benchmark."),
                         "p-value vs benchmark": nf(format="%.4f", help="Diebold–Mariano test of equal accuracy "
                                                                        "(squared errors, small-sample correction).")})
        st.caption("The Diebold–Mariano test is conservative when one model contains the other (for example the "
                   "historical average inside a regression), so a borderline result may understate a real gain.")
        for n in res["notes"][2:]:
            st.caption(n)
        if is_ts:
            s = res["series"]
            x = ss.suit.dates.reindex(s.index) if ss.suit.dates is not None else s.index
            fig = go.Figure()
            fig.add_scatter(x=x, y=s["actual"], name="Actual", mode="lines", line=dict(color=LINE, width=2))
            best = res["best"]
            if best in s:
                fig.add_scatter(x=x, y=s[best], name=best, mode="lines", line=dict(color=ACCENT, width=1.6))
            fig.add_scatter(x=x, y=s[t.iloc[0]["Model"]], name=t.iloc[0]["Model"], mode="lines",
                            line=dict(color=POINT, width=1.2, dash="dot"))
            show(style_fig(fig, 320, title="Test period: forecasts vs actual"), "fc_chart")
