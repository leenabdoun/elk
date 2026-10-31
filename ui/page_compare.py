"""Step 4 — Model Comparison & Repair: build a Model B, compare side by side."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.insights import fmt, fmt_p, p_eq, suggest_remedies
from core.models import COV_LABELS, ModelSpec, compare, fit_ols, newey_west_lags, out_of_sample, summarize_change
from ui.analysis import get_diag, get_influence, get_vif
from ui.common import intercept_label, log, nav_buttons, need_model, page_header, ss


DRAFT = "[Draft — edit to your own reasoning] "


def _preset(rid, fit, remedy):
    ss.b_autorun = True  # estimate straight away; the form below stays editable
    ss.b_xs = list(fit.spec.xs)
    ss.b_excl = False
    ss.b_cov = "nonrobust"
    if not ss.get("b_just") or ss.b_just.startswith(DRAFT):
        ss.b_just = ""
    if rid == "HC3":
        ss.b_cov = "HC3"
    elif rid == "HAC":
        ss.b_cov = "HAC"
    elif rid == "DROP":
        cand = remedy.get("candidates", [])
        # suggest dropping the collinear variable with the larger p-value
        hi = [c for c in cand if c in fit.coef.index][:2]
        if hi:
            drop = max(hi, key=lambda c: fit.coef.loc[c, "p_value"])
            keep = [c for c in hi if c != drop]
            ss.b_xs = [x for x in fit.spec.xs if x != drop]
            if not ss.b_just:
                ss.b_just = (DRAFT + f"{drop} is highly collinear with {', '.join(keep) or 'other regressors'} "
                             f"(high VIF), so both measure largely the same thing; {drop} is removed and "
                             f"{', '.join(keep) or 'the remaining variables'} kept as the primary measure.")
    elif rid == "EXCL":
        ss.b_excl = True
        if not ss.b_just:
            ss.b_just = (DRAFT + "Sensitivity check: re-estimate without the observations flagged as outliers or "
                         "influential to see whether conclusions depend on them. The full-sample model remains the reference.")


def render():
    page_header("Compare & Repair", "Build an alternative Model B — a robust inference adjustment, a revised set of "
                                    "variables, or a sensitivity check — and compare it with the original. Nothing is "
                                    "applied automatically.")
    if not need_model():
        return
    A = ss.fitA
    sp = ss.spec
    alpha = sp["alpha"]
    d = get_diag(A)
    vif = get_vif(A)
    _, flagged = get_influence(A)

    rem = suggest_remedies(d, vif, flagged, sp["is_ts"])
    st.markdown("##### Suggested options based on the diagnostics")
    if not rem:
        st.caption("No diagnostic flagged a problem. You can still build an alternative specification below.")
    cols = st.columns(max(1, min(len(rem), 4)))
    for i, r in enumerate(rem):
        with cols[i % len(cols)].container(border=True, height="stretch"):
            st.markdown(f"**{r['title']}**")
            st.caption(f"Why: {r['why']}")
            st.caption(f"Effect: {r['effect']}")
            st.button("Run as Model B", key=f"rem_{r['id']}", on_click=_preset, args=(r["id"], A, r), width="stretch",
                      icon=":material/play_arrow:")

    st.markdown("##### Model B")
    st.caption("Use a suggestion above, or set your own combination here (for example HC3 errors *and* a revised variable "
               "list), then click Estimate.")
    ss.setdefault("b_xs", list(A.spec.xs))
    ss.setdefault("b_cov", "nonrobust")
    ss.setdefault("b_excl", False)
    ss.b_xs = [x for x in ss.b_xs if x in ss.suit.frame.columns or x in ss.data["df"].columns]
    with st.container(border=True):
        c1, c2 = st.columns([1.2, 2])
        cov = c1.selectbox("Standard errors", list(COV_LABELS), format_func=COV_LABELS.get, key="b_cov",
                           help="Robust options change standard errors and p-values only — never the coefficients.")
        hac = None
        if cov == "HAC":
            if not sp["is_ts"]:
                c1.warning("HAC is designed for time-ordered data; with cross-sectional data use HC errors.")
            hac = c1.number_input("HAC lags", 1, 60, newey_west_lags(A.stats["n"]), key="b_hac",
                                  help="Default: Newey–West rule floor(4·(n/100)^(2/9)).")
        numeric = [c for c in ss.data["df"].columns if c != sp["y"] and pd.api.types.is_numeric_dtype(ss.data["df"][c])]
        xs = c2.multiselect("Explanatory variables", numeric, key="b_xs")
        excl = c2.checkbox(f"Sensitivity check: exclude the {len(flagged)} flagged observation(s)", key="b_excl",
                           disabled=len(flagged) == 0,
                           help="Estimates Model B without the flagged rows. The data and Model A are unchanged.")
        changes_spec = set(xs) != set(A.spec.xs) or excl
        ss.setdefault("b_just", "")
        just = st.text_area("Justification" + (" — required, because Model B changes the variables or observations"
                                               if changes_spec else " (optional)"),
                            key="b_just", height=90,
                            placeholder="e.g. Interest coverage is mechanically ~1/(leverage × cost of debt), so it duplicates "
                                        "the leverage information; leverage is kept as the primary credit metric.")
        if just.startswith(DRAFT):
            st.caption(":material/edit: A draft justification was filled in from the diagnostics. Edit it to reflect your "
                       "own reasoning — it is recorded in the report and audit trail.")
        clicked = st.button("Estimate Model B", type="primary", icon=":material/play_arrow:", disabled=not xs)
        clicked = bool(ss.pop("b_autorun", False)) or clicked
        if not xs:
            st.caption("Select at least one explanatory variable.")
        if clicked and changes_spec and not just.strip():
            st.error("Please write a short justification above. Changing variables or excluding observations changes the "
                     "coefficients, so ELK records why.", icon=":material/edit_note:")
        elif clicked:
            frame = ss.suit.frame
            if set(xs) - set(frame.columns):
                # variables outside Model A's suitability frame → rebuild a complete-case frame on the union
                from core.suitability import assess
                s2 = assess(ss.data["df"], sp["y"], list(dict.fromkeys(list(A.spec.xs) + list(xs))), sp["date_col"],
                            sp["id_col"], sp["freq"])
                if s2.blocked:
                    st.error("; ".join(i.title for i in s2.issues if i.level == "block"))
                    st.stop()
                frame = s2.frame
            spec = ModelSpec(sp["y"], list(xs), cov, hac, list(flagged.index) if excl else [], "Model B", just.strip())
            try:
                ss.fitB = fit_ols(frame, spec)
            except Exception as e:
                st.error(f"Model B could not be estimated: {e}")
                st.stop()
            ss.b_cfg = {"justification": just.strip()}
            ss.oos = None
            desc = f"{spec.describe()} · {COV_LABELS[cov]}" + (f" ({spec.hac_lags} lags)" if cov == "HAC" else "") + \
                   (f" · excluding {len(flagged)} flagged rows" if excl else "")
            log("Model B estimated", desc + (f" · justification: {just.strip()}" if just.strip() else ""))
            st.rerun()

    B = ss.fitB
    if B is None:
        nav_buttons(2, 4, "Insights & export")
        return

    if not A.frame.index.equals(B.frame.index) and set(A.spec.xs) - set(ss.suit.frame.columns) == set():
        pass
    st.markdown("##### A vs B")
    res = compare(A, B, alpha)
    cA, cB = st.columns(2)
    cA.markdown(f"**Model A** · {A.spec.describe()}  \n{COV_LABELS[A.spec.cov_type]} · n = {A.stats['n']}")
    cB.markdown(f"**Model B** · {B.spec.describe()}  \n{COV_LABELS[B.spec.cov_type]}"
                + (f" ({B.spec.hac_lags} lags)" if B.spec.cov_type == "HAC" else "") + f" · n = {B.stats['n']}"
                + (f" · {len(B.spec.exclude_rows)} rows excluded" if B.spec.exclude_rows else ""))
    for w in res["warnings"]:
        st.warning(w, icon=":material/warning:")
    chg = summarize_change(A, B, alpha)
    with st.container(border=True):
        st.markdown("**What changed from A to B**")
        st.markdown("\n".join(f"- {line}" for line in chg["lines"]))
    if res.get("nested_note") and not chg["inference_only"]:
        st.info(res["nested_note"], icon=":material/info:")

    ct = res["coef_table"].copy()
    ct.index = [intercept_label() if i == "const" else i for i in ct.index]
    for lab in ("A", "B"):
        ct[f"sig {lab}"] = np.where(ct[f"p {lab}"] < alpha, "✓", np.where(ct[f"p {lab}"].isna(), "", "·"))
    ct = ct[["coef A", "coef B", "coef change", "SE A", "SE B", "SE change", "p A", "sig A", "p B", "sig B"]]
    nf = st.column_config.NumberColumn
    st.dataframe(ct, width="stretch", column_config={
        **{c: nf(format="%.4f") for c in ["coef A", "coef B", "coef change", "SE A", "SE B", "SE change", "p A", "p B"]},
        "sig A": st.column_config.TextColumn(f"A sig. {alpha:.0%}"), "sig B": st.column_config.TextColumn(f"B sig. {alpha:.0%}")})
    if set(B.spec.xs) != set(A.spec.xs):
        st.caption("Removing or adding variables changes the other coefficients. A change is not automatically an "
                   "improvement — removing a relevant variable can bias the remaining estimates (omitted-variable bias).")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        m = res["metrics"].copy()
        for c in ["Model A", "Model B", "difference (B − A)"]:
            m[c] = [f"{v:,.0f}" if r in ("Observations", "Slope parameters") else
                    (f"{v:.4f}" if abs(v) < 10 else f"{v:,.2f}") for r, v in zip(m["metric"], m[c])]
        st.dataframe(m, hide_index=True, width="stretch")
    with c2:
        for t in res["tests"]:
            with st.container(border=True):
                st.markdown(f"**{t['name']}**  \nstatistic {fmt(t['stat'])} · df {t['df']} · **{p_eq(t['p_value'])}**")
                st.caption(f"H₀: {t['null']} {t['reading']} {t['caveat']}")
        if chg["inference_only"]:
            st.info("Residual diagnostics are identical for A and B: a standard-error adjustment doesn't change the "
                    "residuals. It changes how far the p-values can be trusted given the problems the diagnostics found. "
                    "To see diagnostics change, alter the variables or exclude flagged rows.", icon=":material/info:")
        dA, dB = get_diag(A), get_diag(B)
        rows = []
        for k in ["bp", "bg", "arch", "jb", "reset"]:
            if k in dA and k in dB:
                rows.append({"test": dA[k]["name"], "p (A)": dA[k]["p"], "p (B)": dB[k]["p"]})
        vb = get_vif(B)
        if len(vif) or len(vb):
            rows.append({"test": "Max VIF (value)", "p (A)": vif["VIF"].max() if len(vif) else np.nan,
                         "p (B)": vb["VIF"].max() if len(vb) else np.nan})
        if not chg["inference_only"]:
            st.markdown("**Diagnostics, A vs B** (p-values; VIF is a value)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"p (A)": nf(format="%.4f"), "p (B)": nf(format="%.4f")})

    st.markdown("##### Out-of-sample check")
    with st.container(border=True):
        if sp["is_ts"]:
            hold = st.slider("Holdout share (most recent observations)", 0.1, 0.4, 0.2, 0.05, key="oos_hold")
        else:
            hold = 0.2
            st.caption("Cross-sectional data: 5-fold cross-validation with a fixed seed.")
        if st.button("Run out-of-sample comparison", icon=":material/science:"):
            same = A.frame.index.intersection(B.frame.index)
            frame = ss.suit.frame.loc[ss.suit.frame.index.intersection(same)] if not B.spec.exclude_rows else A.frame.loc[same]
            if set(B.spec.xs) - set(frame.columns):
                frame = B.frame.loc[B.frame.index.intersection(same)]
            ss.oos = out_of_sample(frame, [A.spec, B.spec], sp["is_ts"], hold)
            log("Out-of-sample comparison", ss.oos["method"])
        if ss.oos:
            st.caption(ss.oos["method"])
            st.dataframe(ss.oos["table"], hide_index=True, width="stretch",
                         column_config={"RMSE": nf(format="%.5g"), "MAE": nf(format="%.5g")})
            if A.spec.xs == B.spec.xs:
                st.caption("Identical variables give identical predictions — robust errors do not change forecasts.")
            st.caption("Lower error is better. In-sample fit does not guarantee forecasting performance; a single holdout is "
                       "one draw of history.")

    st.markdown("##### Which model goes in the report?")
    ss.report_model = st.radio("Primary model for Insights & Export", ["A", "B"], horizontal=True,
                               format_func=lambda v: "Model A (original)" if v == "A" else "Model B (alternative)",
                               index=0 if ss.report_model == "A" else 1, label_visibility="collapsed")
    st.caption("Both models and the comparison are always included in the report; this chooses which one the narrative describes.")
    nav_buttons(2, 4, "Insights & export")
