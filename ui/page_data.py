"""Step 1 — load data, review it, prepare it, choose variables, check suitability."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from core import market_data as md
from core.data_io import (FREQ_PERIODS_PER_YEAR, infer_frequency, list_sheets, looks_like_date, make_dummies,
                          parse_dates, profile_columns, read_table, try_numeric, unit_hint)
from core.demo_data import DEMO_B_DICTIONARY, make_private_credit
from core.insights import RETURN_UNITS, UNITS
from core.suitability import assess
from ui.common import LEVEL_ICON, go_to, log, nav_buttons, page_header, reset_models, ss

FREQS = ["daily", "weekly", "monthly", "quarterly", "annual"]
DEMO_B_UNITS = {"origination_year": "unitless / other", "sponsor_backed": "0/1 indicator", "ebitda_musd": "USD millions",
                "log_ebitda": "log value", "leverage_x": "multiple (x)", "interest_coverage_x": "multiple (x)",
                "ltv_pct": "percent (1 = 1%)", "spread_bps": "basis points", "default_flag": "0/1 indicator"}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _demo_a(tickers, model, freq, start, end):
    return md.build_demo_a(list(tickers), model, freq, start, end)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _rf(freq, source):
    return md.get_risk_free(freq, source)


def _set_data(df, name, kind, provenance, synthetic=False, units=None, spec_defaults=None, note=None):
    ss.data = {"df": df, "orig": df.copy(), "name": name, "kind": kind, "provenance": provenance,
               "synthetic": synthetic, "note": note, "transforms": []}
    ss.units = dict(units or {})
    ss.excess_cols = set()
    ss.spec = {}
    ss.suit = None
    ss.log = []
    ss.spec_defaults = spec_defaults or {}
    for k in ["w_y", "w_xs", "w_date", "w_id", "w_freq", "w_alpha", "w_xs_excess"]:
        ss.pop(k, None)
    reset_models()
    log("Data loaded", f"{name}: {len(df)} rows × {df.shape[1]} columns")


# ----------------------------------------------------------------------------- loaders
def _load_section():
    st.markdown("#### 1 · Load data")
    d = ss.data
    if d is not None:
        c1, c2 = st.columns([4, 1])
        tag = " · **synthetic**" if d["synthetic"] else ""
        c1.success(f"**{d['name']}** — {len(d['df']):,} rows × {d['df'].shape[1]} columns{tag}", icon=":material/check_circle:")
        if c2.button("Change data", icon=":material/swap_horiz:", width="stretch"):
            ss.data = None
            reset_models()
            st.rerun()
        if d.get("note"):
            st.caption(d["note"])
        return
    choice = st.segmented_control(
        "Choose a data source", ["Upload your file", "Demo A · ETF factor returns", "Demo B · Private credit loans"],
        default="Upload your file", key="src_choice")
    with st.container(border=True):
        if choice == "Upload your file":
            _upload()
        elif choice and choice.startswith("Demo A"):
            _demo_a_ui()
        elif choice:
            _demo_b_ui()


def _upload():
    st.warning("**Public demo — do not upload confidential data.** Anything uploaded here is processed on a shared "
               "public server. Use public, synthetic or approved data only.", icon=":material/shield:")
    st.markdown("One row per observation, one column per variable, column names in the first row. "
                "Optional date or ID column.")
    f = st.file_uploader("Excel (.xlsx) or CSV", type=["xlsx", "xlsm", "csv"], label_visibility="collapsed")
    if not f:
        return
    data = f.getvalue()
    sheet = None
    if f.name.lower().endswith((".xlsx", ".xlsm")):
        try:
            sheets = list_sheets(data)
        except Exception as e:
            st.error(f"Could not open this Excel file: {e}")
            return
        sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
    if st.button("Load file", type="primary", icon=":material/upload:"):
        try:
            df = read_table(data, f.name, sheet)
        except Exception as e:
            st.error(str(e))
            return
        if df.empty:
            st.error("The file has no data rows.")
            return
        prov = [("Source", "Analyst upload"), ("File", f.name + (f" · sheet '{sheet}'" if sheet else "")),
                ("Loaded", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), ("Rows × columns", f"{len(df)} × {df.shape[1]}")]
        _set_data(df, f.name, "upload", prov)
        st.rerun()


def _demo_a_ui():
    st.markdown("Monthly (or daily / weekly / quarterly) total returns for ETFs from **Yahoo Finance**, with "
                "**Fama–French** factors and the **risk-free rate** from the Kenneth R. French Data Library.")
    c1, c2 = st.columns(2)
    tickers = c1.multiselect("ETFs", options=md.DEFAULT_TICKERS, default=md.DEFAULT_TICKERS, accept_new_options=True,
                             help="Type any Yahoo ticker to add it.", key="demoA_tickers")
    model = c2.selectbox("Factor model", list(md.FACTOR_MODELS), index=3, key="demoA_model")
    c3, c4, c5 = st.columns(3)
    freq = c3.selectbox("Frequency", ["monthly", "weekly", "daily", "quarterly"], key="demoA_freq")
    today = date.today()
    start = c4.date_input("Start", date(today.year - 10, today.month, 1), min_value=date(1993, 1, 1), key="demoA_start")
    end = c5.date_input("End", today, key="demoA_end")
    with st.expander("About these funds and factors"):
        st.markdown("\n".join(f"- **{t}** — {n}" for t, n in md.TICKER_NOTES.items()))
        st.markdown("\n".join(f"- **{k}** — {v}" for k, v in md.FACTOR_DESCRIPTIONS.items()))
    if st.button("Load Demo A", type="primary", icon=":material/download:", disabled=not tickers):
        with st.spinner("Fetching prices, factors and the risk-free rate…"):
            try:
                df, prov = _demo_a(tuple(tickers), model, freq, str(start), str(end))
            except Exception as e:
                st.error(f"Could not load market data: {e}")
                return
        if prov["missing_tickers"]:
            st.warning(f"No data for: {', '.join(prov['missing_tickers'])}.")
        etfs = [t for t in tickers if t in df.columns]
        rows = [("Source — prices", prov["prices"]["source"]), ("Prices retrieved", prov["prices"]["retrieved"])]
        for p in prov["factors"]:
            rows.append(("Source — factors", f"{p['source']} · {p['file']}"))
            rows.append(("Factors retrieved", p["retrieved"]))
        rows += [("Frequency", freq), ("Factor model", model), ("ETFs", ", ".join(etfs)),
                 ("Sample window", f"{df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}"),
                 ("Licensing", "Ken French data: free for research/education, cite the library. Yahoo: personal / non-commercial use.")]
        rows += [("Transformation", t) for t in prov["transformations"]]
        units = {c: "decimal return (0.01 = 1%)" for c in df.columns if c != "Date"}
        # convert ETF returns to excess returns using the matched RF column
        for t in etfs:
            df[f"{t}_excess"] = df[t] - df["RF"]
        excess = [f"{t}_excess" for t in etfs]
        for c in excess:
            units[c] = "decimal return (0.01 = 1%)"
        factors = md.FACTOR_MODELS[model]
        note = ("Live data" if prov["prices"]["mode"] == "live" else "Saved snapshot (live source unreachable)") + \
               f" · ETF excess returns created as return − RF for {', '.join(etfs)}."
        _set_data(df, f"Demo A — ETF factor returns ({freq})", "demo_a", rows, units=units, note=note,
                  spec_defaults={"y": excess[0] if excess else None, "xs": factors, "date": "Date", "freq": freq})
        ss.excess_cols = set(excess) | set(factors) | {"Mkt-RF"}
        ss.data["transforms"].append(f"Excess returns: {', '.join(excess)} = ETF return − RF (Ken French 1-month T-bill, same period)")
        log("Excess returns created", ", ".join(excess))
        st.rerun()


def _demo_b_ui():
    st.markdown("A **synthetic** direct-lending loan tape: 300 fabricated loans with spreads, leverage, coverage, size, "
                "LTV, seniority, sponsor and sector. **No real borrower or portfolio is represented.** "
                "It contains deliberately planted problems for ELK to detect.")
    with st.expander("Variables"):
        st.table(pd.DataFrame({"column": DEMO_B_DICTIONARY.keys(), "meaning": DEMO_B_DICTIONARY.values()}).set_index("column"))
    if st.button("Load Demo B", type="primary", icon=":material/download:"):
        df = make_private_credit()
        prov = [("Source", "SYNTHETIC — generated by ELK (core/demo_data.py, fixed random seed 7)"),
                ("Represents", "Fabricated direct-lending loans; not any real borrower, lender or portfolio"),
                ("Rows × columns", f"{len(df)} × {df.shape[1]}"),
                ("Planted features", "Leverage–coverage collinearity; variance falling with borrower size; "
                                     "a few distressed influential loans; binary default flag; a few missing values")]
        _set_data(df, "Demo B — Private credit loans (SYNTHETIC)", "demo_b", prov, synthetic=True, units=DEMO_B_UNITS,
                  spec_defaults={"y": "spread_bps", "xs": ["leverage_x", "interest_coverage_x", "log_ebitda", "ltv_pct",
                                                          "sponsor_backed"], "id": "loan_id", "dummies": ["seniority"]},
                  note="Tip: convert 'seniority' to dummy variables in step 3 to include it in the model.")
        st.rerun()


# ----------------------------------------------------------------------------- review
def _review_section(df):
    st.markdown("#### 2 · Review the data")
    prof = profile_columns(df)
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Cells missing", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", int(df.duplicated().sum()))
    t1, t2 = st.tabs(["Column profile", "Data preview"])
    with t1:
        st.dataframe(prof[["name", "kind", "missing", "missing_pct", "unique", "example", "note"]], hide_index=True,
                     width="stretch", column_config={"name": "Column", "kind": "Detected type", "missing": "Missing",
                                                     "missing_pct": st.column_config.NumberColumn("Missing %", format="%.1f"),
                                                     "unique": "Distinct", "example": "Example", "note": "Note"})
    with t2:
        st.dataframe(df.head(500), width="stretch", height=300)
    return prof


# ----------------------------------------------------------------------------- prepare
def _prepare_section(df, prof):
    st.markdown("#### 3 · Prepare variables")
    numeric = prof[prof["kind"].isin(["numeric", "text-numeric"])]["name"].tolist()
    done_dummies = ss.data.setdefault("dummied", [])
    cats = [c for c in prof[prof["kind"].isin(["categorical"])]["name"].tolist() if c not in done_dummies]
    dates = prof[prof["kind"] == "date"]["name"].tolist()

    with st.expander("Units — label what each number measures", expanded=ss.data["kind"] == "upload", icon=":material/straighten:"):
        st.caption("ELK never guesses units. The hint describes the scale only; please confirm. Units drive how results are worded.")
        rows = [{"column": c, "unit": ss.units.get(c, "unitless / other"), "hint": unit_hint(df[c])} for c in numeric]
        ed = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["column", "hint"],
                            key=f"units_editor_{ss.data['name']}",
                            column_config={"unit": st.column_config.SelectboxColumn("Unit", options=UNITS, required=True),
                                           "column": "Column", "hint": "Scale hint"})
        new_units = dict(zip(ed["column"], ed["unit"]))
        if new_units != {c: ss.units.get(c, "unitless / other") for c in numeric}:
            ss.units.update(new_units)
            reset_models()

    if cats:
        with st.expander("Categorical columns → dummy variables", icon=":material/category:",
                         expanded=bool(ss.get("spec_defaults", {}).get("dummies"))):
            st.caption("OLS needs numbers. Each category becomes a 0/1 column; one category is left out as the reference "
                       "(coefficients are measured relative to it).")
            c1, c2, c3 = st.columns([2, 2, 1])
            want = [c for c in ss.get("spec_defaults", {}).get("dummies", []) if c in cats]
            if want and ss.get("dum_col") not in cats:
                ss.dum_col = want[0]
            col = c1.selectbox("Column", cats, key="dum_col")
            counts = df[col].value_counts()
            ref = c2.selectbox("Reference category", counts.index.astype(str).tolist(), key=f"dum_ref_{col}",
                               help="Default: the most common category.")
            c3.write("")
            if c3.button("Create", key="dum_go", width="stretch"):
                new, made = make_dummies(df, col, ref)
                ss.data["df"] = new
                done_dummies.append(col)
                for m in made:
                    ss.units[m] = "0/1 indicator"
                ss.w_xs = list(ss.get("w_xs", [])) + made
                ss.pop("dum_col", None)
                st.toast(f"Added {', '.join(made)} to the explanatory variables.", icon=":material/check:")
                ss.data["transforms"].append(f"Dummy variables from '{col}' (reference: {ref}): {', '.join(made)}")
                log("Dummy variables created", f"{col} → {', '.join(made)} (reference {ref})")
                reset_models()
                st.rerun()

    ret_cols = [c for c in numeric if ss.units.get(c) in RETURN_UNITS]
    with st.expander("Excess returns — subtract the risk-free rate", icon=":material/trending_up:"):
        if not ret_cols:
            st.caption("Mark one or more columns as returns in **Units** to enable this.")
        elif not dates:
            st.caption("A date column is needed to match each period's risk-free rate.")
        else:
            st.caption("Excess return = return − risk-free rate for the same period. Needed before interpreting an intercept as "
                       "alpha. Ken French factors (Mkt-RF, SMB, HML, RMW, CMA, Mom) are already excess / long–short and must "
                       "not be adjusted again.")
            todo = st.multiselect("Return columns to convert", [c for c in ret_cols if not c.endswith("_excess")], key="ex_cols")
            c1, c2, c3 = st.columns(3)
            dcol = c1.selectbox("Date column", dates, key="ex_date")
            in_data = [c for c in numeric if c.upper() in ("RF", "RISK_FREE", "RISKFREE", "TBILL")]
            sources = ([f"Column in data: {c}" for c in in_data] +
                       ["Fetch: Ken French (1-month T-bill)", "Fetch: FRED (3-month T-bill, DTB3)"])
            src = c2.selectbox("Risk-free source", sources, key="ex_src")
            fq_guess, _ = infer_frequency(parse_dates(df[dcol]))
            fq = c3.selectbox("Data frequency", FREQS, index=FREQS.index(fq_guess) if fq_guess in FREQS else 2, key="ex_freq")
            if st.button("Create excess-return columns", disabled=not todo):
                try:
                    if src.startswith("Column"):
                        rf = pd.to_numeric(df[src.split(": ", 1)[1]], errors="coerce")
                        rf_unit = ss.units.get(src.split(": ", 1)[1])
                        how = f"column '{src.split(': ', 1)[1]}'"
                    else:
                        with st.spinner("Fetching risk-free rate…"):
                            series, prov = _rf(fq, "Ken French" if "French" in src else "FRED")
                        rf = md.attach_rf(parse_dates(df[dcol]), series, fq)
                        rf_unit = "decimal return (0.01 = 1%)"
                        how = f"{prov['source']} · {prov.get('series', '')} · retrieved {prov['retrieved']}"
                        ss.data["provenance"].append(("Risk-free rate", how))
                    new = df.copy()
                    unmatched = int(rf.isna().sum())
                    for c in todo:
                        r = rf * (100 if ss.units.get(c) == "percent (1 = 1%)" and rf_unit != "percent (1 = 1%)" else 1)
                        new[f"{c}_excess"] = pd.to_numeric(new[c], errors="coerce") - r
                        ss.units[f"{c}_excess"] = ss.units.get(c)
                        ss.excess_cols.add(f"{c}_excess")
                    ss.data["df"] = new
                    ss.data["transforms"].append(f"Excess returns for {', '.join(todo)} using RF from {how} ({fq})")
                    log("Excess returns created", f"{', '.join(todo)} − RF ({how})")
                    if unmatched:
                        st.warning(f"{unmatched} row(s) had no matching risk-free value (outside the RF series' dates); "
                                   "their excess returns are missing.")
                    reset_models()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not create excess returns: {e}")


# ----------------------------------------------------------------------------- variables
def _variables_section(df, prof):
    st.markdown("#### 4 · Choose the model")
    numeric = prof[prof["kind"].isin(["numeric", "text-numeric"])]["name"].tolist()
    dates = prof[prof["kind"] == "date"]["name"].tolist()
    ids = prof[prof["kind"].isin(["id-like", "categorical"])]["name"].tolist()
    dflt = ss.get("spec_defaults", {})

    if "w_y" not in ss or ss.w_y not in numeric:
        ss.w_y = dflt.get("y") if dflt.get("y") in numeric else (numeric[0] if numeric else None)
    if "w_xs" not in ss:
        ss.w_xs = [x for x in dflt.get("xs", []) if x in numeric and x != ss.w_y]
    ss.w_xs = [x for x in ss.w_xs if x in numeric]
    if "w_date" not in ss:
        ss.w_date = dflt.get("date") if dflt.get("date") in dates else (dates[0] if dates else "(none)")
    if "w_id" not in ss:
        ss.w_id = dflt.get("id") if dflt.get("id") in ids else "(none)"

    c1, c2 = st.columns([1, 2])
    y = c1.selectbox("Outcome (Y)", numeric, key="w_y", help="The variable you want to explain.")
    xs = c2.multiselect("Explanatory variables (X)", [c for c in numeric if c != y], key="w_xs",
                        help="One or more numeric variables. Categorical columns must be converted to dummies first.")
    c3, c4, c5, c6 = st.columns(4)
    date_col = c3.selectbox("Date column", ["(none)"] + dates, key="w_date",
                            help="Selecting a date makes this a time series: rows are sorted chronologically and "
                                 "time-series tests become available.")
    id_col = c4.selectbox("Row label / ID (optional)", ["(none)"] + ids, key="w_id")
    date_col = None if date_col == "(none)" else date_col
    id_col = None if id_col == "(none)" else id_col
    freq = None
    if date_col:
        g, _ = infer_frequency(parse_dates(df[date_col]))
        if "w_freq" not in ss:
            ss.w_freq = dflt.get("freq") or (g if g in FREQS else "monthly")
        freq = c5.selectbox("Frequency", FREQS, key="w_freq", help=f"Detected from dates: {g}.")
    else:
        c5.selectbox("Frequency", ["cross-section"], disabled=True, key="w_freq_cs")
    alpha = c6.segmented_control("Significance level", [0.01, 0.05, 0.10], default=0.05, format_func=lambda v: f"{v:.0%}",
                                 key="w_alpha") or 0.05

    # alpha eligibility
    alpha_ok, reason = False, ""
    if ss.units.get(y) in RETURN_UNITS and date_col:
        y_excess = y in ss.excess_cols
        if not y_excess:
            y_excess = st.checkbox(f"'{y}' is already an excess return (return minus risk-free)", key=f"yex_{y}")
            if y_excess:
                ss.excess_cols.add(y)
        x_ok = all(x in ss.excess_cols for x in xs) if xs else False
        if xs and not x_ok:
            x_ok = st.checkbox("All explanatory variables are excess or long–short factor returns", key="w_xs_excess")
        alpha_ok = bool(y_excess and x_ok)
        if not y_excess:
            reason = f"'{y}' is not marked as an excess return. Convert it in step 3 or tick the box above if it already is."
        elif not x_ok:
            reason = "the explanatory variables are not all confirmed as excess or long–short factor returns."
    elif ss.units.get(y) in RETURN_UNITS:
        reason = "alpha needs a time series of returns (select a date column)."
    else:
        reason = "the outcome is not a return series, so the intercept has no alpha interpretation."
    if alpha_ok:
        st.caption(":material/check: The intercept will be reported as **alpha**: excess-return outcome, excess / long–short "
                   "factors, time-ordered data.")

    spec = {"y": y, "xs": list(xs), "date_col": date_col, "id_col": id_col, "freq": freq, "alpha": float(alpha),
            "alpha_ok": alpha_ok, "alpha_reason": reason, "is_ts": bool(date_col),
            "ppy": FREQ_PERIODS_PER_YEAR.get(freq) if freq else None}
    if spec != ss.spec:
        changed = {k for k in spec if spec.get(k) != ss.spec.get(k)}
        if ss.spec and changed - {"alpha"}:
            reset_models()
        ss.spec = spec
    return spec


# ----------------------------------------------------------------------------- suitability
def _suitability_section(df, spec):
    st.markdown("#### 5 · Data suitability")
    if not spec["y"] or not spec["xs"]:
        st.info("Choose an outcome and at least one explanatory variable to run the suitability check.", icon=":material/info:")
        return None
    suit = assess(df, spec["y"], spec["xs"], spec["date_col"], spec["id_col"], spec["freq"])
    ss.suit = suit
    m = suit.metrics
    c = st.columns(5)
    c[0].metric("Rows supplied", f"{m.get('n_raw', 0):,}")
    c[1].metric("Rows usable", f"{m.get('n_complete', 0):,}",
                help=f"{m.get('n_dropped_missing', 0)} row(s) excluded for missing values")
    c[2].metric("Parameters", m.get("k_params", "—"))
    c[3].metric("Obs. per parameter", f"{m.get('obs_per_param', 0):.1f}" if m.get("obs_per_param") else "—")
    c[4].metric("Structure", "Time series" if spec["is_ts"] else "Cross-section",
                help=f"Detected frequency: {m.get('detected_frequency')}" if spec["is_ts"] else "No date column selected")
    if spec["is_ts"] and m.get("start") is not None:
        st.caption(f"Sample window: {m['start']:%Y-%m-%d} to {m['end']:%Y-%m-%d}")
    order = {"block": 0, "warn": 1, "info": 2}
    for i in sorted(suit.issues, key=lambda i: order[i.level]):
        fn = {"block": st.error, "warn": st.warning, "info": st.info}[i.level]
        fn(f"**{i.title}.** {i.detail}", icon=LEVEL_ICON[i.level])
    if "variability" in m:
        with st.expander("Summary statistics of modelled variables"):
            st.dataframe(m["variability"], hide_index=True, width="stretch")
    return suit


def render():
    page_header("Data & Suitability", "Load a dataset, check it, choose the outcome and explanatory variables, "
                                      "and review whether the data can support the analysis.")
    _load_section()
    if ss.data is None:
        return
    df = ss.data["df"]
    prof = _review_section(df)
    _prepare_section(df, prof)
    df = ss.data["df"]
    prof = profile_columns(df)
    spec = _variables_section(df, prof)
    suit = _suitability_section(df, spec)
    blocked = suit is None or suit.blocked
    nav_buttons(None, 1, "Run regression", disabled=blocked,
                help="Resolve the blocking issues above first." if (suit is not None and suit.blocked) else None)
