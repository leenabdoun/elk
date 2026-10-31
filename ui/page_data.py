"""Step 1 — Data & Setup: load → check what ELK found → choose Y and X → readiness → Run analysis.
Everything technical lives in collapsed 'Advanced' sections."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from core import market_data as md
from core.context import guess_unit
from core.data_io import FREQ_PERIODS_PER_YEAR, infer_frequency, make_dummies, parse_dates, profile_columns
from core.demo_data import DEMO_B_DICTIONARY, make_private_credit
from core.importer import import_file, read_grids
from core.insights import RETURN_UNITS, UNITS
from core.suitability import assess
from ui.common import LEVEL_ICON, go_to, log, page_header, reset_models, ss

FREQS = ["daily", "weekly", "monthly", "quarterly", "annual"]
DEMO_B_UNITS = {"origination_year": "unitless / other", "sponsor_backed": "0/1 indicator", "ebitda_musd": "USD millions",
                "log_ebitda": "log value", "leverage_x": "multiple (x)", "interest_coverage_x": "multiple (x)",
                "ltv_pct": "percent (1 = 1%)", "spread_bps": "basis points", "default_flag": "0/1 indicator"}
INTRO = [
    (":material/functions:", "Regression", "Measures how an outcome (a fund's return, a loan's spread, a property's NOI) "
     "moves with the drivers you choose.", "Useful for exposures, pricing drivers, operating relationships."),
    (":material/troubleshoot:", "Model checks", "Tests whether the regression's conclusions can be trusted — "
     "e.g. whether a few data points drive everything, or two drivers measure the same thing.",
     "Useful before relying on a number in an investment memo."),
    (":material/compare_arrows:", "Comparison", "Puts two versions of the model side by side to see which explains "
     "the data better and whether conclusions change.", "Useful when choosing between factor sets or drivers."),
    (":material/tune:", "Adjustments", "Re-checks the confidence in results using corrections designed for messy "
     "real-world data, without changing the data itself.", "Useful when a model check raises a warning."),
]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _demo_a(tickers, model, freq, start, end):
    return md.build_demo_a(list(tickers), model, freq, start, end)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _rf(freq, source):
    return md.get_risk_free(freq, source)


def _set_data(df, name, kind, provenance, synthetic=False, units=None, spec_defaults=None, note=None,
              imp=None, file=None, descriptions=None):
    ss.data = {"df": df, "orig": df.copy(), "name": name, "kind": kind, "provenance": provenance,
               "synthetic": synthetic, "note": note, "transforms": [], "imp": imp, "file": file,
               "confirmed": not (imp is not None and imp.needs_confirmation), "dummied": []}
    ss.units = dict(units or {})
    ss.units_confirmed = set(units or {})
    ss.descriptions = dict(descriptions or {})
    ss.excess_cols = set()
    ss.spec = {}
    ss.suit = None
    ss.log = []
    ss.analysis_ran = False
    ss.spec_ready = False
    ss.spec_defaults = spec_defaults or {}
    for k in list(ss.keys()):
        if str(k).startswith(("w_", "units_tbl", "miss_ok", "yex_")):
            ss.pop(k, None)
    reset_models()
    log("Data loaded", f"{name}: {len(df)} rows × {df.shape[1]} columns")


# ----------------------------------------------------------------------------- 1. load
def _intro():
    st.markdown("###### What ELK does")
    cols = st.columns(4)
    for c, (icon, title, what, when) in zip(cols, INTRO):
        with c.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.caption(what)
            st.caption(f"*{when}*")
    st.caption("ELK reports statistical evidence to support an investment view. It does not make the decision, and a "
               "relationship in the data is not proof of cause and effect.")


def _load():
    choice = st.segmented_control(
        "Start with", ["Upload your file", "Demo A · ETF factor returns", "Demo B · Private credit loans"],
        default="Upload your file", key="src_choice")
    with st.container(border=True):
        if choice == "Upload your file":
            _upload()
        elif choice and choice.startswith("Demo A"):
            _demo_a_ui()
        elif choice:
            _demo_b_ui()


def _upload():
    st.warning("**Public demo — do not upload confidential data.** This hosted version runs on a shared public server "
               "and is not approved for private fund, client or company information.", icon=":material/shield:")
    st.markdown("Upload an ordinary Excel or CSV file — no template needed. ELK finds the table, the header row and any "
                "dates, and shows you what it found before anything is calculated.")
    f = st.file_uploader("Excel (.xlsx) or CSV", type=["xlsx", "xlsm", "csv"], label_visibility="collapsed")
    if f is None:
        return
    data = f.getvalue()
    with st.spinner("Reading the file…"):
        try:
            imp = import_file(data, f.name)
        except Exception as e:
            st.error(f"{e}", icon=":material/error:")
            return
    _load_import(imp, data, f.name)
    st.rerun()


def _load_import(imp, data, filename, overrides=None):
    prov = [("Source", "Analyst upload"), ("File", filename), ("Table", imp.location),
            ("Loaded", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
            ("Rows × columns", f"{len(imp.df)} × {imp.df.shape[1]}")]
    units = {c: guess_unit(c, imp.df[c]) for c in imp.df.columns if pd.api.types.is_numeric_dtype(imp.df[c])}
    _set_data(imp.df, filename, "upload", prov, units=units, imp=imp, file={"bytes": data, "name": filename},
              note=None)
    ss.units_confirmed = set()  # guesses only
    if overrides:
        ss.data["confirmed"] = True
        log("Import adjusted", f"{imp.location} (analyst override)")


def _demo_a_ui():
    st.markdown("Total returns for a set of ETFs (Yahoo Finance) with **Fama–French** factors and the **risk-free rate** "
                "(Kenneth R. French Data Library). The question: *what drives each fund's returns, and is there "
                "evidence of outperformance (alpha) after accounting for those drivers?*")
    c1, c2 = st.columns(2)
    tickers = c1.multiselect("ETFs", options=md.DEFAULT_TICKERS, default=md.DEFAULT_TICKERS, accept_new_options=True,
                             help="Type any Yahoo ticker to add it.", key="demoA_tickers")
    model = c2.selectbox("Factor model", list(md.FACTOR_MODELS), index=3, key="demoA_model",
                         help="Which market drivers to include. Fama–French 5 is a common default for US equities.")
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
        df = df.copy()
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
        for t in etfs:
            df[f"{t}_excess"] = df[t] - df["RF"]
        excess = [f"{t}_excess" for t in etfs]
        for c in excess:
            units[c] = "decimal return (0.01 = 1%)"
        factors = md.FACTOR_MODELS[model]
        desc = dict(md.FACTOR_SHORT)
        desc.update({f"{t}_excess": f"{t} return above cash" for t in etfs})
        desc.update({t: f"{t} total return" for t in etfs})
        note = ("Live data" if prov["prices"]["mode"] == "live" else "Saved snapshot (live source unreachable)") + \
               f" · Excess returns created as fund return − risk-free rate for {', '.join(etfs)}."
        _set_data(df, f"Demo A — ETF factor returns ({freq})", "demo_a", rows, units=units, note=note,
                  spec_defaults={"y": excess[0] if excess else None, "xs": factors, "date": "Date", "freq": freq,
                                 "question": "What drives this fund's returns, and does it outperform after accounting "
                                             "for market factors?"},
                  descriptions=desc)
        ss.excess_cols = set(excess) | set(factors) | {"Mkt-RF"}
        ss.data["transforms"].append(f"Excess returns: {', '.join(excess)} = ETF return − RF (Ken French 1-month T-bill, same period)")
        log("Excess returns created", ", ".join(excess))
        st.rerun()


def _demo_b_ui():
    st.markdown("A **synthetic** direct-lending loan tape: 300 fabricated loans. The question: *which borrower and loan "
                "characteristics explain the spread a lender charges?* **No real borrower or portfolio is represented**, "
                "and a few realistic problems are planted for ELK to detect.")
    with st.expander("Variables"):
        st.table(pd.DataFrame({"column": DEMO_B_DICTIONARY.keys(), "meaning": DEMO_B_DICTIONARY.values()}).set_index("column"))
    if st.button("Load Demo B", type="primary", icon=":material/download:"):
        df = make_private_credit()
        df, made = make_dummies(df, "seniority", "First Lien")
        prov = [("Source", "SYNTHETIC — generated by ELK (core/demo_data.py, fixed random seed 7)"),
                ("Represents", "Fabricated direct-lending loans; not any real borrower, lender or portfolio"),
                ("Rows × columns", f"{len(df)} × {df.shape[1]}"),
                ("Planted features", "Leverage–coverage overlap; noisier pricing for small borrowers; "
                                     "a few distressed influential loans; binary default flag; a few missing values")]
        units = dict(DEMO_B_UNITS, **{m: "0/1 indicator" for m in made})
        desc = {"spread_bps": "spread", "leverage_x": "leverage (debt/EBITDA)", "interest_coverage_x": "interest coverage",
                "log_ebitda": "borrower EBITDA", "ebitda_musd": "borrower EBITDA ($m)", "ltv_pct": "loan-to-value",
                "sponsor_backed": "sponsor backing", "seniority_Unitranche": "unitranche (vs first lien)",
                "seniority_Second_Lien": "second lien (vs first lien)", "origination_year": "origination year",
                "default_flag": "default within 24 months"}
        _set_data(df, "Demo B — Private credit loans (SYNTHETIC)", "demo_b", prov, synthetic=True, units=units,
                  descriptions=desc,
                  spec_defaults={"y": "spread_bps", "xs": ["leverage_x", "interest_coverage_x", "log_ebitda", "ltv_pct",
                                                          "sponsor_backed"] + made, "id": "loan_id",
                                 "question": "Which borrower and loan characteristics explain the spread charged?"},
                  note="Seniority was converted to 0/1 columns (first lien is the reference).")
        ss.data["dummied"] = ["seniority"]
        ss.data["transforms"].append(f"Dummy variables from 'seniority' (reference: First Lien): {', '.join(made)}")
        st.rerun()


# ----------------------------------------------------------------------------- 1b. what ELK found
def _found(df):
    d = ss.data
    c1, c2 = st.columns([4, 1])
    tag = " · **synthetic**" if d["synthetic"] else ""
    c1.success(f"**{d['name']}** — {len(df):,} rows × {df.shape[1]} columns{tag}", icon=":material/check_circle:")
    if c2.button("Change data", icon=":material/swap_horiz:", width="stretch"):
        ss.data = None
        reset_models()
        st.rerun()
    imp = d.get("imp")
    if imp is not None:
        st.markdown(f":material/location_on: Found on **{imp.location}**.")
        for n in imp.notes:
            st.caption(f"• {n}")
        if not d["confirmed"]:
            with st.container(border=True):
                st.markdown(f":material/help: **Please check:** {imp.question}")
                usable = [s["sheet"] for s in imp.sheets if s.get("usable")]
                c1, c2 = st.columns([2, 1])
                if len(usable) > 1:
                    pick = c1.selectbox("Table to use", usable, index=usable.index(imp.sheet) if imp.sheet in usable else 0,
                                        format_func=lambda s: f"Sheet '{s}'", key="pick_sheet")
                else:
                    pick = imp.sheet
                    c1.caption("If the preview below looks wrong, set the header row in **Advanced import**.")
                c2.write("")
                if c2.button("Use this table", type="primary", icon=":material/check:", width="stretch"):
                    if pick != imp.sheet:
                        new = import_file(d["file"]["bytes"], d["file"]["name"], sheet=pick)
                        _load_import(new, d["file"]["bytes"], d["file"]["name"], overrides=True)
                    ss.data["confirmed"] = True
                    log("Table confirmed", ss.data["imp"].location)
                    st.rerun()
    elif d.get("note"):
        st.caption(d["note"])

    prof = profile_columns(df)
    kinds = prof["kind"].value_counts()
    summary = ", ".join(f"{n} {k.replace('text-numeric', 'numbers stored as text')}" for k, n in kinds.items())
    tabs = ["Preview"] + (["Original file"] if imp is not None else []) + ["Column details"]
    t = st.tabs(tabs)
    with t[0]:
        st.dataframe(df.head(300), width="stretch", height=260)
        st.caption(f"{summary}. Showing the first {min(300, len(df))} rows.")
    if imp is not None:
        with t[1]:
            grid = read_grids(d["file"]["bytes"], d["file"]["name"])[imp.sheet or ""]
            show = grid.head(60).copy()
            show.index = show.index + 1
            show.columns = [chr(65 + i) if i < 26 else f"C{i + 1}" for i in range(show.shape[1])]
            st.dataframe(show, width="stretch", height=260)
            st.caption("The file exactly as stored (first 60 rows, Excel row numbers). ELK never edits it; any changes are "
                       "made to the analysis table and listed in the report.")
    with t[-1]:
        st.dataframe(prof[["name", "kind", "missing", "unique", "example", "note"]], hide_index=True, width="stretch",
                     column_config={"name": "Column", "kind": "Detected type", "missing": "Missing", "unique": "Distinct",
                                    "example": "Example", "note": "Note"})
    if imp is not None:
        _advanced_import(imp)
    return prof


def _advanced_import(imp):
    with st.expander("Advanced import — choose sheet, header row, rows or orientation", icon=":material/settings:"):
        d = ss.data
        sheets = [s["sheet"] for s in imp.sheets] or [imp.sheet]
        c1, c2, c3, c4 = st.columns(4)
        sh = c1.selectbox("Sheet", sheets, index=sheets.index(imp.sheet) if imp.sheet in sheets else 0,
                          format_func=lambda s: s or "(CSV)", key="adv_sheet")
        hdr = c2.number_input("Header row", 0, 10000, int(imp.header_row or 0), help="0 = no header row", key="adv_hdr")
        r1 = c3.number_input("First data row", 1, 10 ** 6, int(imp.first_row), key="adv_r1")
        r2 = c4.number_input("Last data row", 1, 10 ** 6, int(imp.last_row), key="adv_r2")
        tr = st.checkbox("Periods run across the columns (turn the table sideways)", value=imp.transposed, key="adv_tr")
        if st.button("Re-read with these settings", icon=":material/refresh:"):
            try:
                new = import_file(d["file"]["bytes"], d["file"]["name"], sheet=sh or "",
                                  header_row=int(hdr) or None, first_row=int(r1), last_row=int(r2), transpose=tr)
            except Exception as e:
                st.error(str(e))
                return
            _load_import(new, d["file"]["bytes"], d["file"]["name"], overrides=True)
            st.rerun()


# ----------------------------------------------------------------------------- 2. question & variables
def _variables(df, prof):
    numeric = prof[prof["kind"].isin(["numeric", "text-numeric"])]["name"].tolist()
    dates = prof[prof["kind"] == "date"]["name"].tolist()
    ids = prof[prof["kind"].isin(["id-like", "categorical"])]["name"].tolist()
    dflt = ss.get("spec_defaults", {})
    if not numeric:
        st.error("No numeric columns were found, so there is nothing to model. Check the preview or use Advanced import.")
        return None

    if "w_y" not in ss or ss.w_y not in numeric:
        ss.w_y = dflt.get("y") if dflt.get("y") in numeric else numeric[-1] if len(numeric) > 1 else numeric[0]
    if "w_xs" not in ss:
        ss.w_xs = [x for x in dflt.get("xs", []) if x in numeric and x != ss.w_y]
    ss.w_xs = [x for x in ss.w_xs if x in numeric and x != ss.w_y]
    if "w_date" not in ss:
        ss.w_date = dflt.get("date") if dflt.get("date") in dates else (dates[0] if dates else "(none)")
    if "w_id" not in ss:
        uniq_ids = [c for c in prof[prof["kind"] == "id-like"]["name"] if df[c].nunique() == len(df)]
        ss.w_id = dflt.get("id") if dflt.get("id") in ids else (uniq_ids[0] if uniq_ids else "(none)")
    if "w_q" not in ss:
        ss.w_q = dflt.get("question", "")

    st.text_input("What investment question are you investigating? (optional)", key="w_q",
                  placeholder="e.g. Is this manager's performance explained by market exposure? / What drives NOI across our properties?",
                  help="Used only to frame the wording of the findings. Nothing is sent to any external service.")
    c1, c2 = st.columns([1, 2])
    y = c1.selectbox("Outcome to explain (Y)", numeric, key="w_y",
                     help="The number you want to understand — e.g. a fund's excess return, a loan's spread, a margin.")
    xs = c2.multiselect("Possible drivers (X)", [c for c in numeric if c != y], key="w_xs",
                        help="The variables that might explain Y. Text categories (e.g. sector) can be added as 0/1 "
                             "columns under Advanced settings.")
    c3, c4 = st.columns(2)
    date_col = c3.selectbox("Date column", ["(none)"] + dates, key="w_date",
                            help="With a date, ELK treats the data as a time series (ordered in time) and adds time-based checks.")
    date_col = None if date_col == "(none)" else date_col
    if dates and not date_col:
        c3.caption("No date selected: rows are treated as separate observations (e.g. loans, properties, companies).")

    # units & descriptions for the chosen variables only
    chosen = [y] + list(xs)
    rows = []
    for v in chosen:
        if v not in ss.units:
            ss.units[v] = guess_unit(v, df[v])
        rows.append({"variable": v, "role": "outcome (Y)" if v == y else "driver (X)", "unit": ss.units[v],
                     "description": ss.descriptions.get(v, "")})
    unconfirmed = [v for v in chosen if v not in ss.get("units_confirmed", set())]
    with st.container(border=True):
        st.markdown("**Confirm units** — ELK pre-filled its best guess; correct anything that's wrong. "
                    "Descriptions are optional and make the findings easier to read.")
        ed = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["variable", "role"],
                            key=f"units_tbl_{'|'.join(chosen)}",
                            column_config={"unit": st.column_config.SelectboxColumn("Unit", options=UNITS, required=True),
                                           "description": st.column_config.TextColumn("Description (optional)"),
                                           "variable": "Variable", "role": "Role"})
        for r in ed.itertuples():
            if ss.units.get(r.variable) != r.unit:
                ss.units[r.variable] = r.unit
                ss.units_confirmed.add(r.variable)
            ss.descriptions[r.variable] = r.description or ""
        if unconfirmed and ss.data["kind"] == "upload":
            st.caption(":material/info: Units drive the wording (e.g. '1 percentage point' vs '1 unit'), not the maths.")

    adv = _advanced_settings(df, prof, y, xs, date_col, dates, ids)

    # alpha eligibility (fund-return context)
    alpha_ok, reason = False, ""
    if ss.units.get(y) in RETURN_UNITS and date_col:
        y_excess = y in ss.excess_cols
        x_ok = all(x in ss.excess_cols for x in xs) if xs else False
        if not y_excess or not x_ok:
            with st.container(border=True):
                st.markdown(":material/trending_up: **Return analysis.** To read the intercept as *alpha* (performance "
                            "not explained by the drivers), the outcome and drivers must be returns above the risk-free rate.")
                if not y_excess:
                    if st.checkbox(f"'{y}' is already a return above the risk-free rate", key=f"yex_{y}"):
                        ss.excess_cols.add(y)
                        y_excess = True
                    else:
                        st.caption("Or create it under **Advanced settings → Excess returns**.")
                if xs and not x_ok and st.checkbox("The drivers are excess or long–short factor returns", key="w_xs_excess"):
                    x_ok = True
        alpha_ok = bool(y_excess and x_ok)
        if not y_excess:
            reason = f"'{y}' is not confirmed as a return above the risk-free rate."
        elif not x_ok:
            reason = "the drivers are not all confirmed as excess or long–short factor returns."
    elif ss.units.get(y) in RETURN_UNITS:
        reason = "alpha needs returns over time (select a date column)."
    else:
        reason = "the outcome is not a fund return, so the intercept is simply the baseline level of the outcome."

    freq = adv["freq"]
    spec = {"y": y, "xs": list(xs), "date_col": date_col, "id_col": adv["id_col"], "freq": freq,
            "alpha": float(adv["alpha"]), "alpha_ok": alpha_ok, "alpha_reason": reason, "is_ts": bool(date_col),
            "ppy": FREQ_PERIODS_PER_YEAR.get(freq) if freq else None, "question": ss.w_q.strip(),
            "units": {v: ss.units.get(v) for v in chosen}, "desc": {v: ss.descriptions.get(v, "") for v in chosen}}
    if spec != ss.spec:
        ss.spec = spec
    return spec


def _advanced_settings(df, prof, y, xs, date_col, dates, ids):
    out = {"alpha": ss.get("w_alpha", 0.05), "id_col": None, "freq": None}
    with st.expander("Advanced settings — significance level, IDs, categories, excess returns, transformations",
                     icon=":material/tune:"):
        c1, c2, c3 = st.columns(3)
        out["alpha"] = c1.segmented_control("Significance level", [0.01, 0.05, 0.10], default=0.05,
                                            format_func=lambda v: f"{v:.0%}", key="w_alpha",
                                            help="How strong the evidence must be before ELK calls a result 'significant'. "
                                                 "5% is the usual convention.") or 0.05
        idc = c2.selectbox("Row label / ID", ["(none)"] + ids, key="w_id",
                           help="Used to name observations in charts and tables (e.g. loan ID).")
        out["id_col"] = None if idc == "(none)" else idc
        if date_col:
            g, _ = infer_frequency(parse_dates(df[date_col]))
            if "w_freq" not in ss or ss.w_freq not in FREQS:
                ss.w_freq = ss.get("spec_defaults", {}).get("freq") or (g if g in FREQS else "monthly")
            out["freq"] = c3.selectbox("Data frequency", FREQS, key="w_freq", help=f"Detected from the dates: {g}.")
        else:
            c3.caption("Frequency applies only to time series (choose a date column).")

        tabs = st.tabs(["Categories → 0/1 columns", "Excess returns", "Transform a column", "Missing values"])
        with tabs[0]:
            _dummies_ui(df, prof)
        with tabs[1]:
            _excess_ui(df, prof, dates)
        with tabs[2]:
            _transform_ui(df, prof, date_col)
        with tabs[3]:
            _fill_ui(df, prof, date_col)
    return out


def _dummies_ui(df, prof):
    done = ss.data.setdefault("dummied", [])
    cats = [c for c in prof[prof["kind"] == "categorical"]["name"] if c not in done]
    if not cats:
        st.caption("No text-category columns to convert.")
        return
    st.caption("Each category becomes a 0/1 column; one category is left out as the comparison point.")
    c1, c2, c3 = st.columns([2, 2, 1])
    col = c1.selectbox("Column", cats, key="dum_col")
    ref = c2.selectbox("Compare against", df[col].value_counts().index.astype(str).tolist(), key=f"dum_ref_{col}",
                       help="Default: the most common category.")
    c3.write("")
    if c3.button("Create", key="dum_go", width="stretch"):
        new, made = make_dummies(df, col, ref)
        ss.data["df"] = new
        done.append(col)
        for m in made:
            ss.units[m] = "0/1 indicator"
            ss.descriptions[m] = f"{col} = {m.split(col + '_', 1)[-1].replace('_', ' ')} (vs {ref})"
        ss.w_xs = list(ss.get("w_xs", [])) + made
        ss.data["transforms"].append(f"0/1 columns from '{col}' (comparison: {ref}): {', '.join(made)}")
        log("Category columns created", f"{col} → {', '.join(made)} (comparison {ref})")
        st.toast(f"Added {', '.join(made)} as drivers.", icon=":material/check:")
        st.rerun()


def _excess_ui(df, prof, dates):
    numeric = prof[prof["kind"].isin(["numeric", "text-numeric"])]["name"].tolist()
    ret_cols = [c for c in numeric if ss.units.get(c) in RETURN_UNITS and not c.endswith("_excess")]
    if not dates:
        st.caption("Needs a date column to match each period's risk-free rate.")
        return
    st.caption("Return above the risk-free rate = return − risk-free rate for the same period. Factor series such as "
               "Mkt-RF, SMB, HML are already in this form and must not be adjusted again.")
    todo = st.multiselect("Return columns", ret_cols or numeric, key="ex_cols",
                          help="Columns whose unit is a return are listed first; confirm units above.")
    c1, c2, c3 = st.columns(3)
    dcol = c1.selectbox("Date column", dates, key="ex_date")
    in_data = [c for c in numeric if c.upper() in ("RF", "RISK_FREE", "RISKFREE", "TBILL")]
    src = c2.selectbox("Risk-free source", [f"Column in data: {c}" for c in in_data] +
                       ["Fetch: Ken French (1-month T-bill)", "Fetch: FRED (3-month T-bill, DTB3)"], key="ex_src")
    fq_guess, _ = infer_frequency(parse_dates(df[dcol]))
    fq = c3.selectbox("Data frequency", FREQS, index=FREQS.index(fq_guess) if fq_guess in FREQS else 2, key="ex_freq")
    if st.button("Create excess-return columns", disabled=not todo):
        try:
            if src.startswith("Column"):
                col = src.split(": ", 1)[1]
                rf = pd.to_numeric(df[col], errors="coerce")
                rf_unit = ss.units.get(col)
                how = f"column '{col}'"
            else:
                with st.spinner("Fetching the risk-free rate…"):
                    series, prov = _rf(fq, "Ken French" if "French" in src else "FRED")
                rf = md.attach_rf(parse_dates(df[dcol]), series, fq)
                rf_unit = "decimal return (0.01 = 1%)"
                how = f"{prov['source']} · {prov.get('series', '')} · retrieved {prov['retrieved']}"
                ss.data["provenance"].append(("Risk-free rate", how))
            new = df.copy()
            for c in todo:
                pct = ("percent (1 = 1%)", "percent return (1 = 1%)")
                scale = 100 if ss.units.get(c) in pct and rf_unit not in pct else 1
                new[f"{c}_excess"] = pd.to_numeric(new[c], errors="coerce") - rf * scale
                ss.units[f"{c}_excess"] = ss.units.get(c)
                ss.descriptions[f"{c}_excess"] = f"{c} above the risk-free rate"
                ss.excess_cols.add(f"{c}_excess")
            ss.data["df"] = new
            ss.data["transforms"].append(f"Excess returns for {', '.join(todo)} using the risk-free rate from {how} ({fq})")
            log("Excess returns created", f"{', '.join(todo)} − RF ({how})")
            if int(rf.isna().sum()):
                st.warning(f"{int(rf.isna().sum())} row(s) had no matching risk-free value; their excess return is blank.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not create excess returns: {e}")


def _transform_ui(df, prof, date_col):
    numeric = prof[prof["kind"].isin(["numeric", "text-numeric"])]["name"].tolist()
    st.caption("Creates a new column; the original is kept unchanged. Nothing is converted unless you ask.")
    c1, c2, c3 = st.columns([2, 2, 1])
    col = c1.selectbox("Column", numeric, key="tr_col")
    kind = c2.selectbox("Transformation", ["Natural log", "% change from previous row", "Change from previous row",
                                           "Previous row's value (lag 1)"], key="tr_kind",
                        help="Row-to-row transformations use the date order when a date column is selected.")
    c3.write("")
    if c3.button("Add", key="tr_go", width="stretch"):
        s = pd.to_numeric(df[col], errors="coerce")
        order = parse_dates(df[date_col]).sort_values(kind="stable").index if date_col else df.index
        so = s.loc[order]
        if kind == "Natural log":
            bad = int((s <= 0).sum())
            new = np.log(s.where(s > 0))
            name, effect = f"log({col})", f"{bad} non-positive value(s) left blank" if bad else "no values lost"
        elif kind.startswith("%"):
            new = so.pct_change(fill_method=None).reindex(df.index)
            name, effect = f"pct_change({col})", "first row blank"
        elif kind.startswith("Change"):
            new = so.diff().reindex(df.index)
            name, effect = f"diff({col})", "first row blank"
        else:
            new = so.shift(1).reindex(df.index)
            name, effect = f"lag1({col})", "first row blank"
        d2 = df.copy()
        d2[name] = new
        ss.data["df"] = d2
        ss.units[name] = ("log value" if kind == "Natural log" else
                          "decimal return (0.01 = 1%)" if kind.startswith("%") else ss.units.get(col, "unitless / other"))
        ss.data["transforms"].append(f"{name}: {kind.lower()} of '{col}' ({effect})")
        log("Column transformed", f"{name} ({effect})")
        st.toast(f"Added {name} ({effect}).", icon=":material/check:")
        st.rerun()


def _fill_ui(df, prof, date_col):
    numeric = prof[prof["kind"].isin(["numeric", "text-numeric"])]["name"].tolist()
    miss = {c: int(pd.to_numeric(df[c], errors="coerce").isna().sum()) for c in numeric}
    miss = {c: n for c, n in miss.items() if n}
    st.caption("By default ELK leaves gaps alone and simply excludes incomplete rows from the model (you'll see how many). "
               "Filling gaps creates a new, clearly named column.")
    if not miss:
        st.caption("No missing numeric values.")
        return
    c1, c2 = st.columns(2)
    col = c1.selectbox("Column", list(miss), format_func=lambda c: f"{c} ({miss[c]} missing)", key="fill_col")
    opts = ["Column median"] + (["Previous value (time order)"] if date_col else [])
    how = c2.selectbox("Fill with", opts, key="fill_how")
    ok = st.checkbox(f"I understand {miss[col]} value(s) in '{col}' will be filled in a new column", key="fill_ok")
    if st.button("Create filled column", disabled=not ok):
        s = pd.to_numeric(df[col], errors="coerce")
        if how.startswith("Previous"):
            order = parse_dates(df[date_col]).sort_values(kind="stable").index
            new = s.loc[order].ffill().reindex(df.index)
        else:
            new = s.fillna(s.median())
        name = f"{col}_filled"
        d2 = df.copy()
        d2[name] = new
        ss.data["df"] = d2
        ss.units[name] = ss.units.get(col, "unitless / other")
        ss.data["transforms"].append(f"{name}: {miss[col]} missing value(s) of '{col}' filled with {how.lower()}")
        log("Missing values filled", f"{name}: {miss[col]} values, {how.lower()}")
        st.rerun()


# ----------------------------------------------------------------------------- 3. readiness
def _readiness(df, spec):
    if not spec or not spec["xs"]:
        st.info("Choose the outcome and at least one driver.", icon=":material/info:")
        return False
    suit = assess(df, spec["y"], spec["xs"], spec["date_col"], spec["id_col"], spec["freq"])
    ss.suit = suit
    m = suit.metrics
    n_raw, n = m.get("n_raw", 0), m.get("n_complete", 0)
    dropped = n_raw - n
    bits = [f"**{n:,}** of {n_raw:,} rows usable", f"**{m.get('k_params', 0)}** estimated numbers",
            ("time series · " + str(m.get("detected_frequency", ""))) if spec["is_ts"] else "separate observations"]
    if spec["is_ts"] and m.get("start") is not None:
        bits.append(f"{m['start']:%b %Y} – {m['end']:%b %Y}")
    st.markdown(" · ".join(bits))
    blocks = [i for i in suit.issues if i.level == "block"]
    warns = [i for i in suit.issues if i.level == "warn"]
    for i in blocks:
        st.error(f"**{i.title}.** {i.detail}", icon=LEVEL_ICON["block"])
    for i in warns:
        st.warning(f"**{i.title}.** {i.detail}", icon=LEVEL_ICON["warn"])
    if not blocks and not warns:
        st.success("No problems found with the data for this analysis.", icon=":material/check_circle:")
    infos = [i for i in suit.issues if i.level == "info"]
    if infos:
        with st.expander("Details"):
            for i in infos:
                st.markdown(f"- **{i.title}.** {i.detail}")
            if "variability" in m:
                st.dataframe(m["variability"], hide_index=True, width="stretch")
    ok = not blocks and ss.data.get("confirmed", True)
    if dropped and n_raw and dropped / n_raw > 0.05 and not blocks:
        ok = st.checkbox(f"Exclude the {dropped} row(s) ({dropped / n_raw:.0%}) with missing values in these variables",
                         key=f"miss_ok_{dropped}_{'|'.join([spec['y']] + spec['xs'])}") and ok
    if not ss.data.get("confirmed", True):
        st.caption(":material/help: Confirm the table above before running.")
    return ok


def render():
    page_header("Data & Setup", "Load data, check what ELK found, choose what to explain — then run the analysis.")
    if ss.data is None:
        _intro()
        _load()
        return
    st.markdown("#### 1 · Your data")
    df = ss.data["df"]
    prof = _found(df)
    df = ss.data["df"]
    prof = profile_columns(df)
    st.markdown("#### 2 · What do you want to explain?")
    spec = _variables(df, prof)
    st.markdown("#### 3 · Ready to run?")
    ready = _readiness(ss.data["df"], spec)
    ss.spec_ready = bool(ready)
    st.write("")
    c1, c2 = st.columns([3, 1.3])
    label = "Update analysis" if ss.get("analysis_ran") else "Run analysis"
    if c2.button(label, type="primary", icon=":material/play_arrow:", width="stretch", disabled=not ready):
        ss.analysis_ran = True
        log("Analysis run", f"{spec['y']} ~ {', '.join(spec['xs'])}")
        go_to(1)
    if not ready:
        c1.caption("Resolve the items above to run the analysis.")
    elif ss.get("analysis_ran"):
        c1.caption("Results update automatically when you change the selection; you can move freely between steps.")
