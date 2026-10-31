"""Exports: a self-contained HTML report (prints cleanly to PDF) and an Excel results workbook."""
from __future__ import annotations

import base64
import html
import io
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .insights import fmt, fmt_p, p_eq  # noqa: E402

LOGO = (Path(__file__).resolve().parent.parent / "assets" / "elk_logo.svg").read_text()
GREY = "#3E4A56"
ACCENT = "#B07D3E"


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.find("<svg"):]


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#B8BEC4")
    ax.tick_params(colors="#5B6670", labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)


def charts(fit, dates=None) -> list[tuple[str, str]]:
    out = []
    fitted, resid = fit.res.fittedvalues, fit.res.resid
    y = fit.frame[fit.spec.y]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if dates is not None:
        dd = pd.to_datetime(dates.reindex(fit.frame.index))
        ax.plot(dd, y.values, color="#9AA7B4", lw=1.1, label="Actual")
        ax.plot(dd, fitted.values, color=GREY, lw=1.3, label="Fitted")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Actual vs fitted over time", fontsize=9, loc="left")
    else:
        ax.scatter(fitted, y, s=9, color=GREY, alpha=0.6)
        lo, hi = min(fitted.min(), y.min()), max(fitted.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=ACCENT, lw=1)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Actual")
        ax.set_title("Actual vs fitted", fontsize=9, loc="left")
    _style(ax)
    out.append(("Actual vs fitted", _svg(fig)))

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.scatter(fitted, resid, s=9, color=GREY, alpha=0.6)
    ax.axhline(0, color=ACCENT, lw=1)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs fitted", fontsize=9, loc="left")
    _style(ax)
    out.append(("Residuals vs fitted", _svg(fig)))

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    z = np.sort((resid - resid.mean()) / resid.std(ddof=1))
    q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    ax.scatter(q, z, s=9, color=GREY, alpha=0.6)
    ax.plot([q.min(), q.max()], [q.min(), q.max()], color=ACCENT, lw=1)
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Standardised residual")
    ax.set_title("Normal Q–Q", fontsize=9, loc="left")
    _style(ax)
    out.append(("Normal Q–Q", _svg(fig)))
    return out


def _table(df: pd.DataFrame, pcols=(), index=True) -> str:
    d = df.copy()
    cells = []
    head = ([html.escape(str(d.index.name or ""))] if index else []) + [html.escape(str(c)) for c in d.columns]
    cells.append("<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>")
    for idx, row in d.iterrows():
        tds = [f"<td class='l'>{html.escape(str(idx))}</td>"] if index else []
        for c, v in row.items():
            if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool):
                fv = float(v)
                s = fmt_p(fv) if c in pcols else (f"{int(fv):,}" if np.isfinite(fv) and fv.is_integer() else fmt(fv))
                tds.append(f"<td>{s}</td>")
            else:
                tds.append(f"<td class='l'>{html.escape('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))}</td>")
        cells.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(cells) + "</table>"


CSS = """
:root{--ink:#22272C;--mute:#5B6670;--line:#DADDE1;--soft:#F2F3F5;--acc:#B07D3E;--bad:#B5473A;--ok:#4F7A5B}
*{box-sizing:border-box}body{font-family:Inter,'Segoe UI',Helvetica,Arial,sans-serif;color:var(--ink);background:#fff;margin:0;
font-size:13px;line-height:1.5}main{max-width:900px;margin:0 auto;padding:32px 28px 60px}
header{display:flex;align-items:center;gap:14px;border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:6px}
header svg{width:46px;height:46px;color:#3E4A56}h1{font-size:22px;margin:0;letter-spacing:.02em}
.sub{color:var(--mute);font-size:12px}h2{font-size:15px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}
h3{font-size:13px;margin:16px 0 6px}table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:12px}
th{background:var(--soft);text-align:right;font-weight:600;padding:5px 8px;border-bottom:1px solid var(--line)}
td{padding:4px 8px;border-bottom:1px solid #EEF0F2;text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td.l{text-align:left}.kv{display:grid;grid-template-columns:190px 1fr;gap:3px 14px;font-size:12.5px}
.kv div:nth-child(odd){color:var(--mute)}.item{padding:5px 10px;margin:4px 0;border-left:3px solid var(--line);background:#FAFAFA}
.item.warning{border-color:var(--acc)}.item.judgment{border-color:#8A939B}.tag{font-size:10px;text-transform:uppercase;
letter-spacing:.06em;color:var(--mute);margin-right:6px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.charts svg{width:100%;height:auto}.banner{background:#FBF5EC;border:1px solid #E8D6BA;padding:8px 12px;border-radius:6px;
margin:12px 0;font-size:12px}.foot{color:var(--mute);font-size:11px;margin-top:30px;border-top:1px solid var(--line);padding-top:10px}
@media print{main{padding:0}h2{break-after:avoid}table,.item,.charts>div{break-inside:avoid}}
"""


def html_report(ctx: dict) -> str:
    """ctx keys: title, provenance(list of str), spec(dict), suit, fit, diag(list of dict rows), vif(df), flagged(df),
    comparison(dict|None), narrative(list), notes(str), log(list of dict), dates(series|None), alpha"""
    fit = ctx["fit"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>ELK report — {e(ctx['title'])}</title><style>{CSS}</style></head><body><main>",
             f"<header>{LOGO}<div><h1>ELK · Model Audit Report</h1><div class='sub'>{e(ctx['title'])} · generated {now}</div></div></header>"]
    if ctx.get("synthetic"):
        parts.append("<div class='banner'><b>Synthetic data.</b> Every value in this dataset was generated artificially for "
                     "demonstration. It does not describe any real borrower, lender or portfolio.</div>")
    parts.append("<div class='banner' style='background:#F4F5F7;border-color:#DADDE1'>This report describes statistical "
                 "associations in the data supplied. It is not investment advice and does not replace analyst judgment.</div>")

    # 0 key takeaways
    if ctx.get("question"):
        parts.append(f"<p><b>Investment question:</b> {e(ctx['question'])}</p>")
    if ctx.get("takeaways"):
        parts.append("<h2>Key takeaways</h2>")
        for f in ctx["takeaways"]:
            col = {"good": "var(--ok)", "caution": "var(--acc)", "problem": "var(--bad)"}.get(f.tone, "#8A939B")
            body = e(f.finding)
            if f.points:
                body += "<ul>" + "".join(f"<li>{e(p.replace('**', ''))}</li>" for p in f.points) + "</ul>"
            nxt = ("<div><span class='tag'>Next steps</span><ul>" + "".join(f"<li>{e(n)}</li>" for n in f.next)
                   + "</ul></div>") if f.next else ""
            parts.append(f"<div class='item' style='border-color:{col}'><b>{e(f.title)}</b><div>{body}</div>"
                         f"<div><span class='tag'>Why it matters</span>{e(f.why)}</div>{nxt}"
                         + (f"<div class='sub'>{e(f.evidence)}</div>" if f.evidence else "") + "</div>")

    # 1 data & provenance
    parts.append("<h2>1. Data and provenance</h2><div class='kv'>")
    for k, v in ctx["provenance"]:
        parts.append(f"<div>{e(k)}</div><div>{e(str(v))}</div>")
    parts.append("</div>")

    # 2 specification
    sp = ctx["spec"]
    parts.append("<h2>2. Model specification</h2><div class='kv'>")
    for k, v in sp.items():
        parts.append(f"<div>{e(k)}</div><div>{e(str(v))}</div>")
    parts.append("</div>")

    # 3 suitability
    suit = ctx["suit"]
    parts.append("<h2>3. Data suitability</h2>")
    m = suit.metrics
    parts.append(f"<p>Rows supplied: {m.get('n_raw')} · used after complete-case filtering: {m.get('n_complete')} · "
                 f"parameters: {m.get('k_params')} · residual df: {m.get('df_resid')} · structure: {m.get('structure')}</p>")
    for i in suit.issues:
        cls = "warning" if i.level in ("warn", "block") else ""
        parts.append(f"<div class='item {cls}'><span class='tag'>{e(i.level)}</span><b>{e(i.title)}.</b> {e(i.detail)}</div>")

    # 4 results
    parts.append("<h2>4. Regression results</h2>")
    st = fit.stats
    parts.append(f"<p>n = {st['n']} · R² = {st['r2']:.4f} · adjusted R² = {st['adj_r2']:.4f} · {e(st['f_label'])}: "
                 f"F = {fmt(st['f_stat'])}, {p_eq(st['f_pvalue'])} · AIC = {fmt(st['aic'])} · BIC = {fmt(st['bic'])} · "
                 f"residual std. error = {fmt(st['rmse'])}</p>")
    ct = fit.coef.rename(index={"const": ctx.get("intercept_label", "Intercept")})
    ct.columns = ["Estimate", "Std. error", "t", "p-value", "CI low", "CI high"]
    ct.index.name = "Term"
    parts.append(_table(ct, pcols=("p-value",)))
    an = fit.anova.set_index("source")
    an.index.name = "ANOVA"
    parts.append(_table(an, pcols=("p_value",)))
    parts.append("<div class='charts'>" + "".join(f"<div>{s}</div>" for _, s in charts(fit, ctx.get("dates"))) + "</div>")

    # 5 diagnostics
    parts.append("<h2>5. Diagnostics</h2>")
    dg = pd.DataFrame(ctx["diag"]).set_index("Test")
    parts.append(_table(dg, pcols=("p-value",)))
    if ctx.get("vif") is not None and len(ctx["vif"]):
        parts.append("<h3>Variance inflation factors</h3>" + _table(ctx["vif"].set_index("variable")))
    fl = ctx.get("flagged")
    if fl is not None and len(fl):
        f2 = fl[["row", "actual", "fitted", "studentized_resid", "cooks_d", "flag"]].head(15).set_index("row")
        parts.append(f"<h3>Flagged observations ({len(fl)}; top 15 by Cook's distance)</h3>" + _table(f2))
        parts.append("<p class='sub'>Flags are for review. No observations were deleted from the data.</p>")

    # 6 comparison
    cmp_ = ctx.get("comparison")
    if cmp_:
        parts.append("<h2>6. Model comparison and adjustments</h2>")
        parts.append(f"<p><b>Model A:</b> {e(cmp_['a_desc'])}<br><b>Model B:</b> {e(cmp_['b_desc'])}</p>")
        if cmp_.get("justification"):
            parts.append(f"<p><b>Analyst justification for Model B:</b> {e(cmp_['justification'])}</p>")
        for line in cmp_.get("changes", []):
            parts.append(f"<div class='item'>{e(line.replace('**', ''))}</div>")
        for w in cmp_["result"]["warnings"]:
            parts.append(f"<div class='item warning'>{e(w)}</div>")
        parts.append(_table(cmp_["result"]["metrics"].set_index("metric")))
        parts.append(_table(cmp_["result"]["coef_table"], pcols=("p A", "p B")))
        for t in cmp_["result"]["tests"]:
            parts.append(f"<div class='item'><b>{e(t['name'])}</b>: statistic {fmt(t['stat'])}, df {e(t['df'])}, "
                         f"{p_eq(t['p_value'])}. H₀: {e(t['null'])} {e(t['reading'])} <i>{e(t['caveat'])}</i></div>")
        if cmp_["result"].get("nested_note"):
            parts.append(f"<div class='item'>{e(cmp_['result']['nested_note'])}</div>")
        if cmp_.get("oos") is not None:
            parts.append(f"<h3>Out-of-sample check</h3><p>{e(cmp_['oos']['method'])}</p>" + _table(cmp_["oos"]["table"].set_index("model")))
            parts.append("<p class='sub'>In-sample fit does not guarantee forecasting performance.</p>")

    # 7 interpretation
    parts.append("<h2>7. Interpretation</h2>")
    for sec in ctx["narrative"]:
        parts.append(f"<h3>{e(sec['title'])}</h3>")
        for kind, text in sec["items"]:
            lab = {"fact": "measured", "warning": "warning", "judgment": "judgment"}[kind]
            parts.append(f"<div class='item {kind}'><span class='tag'>{lab}</span>{e(text)}</div>")
    if ctx.get("notes"):
        parts.append("<h3>Analyst notes</h3><div class='item judgment'>" + e(ctx["notes"]).replace("\n", "<br>") + "</div>")

    # 8 methodology & log
    parts.append("<h2>8. Methodology, assumptions and audit trail</h2><ul>"
                 "<li>Ordinary least squares with an intercept (statsmodels). Rows with missing values in modelled columns are "
                 "excluded (complete-case); nothing is imputed or interpolated.</li>"
                 "<li>Classical inference assumes linearity, independent errors with constant variance, no perfect collinearity "
                 "and (in small samples) normal errors. Robust options change standard errors only, never coefficients.</li>"
                 "<li>Time-series tests are applied only when a date column orders the data.</li>"
                 "<li>Coefficients describe associations, not causal effects. p-values alone do not establish alpha or skill.</li></ul>")
    if ctx.get("log"):
        lg = pd.DataFrame(ctx["log"])
        lg.index = range(1, len(lg) + 1)
        lg.index.name = "#"
        parts.append("<h3>Audit trail</h3>" + _table(lg))
    import statsmodels
    parts.append(f"<div class='foot'>ELK — Quantitative Model Intelligence · Python {platform.python_version()} · pandas "
                 f"{pd.__version__} · statsmodels {statsmodels.__version__} · numerical results produced by statistical code, "
                 f"interpretation by transparent rules.</div></main></body></html>")
    return "".join(parts)


def excel_report(ctx: dict) -> bytes:
    fit = ctx["fit"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summ = pd.DataFrame(ctx["provenance"] + list(ctx["spec"].items()) + [
            ("Observations", fit.stats["n"]), ("R²", fit.stats["r2"]), ("Adjusted R²", fit.stats["adj_r2"]),
            ("F statistic", fit.stats["f_stat"]), ("F p-value", fit.stats["f_pvalue"]), ("AIC", fit.stats["aic"]),
            ("BIC", fit.stats["bic"]), ("Residual std. error", fit.stats["rmse"])], columns=["Item", "Value"])
        summ["Value"] = summ["Value"].astype(str)
        summ.to_excel(xw, sheet_name="Summary", index=False)
        fit.coef.rename(index={"const": ctx.get("intercept_label", "Intercept")}).to_excel(xw, sheet_name="Coefficients")
        fit.anova.to_excel(xw, sheet_name="ANOVA", index=False)
        pd.DataFrame(ctx["diag"]).to_excel(xw, sheet_name="Diagnostics", index=False)
        if ctx.get("vif") is not None and len(ctx["vif"]):
            ctx["vif"].to_excel(xw, sheet_name="VIF", index=False)
        if ctx.get("flagged") is not None and len(ctx["flagged"]):
            ctx["flagged"].to_excel(xw, sheet_name="Flagged observations", index=False)
        suit = ctx["suit"]
        pd.DataFrame([i.__dict__ for i in suit.issues]).to_excel(xw, sheet_name="Suitability", index=False)
        if ctx.get("comparison"):
            c = ctx["comparison"]["result"]
            c["metrics"].to_excel(xw, sheet_name="Comparison", index=False)
            c["coef_table"].to_excel(xw, sheet_name="Coef comparison")
        if ctx.get("takeaways"):
            pd.DataFrame([{"takeaway": f.title, "finding": f.finding + (" " + " ".join(f.points) if f.points else ""),
                           "why it matters": f.why, "next steps": " | ".join(f.next), "evidence": f.evidence}
                          for f in ctx["takeaways"]]).to_excel(xw, sheet_name="Key takeaways", index=False)
        rows = []
        for sec in ctx["narrative"]:
            for kind, text in sec["items"]:
                rows.append({"section": sec["title"], "type": kind, "text": text})
        pd.DataFrame(rows).to_excel(xw, sheet_name="Interpretation", index=False)
        data = fit.frame.copy()
        if ctx.get("labels") is not None:
            data.insert(0, "row", ctx["labels"].reindex(data.index).values)
        data["fitted"] = fit.res.fittedvalues
        data["residual"] = fit.res.resid
        data.to_excel(xw, sheet_name="Data used", index=False)
        if ctx.get("log"):
            pd.DataFrame(ctx["log"]).to_excel(xw, sheet_name="Audit trail", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(70, max(10, width + 2))
    return buf.getvalue()
