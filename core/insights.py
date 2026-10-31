"""Rule-based interpretation. Every number comes from the fitted model; text only explains it."""
from __future__ import annotations

import numpy as np

UNITS = ["unitless / other", "decimal return (0.01 = 1%)", "percent return (1 = 1%)", "percent (1 = 1%)", "basis points", "USD", "USD millions",
         "multiple (x)", "ratio (0–1)", "count", "0/1 indicator", "log value", "price / index level"]
RETURN_UNITS = ("decimal return (0.01 = 1%)", "percent return (1 = 1%)")
PERCENT_UNITS = ("percent (1 = 1%)", "percent return (1 = 1%)")


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.1f}"
    if ax >= 1:
        return f"{x:,.2f}"
    if ax >= 0.01:
        return f"{x:.{max(3, digits - 1)}f}"
    return f"{x:.{digits}g}" if ax < 1e-4 else f"{x:.{digits + 1}f}"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    """'p = 0.031' or 'p < 0.001'."""
    f = fmt_p(p)
    return f"p {f}" if f.startswith("<") else f"p = {f}"


def in_unit(v, unit):
    """Human-readable value in a column's unit."""
    if unit == "decimal return (0.01 = 1%)":
        return f"{v * 100:.2f}%"
    if unit in PERCENT_UNITS:
        return f"{v:.2f}%"
    if unit == "basis points":
        return f"{v:,.1f} bps"
    if unit == "USD millions":
        return f"${v:,.2f}m"
    if unit == "USD":
        return f"${v:,.2f}"
    if unit == "multiple (x)":
        return f"{v:.2f}x"
    return fmt(v)


def step_phrase(x, unit):
    return {
        "decimal return (0.01 = 1%)": f"a 1 percentage-point higher {x}",
        "percent (1 = 1%)": f"a 1 percentage-point higher {x}",
        "percent return (1 = 1%)": f"a 1 percentage-point higher {x}",
        "basis points": f"a 1 bp higher {x}",
        "multiple (x)": f"a one-turn (1.0x) higher {x}",
        "0/1 indicator": f"{x} = 1 (versus 0)",
        "log value": f"a 1-unit higher {x} (≈ 2.7× the underlying level)",
        "USD millions": f"$1m more {x}",
    }.get(unit, f"a one-unit higher {x}")


def effect_scale(unit_x, unit_y, b, y):
    """Coefficient translated to the stated step of X, expressed in Y's unit, as 'an X higher/lower y'."""
    k = 0.01 if unit_x == "decimal return (0.01 = 1%)" else 1.0
    eff = b * k
    mag = in_unit(abs(eff), unit_y) if unit_y != "unitless / other" else fmt(abs(eff))
    if unit_y in RETURN_UNITS:
        mag = mag.replace("%", " percentage points")
    return f"a {mag} {'higher' if eff >= 0 else 'lower'} {y}"


def build_narrative(fit, diag: dict, vif, flagged, suit, ctx: dict) -> list[dict]:
    a = ctx["alpha"]
    y = fit.spec.y
    uy = ctx["units"].get(y, "unitless / other")
    coef = fit.coef
    st = fit.stats
    secs = []

    # (a) core estimate ---------------------------------------------------------
    items = []
    c0 = coef.loc["const"]
    if ctx.get("alpha_ok"):
        ppy = ctx.get("ppy") or 12
        ann = c0["estimate"] * ppy
        items.append(("fact", f"Alpha (intercept) is {in_unit(c0['estimate'], uy)} per period "
                              f"(≈ {in_unit(ann, uy)} a year, simple annualisation ×{ppy}); "
                              f"{int((1 - a) * 100)}% CI {in_unit(c0['ci_low'], uy)} to {in_unit(c0['ci_high'], uy)}, {p_eq(c0['p_value'])}."))
    else:
        reason = ctx.get("alpha_reason", "")
        items.append(("fact", f"The intercept ({fmt(c0['estimate'])}) is the model's predicted {y} when every explanatory "
                              f"variable equals zero." + (f" It is not labelled alpha: {reason}" if reason else "")))
    for t in fit.spec.xs:
        r = coef.loc[t]
        ux = ctx["units"].get(t, "unitless / other")
        if ctx.get("alpha_ok"):
            items.append(("fact", f"Sensitivity to {t} is {fmt(r['estimate'])}: {step_phrase(t, ux)} is associated with "
                                  f"{effect_scale(ux, uy, r['estimate'], y)}, holding the other factors fixed "
                                  f"({p_eq(r['p_value'])})."))
        else:
            items.append(("fact", f"{_cap(step_phrase(t, ux))} is associated with {effect_scale(ux, uy, r['estimate'], y)}, "
                                  f"holding the other variables fixed ({int((1 - a) * 100)}% CI "
                                  f"{fmt(r['ci_low'])} to {fmt(r['ci_high'])} per unit; {p_eq(r['p_value'])})."))
    secs.append({"title": "Core estimates", "items": items})

    # (b) statistical evidence --------------------------------------------------
    items = []
    sig = [t for t in fit.spec.xs if coef.loc[t, "p_value"] < a]
    ns = [t for t in fit.spec.xs if coef.loc[t, "p_value"] >= a]
    items.append(("fact", f"Using {st['n']} observations, the model explains {st['r2'] * 100:.1f}% of the variation in {y} "
                          f"(adjusted R² {st['adj_r2']:.3f}). {st['f_label']}: F = {fmt(st['f_stat'])}, {p_eq(st['f_pvalue'])}."))
    if sig:
        items.append(("fact", f"Statistically different from zero at the {a:.0%} level: {', '.join(sig)}."))
    if ns:
        items.append(("fact", f"Not statistically different from zero at {a:.0%}: {', '.join(ns)}. This is absence of evidence, "
                              "not evidence of no relationship."))
    items.append(("fact", f"Inference uses {_cov_name(st)} standard errors."))
    secs.append({"title": "Statistical evidence", "items": items})

    # (c) diagnostics & limitations --------------------------------------------
    items = []
    d = diag
    if d.get("bp", {}).get("level") == "warn":
        items.append(("warning", f"Breusch–Pagan rejects constant residual variance ({p_eq(d['bp']['p'])}). Coefficients remain "
                                 "unbiased, but classical standard errors and p-values may be misleading."))
    if d.get("bg", {}).get("level") == "warn":
        items.append(("warning", f"Breusch–Godfrey finds serial correlation in residuals ({p_eq(d['bg']['p'])}, "
                                 f"{d['bg'].get('lags')} lags). Classical standard errors are likely understated."))
    if d.get("dw", {}).get("level") == "warn":
        items.append(("warning", f"Durbin–Watson is {d['dw']['stat']:.2f}, away from 2, indicating first-order autocorrelation."))
    if d.get("arch", {}).get("level") == "warn":
        items.append(("warning", f"ARCH LM detects volatility clustering ({p_eq(d['arch']['p'])})."))
    if d.get("reset", {}).get("level") == "warn":
        items.append(("warning", f"RESET suggests the linear form may be missing curvature or interactions ({p_eq(d['reset']['p'])})."))
    if d.get("jb", {}).get("level") == "warn":
        small = st["n"] < 60
        items.append(("warning" if small else "fact",
                      f"Residuals are not normally distributed (Jarque–Bera {p_eq(d['jb']['p'])}). "
                      + ("With a small sample this weakens t- and F-test reliability." if small else
                         "With this sample size, OLS inference is approximately valid regardless.")))
    if vif is not None and len(vif):
        hi = vif[vif["VIF"] > 10]
        mod = vif[(vif["VIF"] > 5) & (vif["VIF"] <= 10)]
        if len(hi):
            items.append(("warning", "High collinearity: " + ", ".join(f"{r.variable} (VIF {r.VIF:.1f})" for r in hi.itertuples())
                          + ". Their individual coefficients are imprecise and can flip sign; interpret them jointly."))
        elif len(mod):
            items.append(("warning", "Moderate collinearity: " + ", ".join(f"{r.variable} (VIF {r.VIF:.1f})" for r in mod.itertuples()) + "."))
    if flagged is not None and len(flagged):
        items.append(("warning", f"{len(flagged)} observation(s) are flagged as outliers or influential (largest Cook's distance "
                                 f"{flagged['cooks_d'].max():.3f}, at {flagged['row'].iloc[0]}). Results may depend on them."))
    for iss in suit.issues:
        if iss.level == "warn":
            items.append(("warning", f"Data: {iss.title}. {iss.detail}"))
    if not items:
        items.append(("fact", "No diagnostic test flagged a problem at the chosen significance level. Passing tests does not "
                              "prove the model is correctly specified."))
    secs.append({"title": "Diagnostics and limitations", "items": items})

    # (d) can / cannot conclude -------------------------------------------------
    items = []
    unreliable = any(d.get(k, {}).get("level") == "warn" for k in ("bp", "bg", "arch")) and st["cov_type"] == "nonrobust"
    if sig:
        items.append(("fact", f"The data show statistically detectable associations between {y} and {', '.join(sig)} in this sample"
                              + (" — but the p-values rely on standard errors the diagnostics question; confirm with robust errors."
                                 if unreliable else ".")))
    items.append(("judgment", "These are associations, not causal effects. Omitted variables, reverse causality or selection could drive them."))
    if ctx.get("alpha_ok"):
        pa = coef.loc["const", "p_value"]
        if pa < a:
            items.append(("judgment", "Alpha is statistically different from zero in this sample and specification. That alone does not "
                                      "establish persistent skill: it can reflect omitted factors, the sample window, fees, or chance across "
                                      "many funds tested."))
        else:
            items.append(("fact", "Alpha is not statistically different from zero: the data do not support a claim of "
                                  "risk-adjusted outperformance relative to these factors."))
    if not ctx.get("is_ts"):
        items.append(("judgment", "Coefficients compare observations with different characteristics; they do not say what would "
                                  "happen if one observation's characteristics changed."))
    items.append(("judgment", "Statistical significance is not economic significance — judge whether effect sizes matter in practice. "
                              "In-sample fit does not guarantee out-of-sample performance."))
    secs.append({"title": "What can and cannot be concluded", "items": items})

    # (e) next steps ------------------------------------------------------------
    items = []
    if d.get("bp", {}).get("level") == "warn" and st["cov_type"] == "nonrobust":
        items.append(("judgment", "Re-estimate with heteroskedasticity-robust (HC3) standard errors and compare inference."))
    if (d.get("bg", {}).get("level") == "warn" or d.get("arch", {}).get("level") == "warn") and st["cov_type"] != "HAC":
        items.append(("judgment", "Use HAC (Newey–West) standard errors to account for serial dependence."))
    if vif is not None and len(vif) and (vif["VIF"] > 10).any():
        items.append(("judgment", "Consider whether the highly collinear variables measure the same thing; if so, keep one with a stated "
                                  "reason, or combine them. Check that the remaining coefficients are stable."))
    if flagged is not None and len(flagged):
        items.append(("judgment", "Review the flagged rows for data errors; run a sensitivity model excluding them (the original stays "
                                  "the reference)."))
    if d.get("reset", {}).get("level") == "warn":
        items.append(("judgment", "Try transformations (logs, squares) or interactions suggested by economic reasoning."))
    if st["n"] / (st["k"] + 1) < 10:
        items.append(("judgment", "Gather more observations before relying on individual coefficients."))
    items.append(("judgment", "Compare against an alternative specification and check out-of-sample error."))
    secs.append({"title": "Useful next steps", "items": items})
    return secs


def _cap(s):
    return s[:1].upper() + s[1:] if s.startswith("a ") else s


def _cov_name(st):
    return {"nonrobust": "classical", "HC1": "HC1 heteroskedasticity-robust", "HC3": "HC3 heteroskedasticity-robust",
            "HAC": f"HAC/Newey–West ({st.get('hac_lags')} lags)"}.get(st["cov_type"], st["cov_type"])


def suggest_remedies(diag: dict, vif, flagged, is_ts: bool) -> list[dict]:
    out = []
    if diag.get("bp", {}).get("level") == "warn":
        out.append({"id": "HC3", "title": "Re-check confidence allowing for uneven errors (HC3)",
                    "why": "Diagnostics found the model's errors are larger for some observations than others, which can "
                           "overstate confidence.",
                    "effect": "only the confidence levels (standard errors, p-values). Effects stay the same."})
    if is_ts and (diag.get("bg", {}).get("level") == "warn" or diag.get("arch", {}).get("level") == "warn"
                  or diag.get("dw", {}).get("level") == "warn"):
        out.append({"id": "HAC", "title": "Re-check confidence allowing for patterns over time (HAC / Newey–West)",
                    "why": "Diagnostics found errors that carry over between periods or volatility that clusters.",
                    "effect": "only the confidence levels. Effects stay the same."})
    if vif is not None and len(vif) and (vif["VIF"] > 10).any():
        hi = vif.sort_values("VIF", ascending=False)["variable"].tolist()
        out.append({"id": "DROP", "title": "Drop one of two overlapping drivers",
                    "why": f"{' and '.join(vif[vif['VIF'] > 10]['variable'])} carry largely the same information.",
                    "effect": "the effects. Needs a reason; dropping a genuinely relevant driver distorts the others.",
                    "candidates": hi})
    if flagged is not None and len(flagged):
        out.append({"id": "EXCL", "title": "Re-run without the unusual observations",
                    "why": f"{len(flagged)} observation(s) have an outsized influence on the results.",
                    "effect": "shows whether conclusions depend on them. Nothing is deleted from your data."})
    return out
