"""Plain-English findings: Key finding · Why it matters for the investment · Suggested next steps · Evidence.

All numbers come from the fitted statsmodels results. The wording adapts to the kind of data (fund returns vs factors,
other time series, or cross-sectional records such as loans, properties or companies), to the variable names, units
and descriptions the analyst confirmed, and to the optional investment question. Nothing here is causal, and nothing
here is a recommendation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .insights import RETURN_UNITS, fmt, fmt_p, in_unit, p_eq


@dataclass
class Finding:
    title: str
    finding: str
    why: str
    next: list[str] = field(default_factory=list)
    evidence: str = ""
    tone: str = "neutral"  # good | caution | problem | neutral
    points: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers
def context_type(spec: dict, units: dict) -> str:
    if spec.get("alpha_ok"):
        return "fund"
    if spec.get("is_ts"):
        return "returns_ts" if units.get(spec["y"]) in RETURN_UNITS else "time_series"
    return "cross_section"


def nm(v: str, desc: dict) -> str:
    d = (desc or {}).get(v, "")
    return d if d else v


def strength(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return "not available"
    if p < min(0.01, alpha):
        return "strong evidence"
    if p < alpha:
        return "moderate evidence"
    if p < max(0.10, alpha * 2):
        return "weak evidence"
    return "no reliable evidence"


def _step(unit: str):
    """(size of a 'one-step' change in X, phrase)"""
    return {
        "decimal return (0.01 = 1%)": (0.01, "1 percentage point"),
        "percent (1 = 1%)": (1.0, "1 percentage point"),
        "ratio (0–1)": (0.01, "1 percentage point"),
        "basis points": (100.0, "100 bps"),
        "multiple (x)": (1.0, "one turn (1.0x)"),
        "USD millions": (1.0, "$1m"),
        "log value": (np.log(1.1), "10%"),
        "0/1 indicator": (1.0, None),
    }.get(unit, (1.0, "one unit"))


def amount(v: float, unit: str) -> str:
    """Size of a change, in the outcome's unit, without a sign."""
    a = abs(v)
    if unit == "decimal return (0.01 = 1%)":
        return f"{a * 100:.2f} percentage points"
    if unit == "percent (1 = 1%)":
        return f"{a:.2f} percentage points"
    if unit == "ratio (0–1)":
        return f"{a * 100:.2f} percentage points"
    if unit == "log value":
        return f"{(np.exp(a) - 1) * 100:.1f}%"
    return in_unit(a, unit) if unit not in ("unitless / other", "count") else fmt(a)


def effect_sentence(x, b, spec, units, desc):
    y = spec["y"]
    ux, uy = units.get(x, "unitless / other"), units.get(y, "unitless / other")
    size, phrase = _step(ux)
    eff = b * size
    dirn = "higher" if eff >= 0 else "lower"
    if ux == "0/1 indicator":
        return f"Where {nm(x, desc)} applies, {nm(y, desc)} is on average {amount(eff, uy)} {dirn}"
    else:
        xn = nm(x, desc)
        if ux == "log value" and xn == x:
            xn = re.sub(r"^(log_|ln_|log\()|\)$", "", x)
        lead = f"Each {phrase} higher {xn}"
    if uy == "log value":
        return f"{lead} goes with about {amount(eff, uy)} {dirn} {nm(y, desc)}"
    return f"{lead} goes with {amount(eff, uy)} {dirn} {nm(y, desc)}"


def driver_table(fit, spec, units, desc) -> list[dict]:
    a = spec["alpha"]
    X = fit.frame[fit.spec.xs]
    sy = fit.frame[fit.spec.y].std()
    rows = []
    for x in fit.spec.xs:
        r = fit.coef.loc[x]
        sx = X[x].std()
        typical = r["estimate"] * sx
        share = abs(typical) / sy * 100 if sy > 0 else np.nan
        binary = units.get(x) == "0/1 indicator"
        rows.append({
            "variable": x, "label": nm(x, desc),
            "direction": "↑ higher" if r["estimate"] > 0 else "↓ lower",
            "effect": effect_sentence(x, r["estimate"], spec, units, desc),
            "typical": (None if binary else f"{amount(typical, units.get(spec['y'], ''))} "
                        f"({share:.0f}% of the usual variation in {nm(spec['y'], desc)})"),
            "importance": share,
            "evidence": strength(r["p_value"], a), "p": r["p_value"],
            "range": f"{fmt(r['ci_low'])} to {fmt(r['ci_high'])}",
        })
    return rows


# ----------------------------------------------------------------------------- headline takeaways
def takeaways(fit, diag, vif, flagged, suit, spec, units, desc) -> list[Finding]:
    a = spec["alpha"]
    y = spec["y"]
    Y = nm(y, desc)
    ctype = context_type(spec, units)
    rows = driver_table(fit, spec, units, desc)
    sig = sorted([r for r in rows if r["p"] < a], key=lambda r: -r["importance"])
    weak = [r for r in rows if r["p"] >= a]
    st = fit.stats
    out = []

    # 1 — drivers / exposures
    if sig:
        top = sig[:3]
        listing = "The clearest links, largest first:"
        if ctype == "fund":
            title = "What the fund's returns are exposed to"
            why = ("These sensitivities describe the kind of risk the fund carries. Exposure to widely available factors "
                   "can usually be obtained cheaply (e.g. through index or factor funds), so fees and claims of skill "
                   "should be judged against them.")
            nxt = ["Check the exposures match the fund's stated strategy and benchmark.",
                   "Re-estimate over different periods to see whether the exposures are stable."]
        elif ctype in ("returns_ts", "time_series"):
            title = f"What moves with {Y} over time"
            why = (f"These are the variables that have tended to move together with {Y}. They indicate which conditions "
                   f"{Y} is sensitive to — useful for stress-testing and for understanding risk, not for proving cause.")
            nxt = ["Ask whether each relationship has an economic explanation.",
                   "Check stability over sub-periods; time-series relationships often shift."]
        else:
            title = f"What lines up with differences in {Y}"
            why = (f"Across the observations, these characteristics line up with higher or lower {Y}. That helps "
                   f"benchmark a new observation against the sample (e.g. whether its {Y} looks in line with its "
                   "characteristics) — but it compares different observations, not what would happen if one changed.")
            nxt = [f"Use the scenario tool in Insights & Export to see the {Y} the model implies for a specific case.",
                   "Check whether important characteristics are missing from the model."]
        f = Finding(title, listing, why, nxt, tone="good" if len(sig) else "neutral", points=[r["effect"] + "." for r in top],
                    evidence=f"Ranked by how much a typical change moves {Y}; all at {a:.0%} significance or better.")
        if weak:
            f.next.append(f"No reliable evidence for {', '.join(r['label'] for r in weak)} in this data — that is not "
                          "proof they don't matter (the sample may be too small or the effect too noisy).")
        out.append(f)
    else:
        out.append(Finding(f"No driver shows a reliable link to {Y}",
                           f"None of the selected drivers is statistically distinguishable from zero at {a:.0%}.",
                           "On this data the model cannot support a claim that these variables explain the outcome. "
                           "This is absence of evidence, not evidence of no relationship.",
                           ["Consider more observations, other drivers, or a different time window."], tone="caution"))

    # 2 — alpha (fund only)
    if ctype == "fund":
        c0 = fit.coef.loc["const"]
        uy = units.get(y)
        ppy = spec.get("ppy") or 12
        ann = c0["estimate"] * ppy
        beta = fit.coef.loc["Mkt-RF", "estimate"] if "Mkt-RF" in fit.coef.index else None
        btxt = f"The market sensitivity (beta) is {beta:.2f}. " if beta is not None else ""
        if c0["p_value"] < a:
            sign = "positive" if c0["estimate"] > 0 else "negative"
            out.append(Finding(
                f"Statistically {sign} alpha",
                f"{btxt}Alpha — return not explained by the factors — is about {in_unit(ann, uy)} a year "
                f"({in_unit(c0['estimate'], uy)} per period, {p_eq(c0['p_value'])}).",
                ("A " + sign + " alpha in this sample means the factors don't fully account for performance. On its own it "
                 "does not prove persistent skill: it can reflect a missing factor, the chosen period, or luck."),
                ["Test robustness across periods and with robust standard errors (Compare & Adjust).",
                 "Check whether a different benchmark or extra factor removes it.", "Review fees and capacity."],
                tone="good" if c0["estimate"] > 0 else "caution",
                evidence=f"Intercept {fmt(c0['estimate'])}, {int((1 - a) * 100)}% range {fmt(c0['ci_low'])} to {fmt(c0['ci_high'])}."))
        else:
            out.append(Finding(
                "No evidence of alpha",
                f"{btxt}Estimated alpha is {in_unit(ann, uy)} a year, but it is not statistically distinguishable from zero "
                f"({p_eq(c0['p_value'])}).",
                "Performance is accounted for by exposure to the factors; this model gives no support for a claim of "
                "risk-adjusted outperformance. (It doesn't prove there's none — the estimate is just too uncertain.)",
                ["Compare against the most appropriate benchmark and factor set.", "Review fees relative to cheaper "
                 "ways of obtaining the same exposures.", "Check diagnostic warnings before drawing conclusions."],
                tone="neutral",
                evidence=f"Intercept {fmt(c0['estimate'])}, {int((1 - a) * 100)}% range {fmt(c0['ci_low'])} to {fmt(c0['ci_high'])}."))

    # 3 — explanatory power
    r2 = st["r2"]
    what = "the ups and downs in" if spec.get("is_ts") else "the differences in"
    if ctype == "fund":
        why = ("A high share means the fund's returns are largely replicable with these factors; a low share means "
               "much of the return comes from other sources (stock selection, other risks, or noise).")
    elif r2 >= 0.7:
        why = "The drivers capture most of the variation; the model is a reasonable summary of how the outcome behaves."
    elif r2 >= 0.3:
        why = (f"Useful for understanding direction and rough size, but individual {Y} values can differ a lot from the "
               "model — fine for screening and benchmarking, not for precise pricing.")
    else:
        why = "Most of the variation comes from things outside the model; treat any single prediction with caution."
    out.append(Finding(f"The model explains {r2 * 100:.0f}% of {what} {Y}", f"Based on {st['n']} observations. "
                       f"A typical gap between the model and the actual value is {amount(st['rmse'], units.get(y, ''))}.",
                       why, ["Look for missing drivers if the share is low."] if r2 < 0.3 else [],
                       evidence=f"R² {r2:.3f}, adjusted R² {st['adj_r2']:.3f}, {st['f_label']} {p_eq(st['f_pvalue'])}."))

    # 4 — reliability
    issues, nxt = [], []
    if diag.get("bp", {}).get("level") == "warn" and st["cov_type"] == "nonrobust":
        issues.append("the size of the errors isn't consistent across observations")
        nxt.append("Run the HC3 robust-error check (Compare & Adjust, one click) to see whether conclusions survive.")
    if (diag.get("bg", {}).get("level") == "warn" or diag.get("arch", {}).get("level") == "warn") and st["cov_type"] != "HAC":
        issues.append("errors carry over from one period to the next")
        nxt.append("Run the HAC (Newey–West) check in Compare & Adjust.")
    if vif is not None and len(vif) and (vif["VIF"] > 10).any():
        pair = ", ".join(nm(v, desc) for v in vif[vif["VIF"] > 10]["variable"])
        issues.append(f"{pair} overlap heavily, so their individual effects can't be separated reliably")
        nxt.append("Consider keeping only one of the overlapping drivers (Compare & Adjust).")
    if flagged is not None and len(flagged):
        issues.append(f"{len(flagged)} observation(s) have an outsized influence on the results")
        nxt.append("Review those rows and run the sensitivity check excluding them.")
    if diag.get("reset", {}).get("level") == "warn":
        issues.append("a straight-line relationship may be too simple")
        nxt.append("Try a log or squared version of the main drivers (Advanced settings → Transform).")
    for i in suit.issues:
        if i.level == "warn":
            issues.append(i.title[:1].lower() + i.title[1:])
    if issues:
        out.append(Finding("Some results need care before you rely on them",
                           "Checks found that " + "; ".join(issues[:4]) + ".",
                           "These don't necessarily make the estimates wrong, but they can make the model look more "
                           "certain than it is — a driver could appear reliable when it isn't.", nxt[:3], tone="caution",
                           evidence="Details and test statistics are in Diagnostics."))
    else:
        out.append(Finding("No warning signs in the model checks", "The standard checks did not flag a problem.",
                           "That supports — but does not prove — that the model's confidence levels are trustworthy.",
                           ["Still compare an alternative specification before relying on it."], tone="good"))
    return out


# ----------------------------------------------------------------------------- diagnostic cards
def check_cards(fit, diag, vif, flagged, spec, units, desc) -> list[dict]:
    Y = nm(spec["y"], desc)
    a = spec["alpha"]
    cards = []

    def add(key, question, ok_text, bad_text, why_bad, next_bad, level, stat=None):
        cards.append({"key": key, "question": question, "level": level,
                      "finding": bad_text if level == "warn" else ok_text,
                      "why": why_bad if level == "warn" else "",
                      "next": next_bad if level == "warn" else [], "stat": stat})

    d = diag
    if "bp" in d:
        add("bp", "Are the model's errors similar in size across observations?",
            "Yes — no evidence that errors are bigger for some observations than others.",
            f"No — errors are larger for some observations than others ({p_eq(d['bp']['p'])}).",
            ("The estimated effects are still unbiased, but the confidence levels (p-values) can be overstated, so a "
             "driver might look more reliable than it is." + (" In loan or property data this often happens because "
                                                              "small assets are priced less consistently."
                                                              if not spec.get("is_ts") else "")),
            ["Compare with heteroskedasticity-robust (HC3) errors — one click in Compare & Adjust."], d["bp"]["level"], d["bp"])
    if "bg" in d:
        add("bg", "Do errors carry over from one period to the next?",
            "No meaningful carry-over detected.",
            f"Yes — this period's error tends to predict the next ({p_eq(d['bg']['p'])}, {d['bg'].get('lags')} lags).",
            "Patterns over time are left in the errors, so confidence levels are likely overstated; for funds this can "
            "also indicate smoothed or stale pricing.", ["Use HAC (Newey–West) errors in Compare & Adjust.",
                                                          "Consider whether a driver that changes over time is missing."],
            d["bg"]["level"], d["bg"])
    if "arch" in d:
        add("arch", "Do calm and turbulent periods cluster?",
            "No evidence of volatility clustering.",
            f"Yes — large surprises tend to follow large surprises ({p_eq(d['arch']['p'])}).",
            "Risk is not constant over time: the model's uncertainty is larger in stressed periods than an average "
            "suggests. Common in market returns.", ["Use HAC errors; for risk work consider volatility models."],
            d["arch"]["level"], d["arch"])
    if vif is not None and len(vif):
        hi = vif[vif["VIF"] > 10]
        lvl = "warn" if len(hi) else "ok"
        names = ", ".join(nm(v, desc) for v in hi["variable"])
        add("vif", "Do any drivers measure the same thing?",
            f"No — the drivers are distinct enough to separate their effects (highest overlap score {vif['VIF'].max():.1f}).",
            f"Yes — {names} overlap heavily (overlap score above 10).",
            "When two drivers carry the same information, the model can't tell which one matters; their individual "
            "effects become unstable and can even flip sign, although the model as a whole may still fit well.",
            ["Keep the driver that best fits your investment logic and compare (Compare & Adjust).",
             "Or interpret the overlapping drivers jointly rather than one by one."], lvl,
            {"stat": float(vif["VIF"].max()), "p": np.nan, "extra": "max VIF"})
    if flagged is not None:
        lvl = "warn" if len(flagged) else "ok"
        top = flagged["row"].head(3).tolist() if len(flagged) else []
        add("influence", "Do a few observations drive the results?",
            "No single observation has an outsized effect.",
            f"Yes — {len(flagged)} observation(s) stand out" + (f", e.g. {', '.join(map(str, top))}." if top else "."),
            "If a handful of unusual cases (data errors, one-off events, distressed assets) drive the estimates, "
            "conclusions may not hold for the typical case.",
            ["Check these rows for data errors.", "Run the sensitivity check without them (Compare & Adjust)."], lvl,
            {"stat": float(len(flagged)), "p": np.nan, "extra": "flagged observations"})
    if "reset" in d:
        add("reset", "Is a straight-line relationship good enough?",
            "Yes — no sign of a missing curve.",
            f"Possibly not — the relationship may bend ({p_eq(d['reset']['p'])}).",
            "Effects may be stronger at high or low levels of a driver (e.g. spreads rising faster at high leverage), "
            "which a straight line averages away.", ["Try a log or squared version of the main driver."],
            d["reset"]["level"], d["reset"])
    if "jb" in d:
        small = fit.stats["n"] < 60
        lvl = "warn" if d["jb"]["level"] == "warn" and small else "ok"
        add("jb", "Are there unusually large surprises (fat tails)?",
            ("Errors look broadly normal." if d["jb"]["level"] != "warn" else
             f"Some fat tails ({p_eq(d['jb']['p'])}), but with {fit.stats['n']} observations this has little effect on "
             "the reliability of the results."),
            f"Yes, and the sample is small ({fit.stats['n']} observations), so confidence levels are less reliable.",
            "Extreme outcomes happen more often than a normal distribution implies.",
            ["Gather more observations or check the extreme cases."], lvl, d["jb"])
    if "dw" in d:
        dw = d["dw"]
        add("dw", "Quick check: is each error related to the previous one?",
            f"Score {dw['stat']:.2f} — close to 2, so no strong sign.",
            f"Score {dw['stat']:.2f} — away from 2, suggesting errors follow each other.",
            "Same concern as carry-over above.", ["See the Breusch–Godfrey check."], dw["level"], dw)
    return cards


# ----------------------------------------------------------------------------- comparison bottom line
def comparison_takeaway(A, B, alpha, desc) -> Finding:
    common = [t for t in B.coef.index if t in A.coef.index and t != "const"]
    gained = [t for t in common if A.coef.loc[t, "p_value"] >= alpha > B.coef.loc[t, "p_value"]]
    lost = [t for t in common if A.coef.loc[t, "p_value"] < alpha <= B.coef.loc[t, "p_value"]]
    same_vars = set(A.spec.xs) == set(B.spec.xs) and A.frame.index.equals(B.frame.index)
    if same_vars:
        if not lost and not gained:
            return Finding("Your conclusions survive the correction",
                           "After recalculating confidence with robust errors, the same drivers remain reliable.",
                           "The warning from the model checks doesn't change which drivers you can rely on — a good sign "
                           "for using these results in an investment discussion.",
                           ["Report the robust (Model B) confidence levels alongside the original."], tone="good")
        return Finding("The correction changes what you can rely on",
                       ("No longer reliable after correction: " + ", ".join(nm(t, desc) for t in lost) + ". " if lost else "")
                       + ("Now reliable: " + ", ".join(nm(t, desc) for t in gained) + "." if gained else ""),
                       "The original confidence levels were overstated for these drivers; base conclusions on the "
                       "corrected version.", ["Use Model B as the reported model.",
                                               "Avoid strong claims about the drivers that lost reliability."], tone="caution")
    if B.spec.exclude_rows:
        big = []
        for t in common:
            ac, bc = A.coef.loc[t, "estimate"], B.coef.loc[t, "estimate"]
            if ac != 0 and abs(bc - ac) / abs(ac) > 0.25:
                big.append(nm(t, desc))
        if not big and not lost:
            return Finding("Results don't depend on the unusual observations",
                           f"Without the {len(B.spec.exclude_rows)} flagged observations the main effects are similar.",
                           "The findings describe the typical case, not just a few outliers.",
                           ["Keep the full-sample model as the reference."], tone="good")
        return Finding("A few observations matter a lot",
                       "Without the flagged observations, these effects change by more than 25%: "
                       + (", ".join(big) or "—") + (f". No longer reliable: {', '.join(nm(t, desc) for t in lost)}." if lost else "."),
                       "Conclusions about these drivers rest partly on a handful of unusual cases.",
                       ["Investigate the flagged rows (errors? distressed assets? one-off events?).",
                        "Present both versions if the unusual cases are genuine."], tone="caution")
    dropped = sorted(set(A.spec.xs) - set(B.spec.xs))
    added = sorted(set(B.spec.xs) - set(A.spec.xs))
    d_adj = B.stats["adj_r2"] - A.stats["adj_r2"]
    fit_txt = ("explains about as much" if abs(d_adj) < 0.01 else
               f"explains {'more' if d_adj > 0 else 'less'} (adjusted R² {A.stats['adj_r2']:.3f} → {B.stats['adj_r2']:.3f})")
    chg = []
    for t in common:
        ac, bc = A.coef.loc[t, "estimate"], B.coef.loc[t, "estimate"]
        if ac != 0 and abs(bc - ac) / abs(ac) > 0.15:
            chg.append(f"{nm(t, desc)} {fmt(ac)} → {fmt(bc)}")
    what = ("Removing " + ", ".join(nm(t, desc) for t in dropped) if dropped else "") + \
           (" and " if dropped and added else "") + ("adding " + ", ".join(nm(t, desc) for t in added) if added else "")
    return Finding(f"{what[:1].upper() + what[1:]}: the model {fit_txt}",
                   ("Effects that shifted: " + "; ".join(chg[:3]) + "." if chg else "Other effects barely changed."),
                   ("If fit is similar, the simpler model is usually easier to defend — but dropping a genuinely relevant "
                    "driver can distort the others, so the choice should follow investment logic, not statistics alone."),
                   ["Check the out-of-sample comparison below.", "Choose which model goes in the report."], tone="neutral")
