"""Shared UI helpers: session state, audit log, styling, small components."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.diagnostics import EXPLAIN
from core.insights import fmt, fmt_p

ROOT = Path(__file__).resolve().parent.parent
LOGO_SVG = (ROOT / "assets" / "elk_logo.svg").read_text()
POINT = "#7D8B99"
LINE = "#4A5A6A"
ACCENT = "#B07D3E"
STEPS = ["Data & Suitability", "Regression Lab", "Diagnostics", "Compare & Repair", "Insights & Export"]
STEP_ICONS = [":material/table_chart:", ":material/functions:", ":material/troubleshoot:", ":material/compare_arrows:",
              ":material/description:"]

ss = st.session_state


def init_state():
    defaults = {"step": 0, "data": None, "spec": {}, "suit": None, "fitA": None, "fitB": None, "b_cfg": None,
                "log": [], "notes": "", "excess_cols": set(), "units": {}, "report_model": "A", "oos": None}
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v


def log(action: str, detail: str = ""):
    ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "action": action, "detail": detail})


def reset_models():
    ss.fitA = None
    ss.fitB = None
    ss.b_cfg = None
    ss.oos = None


def go_to(i: int):
    ss.step = i
    st.rerun()


def css():
    st.markdown("""
<style>
.block-container{padding-top:2.2rem;max-width:1180px}
section[data-testid="stSidebar"] .stButton button{justify-content:flex-start;text-align:left;width:100%;padding-left:.8rem}
section[data-testid="stSidebar"] .stButton button div, section[data-testid="stSidebar"] .stButton button span{justify-content:flex-start}
section[data-testid="stSidebar"] .stButton .stTooltipIcon{width:100%;display:block}
.elk-brand{display:flex;align-items:center;gap:.7rem;margin:.2rem 0 1.2rem}
.elk-brand svg{width:44px;height:44px;flex:none;color:var(--elk-ink,currentColor)}
.elk-brand .t{font-weight:700;font-size:1.35rem;letter-spacing:.08em;line-height:1}
.elk-brand .s{font-size:.72rem;opacity:.65;letter-spacing:.02em;margin-top:.2rem}
.elk-chip{display:inline-block;padding:.12rem .55rem;border-radius:999px;font-size:.74rem;font-weight:600;
  border:1px solid rgba(128,128,128,.35);margin-right:.25rem;white-space:nowrap}
.elk-chip.ok{color:#4F7A5B;border-color:rgba(79,122,91,.45);background:rgba(79,122,91,.08)}
.elk-chip.warn{color:#B07D3E;border-color:rgba(176,125,62,.5);background:rgba(176,125,62,.09)}
.elk-chip.block{color:#B5473A;border-color:rgba(181,71,58,.5);background:rgba(181,71,58,.08)}
.elk-chip.na,.elk-chip.info{opacity:.75}
.elk-kind{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;font-weight:700;opacity:.7;margin-right:.4rem}
.elk-eq{font-family:'IBM Plex Mono',monospace;font-size:.86rem;opacity:.85}
.elk-hero{font-size:.95rem;opacity:.8;max-width:760px}
div[data-testid="stMetricValue"]{font-size:1.55rem}
</style>""", unsafe_allow_html=True)


def brand():
    st.markdown(f"<div class='elk-brand'>{LOGO_SVG}<div><div class='t'>ELK</div>"
                f"<div class='s'>Quantitative Model Intelligence</div></div></div>", unsafe_allow_html=True)


def chip(text, level="info"):
    return f"<span class='elk-chip {level}'>{text}</span>"


LEVEL_ICON = {"ok": ":material/check_circle:", "warn": ":material/warning:", "block": ":material/block:",
              "info": ":material/info:", "na": ":material/remove_circle_outline:"}
LEVEL_WORD = {"ok": "No issue", "warn": "Flagged", "na": "Not available", "info": "Info", "block": "Blocking"}


def explain(key: str, expanded=False):
    e = EXPLAIN[key]
    with st.expander("What does this mean?", expanded=expanded, icon=":material/help:"):
        st.markdown(f"**What it checks.** {e['what']}  \n**Null hypothesis.** {e['null']}  \n"
                    f"**How to read it.** {e['read']}  \n**Limits.** {e['limits']}")


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<div class='elk-hero'>{subtitle}</div>", unsafe_allow_html=True)
    st.write("")


def nav_buttons(back: int | None, nxt: int | None, next_label: str = "Continue", disabled=False, help=None):
    st.divider()
    c1, _, c3 = st.columns([1, 3, 1.4])
    if back is not None and c1.button("Back", icon=":material/arrow_back:", width="stretch", key=f"back{back}"):
        go_to(back)
    if nxt is not None and c3.button(next_label, icon=":material/arrow_forward:", type="primary", width="stretch",
                                     disabled=disabled, help=help, key=f"next{nxt}"):
        go_to(nxt)


def need_model() -> bool:
    if ss.fitA is None:
        st.info("Run a regression first: choose your variables in **Data & Suitability**, then open the **Regression Lab**.",
                icon=":material/info:")
        if st.button("Go to Data & Suitability", type="primary"):
            go_to(0)
        return False
    return True


def intercept_label() -> str:
    return "Alpha (α)" if ss.spec.get("alpha_ok") else "Intercept"


def coef_display(coef: pd.DataFrame, alpha: float) -> pd.DataFrame:
    d = coef.copy()
    d.index = [intercept_label() if i == "const" else i for i in d.index]
    d.index.name = "Term"
    d["significant"] = np.where(d["p_value"] < alpha, "yes", "no")
    return d


def coef_table(coef: pd.DataFrame, alpha: float):
    d = coef_display(coef, alpha)
    cl = int(round((1 - alpha) * 100))
    st.dataframe(
        d, width="stretch",
        column_config={
            "estimate": st.column_config.NumberColumn("Estimate", format="%.4f"),
            "std_error": st.column_config.NumberColumn("Std. error", format="%.4f"),
            "t_stat": st.column_config.NumberColumn("t", format="%.2f"),
            "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
            "ci_low": st.column_config.NumberColumn(f"{cl}% CI low", format="%.4f"),
            "ci_high": st.column_config.NumberColumn(f"{cl}% CI high", format="%.4f"),
            "significant": st.column_config.TextColumn(f"Significant at {alpha:.0%}"),
        })


def style_fig(fig: go.Figure, height=340, title=None, top=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top or (40 if title else 10), b=10),
                      title=dict(text=title, x=0, font=dict(size=14)) if title else None,
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key, config={"displaylogo": False})


def fit_cache(fit) -> dict:
    if not hasattr(fit, "cache"):
        fit.cache = {}
    return fit.cache


__all__ = ["ss", "fmt", "fmt_p"]
