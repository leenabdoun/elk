"""ELK — Quantitative Model Intelligence. Run with:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ELK · Quantitative Model Intelligence", page_icon=str(Path(__file__).parent / "assets" / "elk_icon.png"),
                   layout="wide", initial_sidebar_state="expanded")

from ui import page_compare, page_data, page_diagnostics, page_insights, page_regression  # noqa: E402
from ui.common import STEP_ICONS, STEPS, brand, css, init_state, ss  # noqa: E402

init_state()
css()

PAGES = [page_data.render, page_regression.render, page_diagnostics.render, page_compare.render, page_insights.render]


def unlocked(i: int) -> tuple[bool, str]:
    return True, ""


def done(i: int) -> bool:
    return [bool(ss.get("analysis_ran")) and ss.fitA is not None, ss.fitA is not None, ss.fitA is not None and ss.step > 2,
            ss.fitB is not None, False][i]


page_regression.ensure_model_a()

with st.sidebar:
    brand()
    st.caption("Analyze · Validate · Interpret")
    for i, name in enumerate(STEPS):
        ok, why = unlocked(i)
        icon = ":material/check_circle:" if done(i) and i != ss.step else STEP_ICONS[i]
        if st.button(f"{i + 1}. {name}", key=f"nav{i}", icon=icon, disabled=not ok, help=why or None,
                     type="primary" if i == ss.step else "tertiary", width="stretch"):
            ss.step = i
            st.rerun()
    st.divider()
    if ss.data is not None:
        d = ss.data
        st.caption(f"**Data:** {d['name']}")
        if ss.spec.get("y"):
            st.caption(f"**Y:** {ss.spec['y']}  \n**X:** {', '.join(ss.spec.get('xs', [])) or '—'}")
        if ss.fitA is not None:
            st.caption(f"**Model A:** R² {ss.fitA.stats['r2']:.3f} · n {ss.fitA.stats['n']}")
        if ss.fitB is not None:
            st.caption(f"**Model B:** R² {ss.fitB.stats['r2']:.3f} · n {ss.fitB.stats['n']}")
    with st.expander("About ELK", icon=":material/info:"):
        st.markdown("ELK checks whether quantitative conclusions are supported by the data. It does not make investment "
                    "decisions. Numbers come from statistical code (statsmodels); interpretations use transparent rules.\n\n"
                    "**Public demo:** do not upload confidential data.")

PAGES[ss.step]()
