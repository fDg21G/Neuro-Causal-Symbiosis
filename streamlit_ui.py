"""
Neuro-Causal Symbiosis (NCS) — Interactive Web Interface v2.0
Dark-Mode Streamlit UI with Real-Time Causal Analysis
"""

import os
import time
from typing import Optional

import streamlit as st
from streamlit.components.v1 import html
from ncs_core import NCS_Engine, CausalResult

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ─ PAGE CONFIG & THEMING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NCS Engine — Causal Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

dark_theme_css = """
<style>
:root {
    --primary-color: #00d4ff;
    --primary-dark: #0099cc;
    --secondary-color: #9d4edd;
    --bg-dark: #0a0e27;
    --surface-dark: #1a1f3a;
    --surface-lighter: #252d47;
    --text-primary: #e8f0ff;
    --text-secondary: #a8b5cc;
    --border-color: #2d3a54;
}

body, [data-testid="stAppViewContainer"] { background-color: var(--bg-dark) !important; color: var(--text-primary) !important; }
[data-testid="stSidebar"] { background-color: var(--surface-dark) !important; border-right: 2px solid var(--border-color) !important; }
body, div, p, h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, textarea { background-color: var(--surface-lighter) !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
button { background-color: var(--primary-color) !important; color: var(--bg-dark) !important; border-radius: 8px !important; font-weight: 600 !important; border: none !important; transition: all 0.3s ease !important; }
button:hover { background-color: var(--primary-dark) !important; transform: scale(1.02) !important; }
[data-testid="stContainer"], .streamlit-expander, [data-testid="stMetricValue"] { background-color: var(--surface-lighter) !important; border-radius: 12px !important; }
.streamlit-expanderContent { background-color: var(--surface-dark) !important; border: 1px solid var(--border-color) !important; }
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ─ HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_confidence_bar(confidence: float) -> str:
    return f"""
    <div style="display: flex; align-items: center; gap: 12px; margin: 12px 0;">
        <div style="font-weight: 700; color: #00d4ff; font-size: 14px;">CONFIDENCE</div>
        <div style="flex: 1; background: #1a1f3a; border: 1px solid #2d3a54; border-radius: 4px; height: 24px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #00ff88, #00d4ff); width: {confidence*100}%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; color: #0a0e27;">
                {confidence:.1%}
            </div>
        </div>
    </div>
    """

def render_system_badge(system_used: int) -> str:
    if system_used == 1:
        return """<div style="display: inline-block; padding: 8px 16px; border-radius: 20px; background: rgba(0, 255, 136, 0.1); border: 2px solid #00ff88; color: #00ff88; font-weight: 700; font-size: 13px;">⚡ SYSTEM 1: EMPIRICAL REFLEX CORE</div>"""
    return """<div style="display: inline-block; padding: 8px 16px; border-radius: 20px; background: rgba(157, 78, 221, 0.1); border: 2px solid #9d4edd; color: #9d4edd; font-weight: 700; font-size: 13px;">🧠 SYSTEM 2: SEMANTIC CLOUD (LLM)</div>"""

def render_result_card(result: CausalResult) -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div style='text-align: center;'><div style='font-size: 32px; color: #00d4ff;'>⚗️</div><div style='color: #a8b5cc; font-size: 12px;'>CAUSAL ANALYSIS</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(render_system_badge(result.system_used), unsafe_allow_html=True)
        st.markdown(f"<div style='color: #e8f0ff; font-size: 20px; font-weight: 700; margin: 12px 0;'>{result.entity_a.title()} <span style='color: #00d4ff;'>→</span> {result.entity_b.title()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #a8b5cc; font-size: 14px; font-style: italic;'>{result.direction}</div>", unsafe_allow_html=True)
    with col3:
        conf_color = "#00ff88" if result.confidence >= 0.80 else "#ffaa00" if result.confidence >= 0.60 else "#ff3366"
        st.markdown(f"<div style='text-align: right;'><div style='color: {conf_color}; font-size: 36px; font-weight: 700;'>{result.confidence:.0%}</div><div style='color: #a8b5cc; font-size: 12px;'>CONFIDENCE</div></div>", unsafe_allow_html=True)
    
    st.markdown(render_confidence_bar(result.confidence), unsafe_allow_html=True)
    st.markdown(f"<div style='display: flex; gap: 20px; margin-top: 16px; padding: 12px; background: rgba(45, 58, 84, 0.5); border-radius: 8px;'><div><div style='color: #a8b5cc; font-size: 11px; text-transform: uppercase;'>Data Source</div><div style='color: #00d4ff; font-size: 13px; font-weight: 600;'>{result.data_source}</div></div><div><div style='color: #a8b5cc; font-size: 11px; text-transform: uppercase;'>Method</div><div style='color: #00d4ff; font-size: 13px; font-weight: 600;'>{result.method}</div></div></div>", unsafe_allow_html=True)
    
    if result.time_range:
        st.markdown(f"<div style='color: #a8b5cc; font-size: 12px; margin-top: 12px;'><strong>Time Range:</strong> {result.time_range}</div>", unsafe_allow_html=True)
    if result.explanation:
        with st.expander("📖 Explanation"):
            st.markdown(f"<div style='color: #e8f0ff; line-height: 1.6;'>{result.explanation}</div>", unsafe_allow_html=True)
    if result.warnings:
        with st.expander("⚠️ Warnings"):
            for warning in result.warnings:
                st.warning(warning)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ─ PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""<div style='text-align: center; margin-bottom: 40px;'><h1 style='color: #00d4ff; font-size: 48px; margin: 0;'>⚗️ NEURO-CAUSAL SYMBIOSIS</h1><p style='color: #a8b5cc;'>Empirical Causal Analysis meets Semantic Intelligence</p></div>""", unsafe_allow_html=True)

col_input, col_button = st.columns([4, 1])
with col_input:
    query = st.text_input(label="Ask a causal question", placeholder="e.g., 'Does inflation cause interest rates to rise?'", key="query_input", label_visibility="collapsed")
with col_button:
    analyze_button = st.button("🔍 Analyze", use_container_width=True)

with st.expander("💡 Example Queries"):
    example_queries = ["Does inflation cause interest rates to rise?", "Do oil prices affect producer prices?", "Does freedom lead to happiness?", "Can employment growth reduce unemployment?"]
    col1, col2 = st.columns(2)
    for i, eq in enumerate(example_queries):
        with col1 if i % 2 == 0 else col2:
            if st.button(f"📋 {eq[:45]}...", use_container_width=True, key=f"example_{i}"):
                st.session_state.query_input = eq
                st.rerun()

if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "engine" not in st.session_state:
    llm_provider = st.sidebar.selectbox("LLM Provider for System 2", ["anthropic", "openai"], index=0)
    llm_api_key = st.sidebar.text_input(f"{llm_provider.upper()} API Key", type="password")
    st.session_state.engine = NCS_Engine(llm_provider=llm_provider, llm_api_key=llm_api_key if llm_api_key else None, confidence_threshold=st.sidebar.slider("System 1 Threshold", 0.5, 0.95, 0.80, 0.05))

if analyze_button and query:
    with st.spinner("⚙️ Analyzing query..."):
        try:
            result = st.session_state.engine.analyze(query)
            st.session_state.results_history.insert(0, result)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

if st.session_state.results_history:
    for i, result in enumerate(st.session_state.results_history):
        with st.container():
            render_result_card(result)
            with st.expander("📄 JSON Export"):
                st.json(result.to_dict())
            st.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📚 About NCS")
    st.info("System 1 (⚡): Real FRED time-series.\nSystem 2 (🧠): LLM semantic reasoning.")
    if st.session_state.results_history:
        if st.button("🗑️ Clear History"):
            st.session_state.results_history = []
            st.rerun()