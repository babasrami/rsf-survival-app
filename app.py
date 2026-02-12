import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="RSF Survival Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS — Premium Light Theme with 3D & Animations
# ---------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* =============================================
       CSS VARIABLES — LIGHT THEME
    ============================================= */
    :root {
        --bg-body: #f0f2f8;
        --bg-white: #ffffff;
        --bg-card: #ffffff;
        --bg-card-alt: #f8f9fc;
        --bg-sidebar: linear-gradient(180deg, #1e1b4b 0%, #312e81 50%, #3730a3 100%);
        --border-light: rgba(0,0,0,0.06);
        --border-card: rgba(0,0,0,0.08);
        --border-accent: rgba(124,58,237,0.25);
        --purple-50: #f5f3ff;
        --purple-100: #ede9fe;
        --purple-200: #ddd6fe;
        --purple-300: #c4b5fd;
        --purple-400: #a78bfa;
        --purple-500: #8b5cf6;
        --purple-600: #7c3aed;
        --purple-700: #6d28d9;
        --purple-800: #5b21b6;
        --purple-900: #4c1d95;
        --indigo-50: #eef2ff;
        --indigo-600: #4f46e5;
        --cyan-400: #22d3ee;
        --cyan-500: #06b6d4;
        --emerald-500: #10b981;
        --amber-500: #f59e0b;
        --rose-500: #f43f5e;
        --text-dark: #1e1b4b;
        --text-body: #374151;
        --text-muted: #6b7280;
        --text-light: #9ca3af;
        --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.04);
        --shadow-3d: 0 8px 30px rgba(124,58,237,0.10), 0 2px 8px rgba(0,0,0,0.06);
        --shadow-3d-hover: 0 20px 40px rgba(124,58,237,0.16), 0 8px 16px rgba(0,0,0,0.08);
        --shadow-glow: 0 0 30px rgba(124,58,237,0.12);
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 18px;
        --radius-xl: 24px;
    }

    /* =============================================
       GLOBAL
    ============================================= */
    .stApp,
    .stApp > header,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--bg-body) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp {
        background: linear-gradient(145deg, #f0f2f8 0%, #e8eaf5 35%, #f0f2f8 65%, #eee8fa 100%) !important;
        background-attachment: fixed !important;
    }

    /* =============================================
       MAIN CONTENT
    ============================================= */
    .stMainBlockContainer, .main .block-container,
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        max-width: 1100px;
        padding-top: 1.5rem !important;
    }

    /* =============================================
       SIDEBAR — Deep Indigo Gradient
    ============================================= */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: var(--bg-sidebar) !important;
        border-right: none !important;
        box-shadow: 4px 0 20px rgba(30,27,75,0.15);
    }
    section[data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: none !important;
    }
    section[data-testid="stSidebar"] code {
        background: rgba(255,255,255,0.12) !important;
        color: var(--purple-200) !important;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 0.82rem;
    }
    section[data-testid="stSidebar"] .stDivider,
    section[data-testid="stSidebar"] hr {
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #fff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div:hover,
    section[data-testid="stSidebar"] .stMultiSelect > div > div:hover {
        border-color: rgba(255,255,255,0.35) !important;
        background: rgba(255,255,255,0.12) !important;
    }

    /* =============================================
       TYPOGRAPHY
    ============================================= */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: var(--text-dark) !important;
        font-weight: 700 !important;
    }
    h1 {
        font-size: 2.6rem !important;
        letter-spacing: -0.8px !important;
        background: linear-gradient(135deg, var(--purple-900) 0%, var(--purple-600) 40%, var(--indigo-600) 70%, var(--cyan-500) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: gradientShift 6s ease-in-out infinite;
        background-size: 200% 200% !important;
    }
    h2 {
        font-size: 1.4rem !important;
        letter-spacing: -0.3px !important;
        color: var(--text-dark) !important;
    }
    h3 {
        font-size: 1.1rem !important;
        color: var(--text-body) !important;
    }
    p, li, span, div, label {
        color: var(--text-body) !important;
    }
    a { color: var(--purple-600) !important; text-decoration: none !important; }
    a:hover { color: var(--purple-700) !important; }
    small, .stCaption, caption {
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
    }

    /* =============================================
       ANIMATIONS
    ============================================= */
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes float3d {
        0%, 100% { transform: translateY(0) perspective(800px) rotateX(0deg); }
        50% { transform: translateY(-6px) perspective(800px) rotateX(1deg); }
    }
    @keyframes cardEntrance {
        from { opacity: 0; transform: translateY(20px) scale(0.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: var(--shadow-3d); }
        50% { box-shadow: var(--shadow-3d-hover), 0 0 20px rgba(124,58,237,0.08); }
    }

    /* =============================================
       DIVIDERS
    ============================================= */
    hr, .stDivider {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent 0%, var(--border-accent) 50%, transparent 100%) !important;
        margin: 2rem 0 !important;
    }

    /* =============================================
       BUTTONS — 3D Elevated
    ============================================= */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--purple-600) 0%, var(--purple-700) 50%, var(--indigo-600) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.7rem 1.8rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.2px;
        box-shadow:
            0 4px 14px rgba(124,58,237,0.30),
            0 2px 4px rgba(0,0,0,0.08),
            inset 0 1px 0 rgba(255,255,255,0.15);
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::after,
    .stDownloadButton > button::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 200%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        transition: left 0.5s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow:
            0 12px 28px rgba(124,58,237,0.35),
            0 6px 12px rgba(0,0,0,0.10),
            inset 0 1px 0 rgba(255,255,255,0.2);
    }
    .stButton > button:hover::after,
    .stDownloadButton > button:hover::after {
        left: 100%;
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(-1px) scale(0.99);
        box-shadow:
            0 4px 10px rgba(124,58,237,0.25),
            0 1px 3px rgba(0,0,0,0.08),
            inset 0 2px 4px rgba(0,0,0,0.1);
    }

    /* =============================================
       FILE UPLOADERS — Glass Card
    ============================================= */
    div[data-testid="stFileUploader"] {
        background: var(--bg-white) !important;
        border: 2px dashed rgba(124,58,237,0.20) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.3rem !important;
        box-shadow: var(--shadow-md);
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--purple-400) !important;
        box-shadow: var(--shadow-3d-hover);
        transform: translateY(-3px);
    }
    div[data-testid="stFileUploader"] label {
        color: var(--text-dark) !important;
        font-weight: 600;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button {
        background: var(--purple-50) !important;
        border: 1px solid var(--border-accent) !important;
        color: var(--purple-700) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
    }

    /* =============================================
       SELECT / MULTISELECT / INPUT
    ============================================= */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-white) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-dark) !important;
        box-shadow: var(--shadow-xs);
        transition: all 0.25s ease;
    }
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: var(--purple-400) !important;
        box-shadow: var(--shadow-sm), 0 0 0 3px rgba(124,58,237,0.06);
    }
    .stSelectbox label,
    .stMultiSelect label {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }
    .stRadio > div {
        background: var(--bg-white) !important;
        border-radius: var(--radius-md);
        padding: 0.5rem 0.8rem;
        border: 1px solid var(--border-card);
        box-shadow: var(--shadow-xs);
    }
    .stRadio label span { color: var(--text-body) !important; }
    .stToggle label span { color: var(--text-dark) !important; }

    /* =============================================
       DATA TABLE / DATAFRAME — 3D Card
    ============================================= */
    .stDataFrame, div[data-testid="stDataFrame"] {
        background: var(--bg-white) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-3d);
        transition: all 0.35s ease;
    }
    .stDataFrame:hover, div[data-testid="stDataFrame"]:hover {
        box-shadow: var(--shadow-3d-hover);
        transform: translateY(-2px);
    }
    .stDataFrame th {
        background: linear-gradient(135deg, var(--purple-50) 0%, var(--indigo-50) 100%) !important;
        color: var(--purple-700) !important;
        font-weight: 700 !important;
        border-bottom: 2px solid rgba(124,58,237,0.15) !important;
    }
    .stDataFrame td {
        background: transparent !important;
        color: var(--text-body) !important;
        border-bottom: 1px solid var(--border-light) !important;
    }
    .stDataFrame tr:hover td {
        background: rgba(124,58,237,0.03) !important;
    }

    /* =============================================
       METRIC CARDS — 3D Animated
    ============================================= */
    div[data-testid="stMetric"] {
        background: var(--bg-white) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.2rem 1.4rem !important;
        box-shadow: var(--shadow-3d);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: cardEntrance 0.5s ease forwards;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--purple-500), var(--indigo-600), var(--cyan-400));
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02) perspective(800px) rotateX(2deg);
        box-shadow: var(--shadow-3d-hover);
        border-color: var(--border-accent) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-dark) !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* =============================================
       EXPANDER — 3D Card
    ============================================= */
    .streamlit-expanderHeader,
    details > summary {
        background: var(--bg-white) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-dark) !important;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    .streamlit-expanderHeader:hover,
    details > summary:hover {
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        transform: translateY(-2px);
        border-color: var(--border-accent) !important;
    }
    .streamlit-expanderContent,
    details > div {
        background: var(--bg-card-alt) !important;
        border: 1px solid var(--border-card) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }

    /* =============================================
       ALERTS
    ============================================= */
    .stAlert, div[data-testid="stAlert"] {
        background: var(--bg-white) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stAlert"] > div,
    div[role="alert"] {
        color: var(--text-body) !important;
    }

    /* =============================================
       PLOT CONTAINERS — 3D Float
    ============================================= */
    div.stPyplot,
    div[data-testid="stPlotlyChart"],
    .stPlotlyChart {
        background: var(--bg-white) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-3d);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    div.stPyplot:hover {
        box-shadow: var(--shadow-3d-hover);
        transform: translateY(-4px);
    }

    /* =============================================
       TABLE (st.table)
    ============================================= */
    .stTable {
        background: var(--bg-white) !important;
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    .stTable th {
        background: var(--purple-50) !important;
        color: var(--purple-700) !important;
    }
    .stTable td {
        color: var(--text-body) !important;
        border-color: var(--border-light) !important;
    }

    /* =============================================
       SCROLLBAR — Subtle
    ============================================= */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #f0f2f8; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--purple-400), var(--purple-600));
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--purple-500), var(--purple-700));
    }

    /* =============================================
       CUSTOM UTILITY CLASSES
    ============================================= */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--purple-50), var(--indigo-50));
        border: 1px solid rgba(124,58,237,0.15);
        border-radius: 50px;
        padding: 7px 22px;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--purple-700) !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        box-shadow: var(--shadow-sm);
        animation: float3d 4s ease-in-out infinite;
    }
    .hero-subtitle {
        color: var(--text-muted) !important;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.65;
    }
    .glass-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-card);
        border-radius: var(--radius-xl);
        padding: 1.6rem 1.8rem;
        box-shadow: var(--shadow-3d);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: cardEntrance 0.6s ease forwards;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--purple-400), var(--indigo-600), var(--cyan-400));
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }
    .glass-card:hover {
        transform: translateY(-4px) perspective(1000px) rotateX(1deg);
        box-shadow: var(--shadow-3d-hover);
        border-color: var(--border-accent);
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    .section-icon {
        width: 38px; height: 38px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, var(--purple-100), var(--indigo-50));
        border-radius: 10px;
        font-size: 1.15rem;
        box-shadow: var(--shadow-sm);
    }
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-dark) !important;
        margin: 0;
    }
    .section-desc {
        font-size: 0.87rem;
        color: var(--text-muted) !important;
        margin: 4px 0 0;
        padding-left: 48px;
    }
    .results-header {
        text-align: center;
        margin: 0.5rem 0 1.25rem;
    }
    .results-header h2 {
        display: inline-block;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.6rem !important;
        background: linear-gradient(135deg, var(--purple-800) 0%, var(--purple-600) 40%, var(--cyan-500) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 0 !important;
    }
    .feature-chip {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 4px 11px;
        font-size: 0.76rem;
        color: var(--purple-100) !important;
        margin: 3px 3px;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-weight: 500;
        backdrop-filter: blur(4px);
        transition: all 0.2s ease;
    }
    .feature-chip:hover {
        background: rgba(255,255,255,0.35);
        transform: translateY(-1px);
    }
    .footer-bar {
        text-align: center;
        padding: 2rem 0 1rem;
        font-size: 0.82rem;
        color: var(--text-light) !important;
        border-top: 1px solid var(--border-light);
        margin-top: 2.5rem;
    }
    .footer-bar span {
        color: var(--text-light) !important;
    }
    .footer-bar .footer-dot {
        display: inline-block;
        width: 4px; height: 4px;
        background: var(--purple-400);
        border-radius: 50%;
        vertical-align: middle;
        margin: 0 10px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------
# Helpers
# ---------------------------
ROMAN_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

def parse_stage_ordinal(stage_val) -> float:
    """Extract roman numeral from strings like 'Stage II' and map to ordinal."""
    if stage_val is None or (isinstance(stage_val, float) and np.isnan(stage_val)):
        return 0.0
    s = str(stage_val)
    m = re.search(r"Stage\s*([IVX]+)", s, flags=re.IGNORECASE)
    if not m:
        # also accept plain roman numeral or digits
        s2 = s.strip().upper()
        if s2 in ROMAN_MAP:
            return float(ROMAN_MAP[s2])
        try:
            return float(s2)
        except Exception:
            return 0.0
    roman = m.group(1).upper()
    return float(ROMAN_MAP.get(roman, 0.0))


def extract_features(bundle: dict) -> list:
    """Return the feature column list from different possible bundle key names."""
    if not isinstance(bundle, dict):
        return []
    # Common key names
    for k in ["features", "feature_cols", "feature_columns", "X_columns", "columns", "ptpm_cols"]:
        v = bundle.get(k, None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return list(v)
    # Nested preprocess dict (some notebooks save it there)
    pre = bundle.get("preprocess", None)
    if isinstance(pre, dict):
        for k in ["features", "feature_cols", "feature_columns"]:
            v = pre.get(k, None)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return list(v)
    return []

def extract_scaler(bundle: dict):
    if not isinstance(bundle, dict):
        return None
    if bundle.get("scaler", None) is not None:
        return bundle.get("scaler")
    pre = bundle.get("preprocess", None)
    if isinstance(pre, dict):
        return pre.get("scaler", None)
    return None

def extract_scaler_cols(bundle: dict, default_cols: list):
    if not isinstance(bundle, dict):
        return list(default_cols)
    v = bundle.get("scaler_cols", None)
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return list(v)
    pre = bundle.get("preprocess", None)
    if isinstance(pre, dict):
        v = pre.get("scaler_cols", None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return list(v)
    return list(default_cols)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names lightly (strip) without changing meaning."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def read_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Unsupported file type. Please upload .csv or .xlsx")

def compute_time_event(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'time' and 'event' fields expected by the training notebook, if possible."""
    df = df.copy()
    if "event" not in df.columns:
        if "status" in df.columns:
            df["event"] = pd.to_numeric(df["status"], errors="coerce")
        else:
            df["event"] = np.nan
    if "time" not in df.columns:
        if "days" in df.columns:
            df["time"] = pd.to_numeric(df["days"], errors="coerce")
        else:
            df["time"] = np.nan
    return df

def preprocess_for_model(df: pd.DataFrame, bundle: dict, input_is_raw_ptpm: bool) -> pd.DataFrame:
    """
    Prepare X matrix in the exact feature order expected by the RSF model bundle.
    Handles:
      - stage_ordinal creation (from stage)
      - optional raw pTPM transform (log1p + scaler if present in bundle)
      - median imputation for missing values using bundle["feature_medians"]
    """
    df = ensure_columns(df)
    df = compute_time_event(df)

    # stage_ordinal
    if "stage_ordinal" not in df.columns:
        if "stage" in df.columns:
            df["stage_ordinal"] = df["stage"].apply(parse_stage_ordinal)
        else:
            df["stage_ordinal"] = 0.0

    features = extract_features(bundle)
    if not features:
        raise ValueError("Model bundle is missing the feature list. Please re-save the bundle to include one of: 'features' or 'feature_cols'.")

    protein_cols = [c for c in df.columns if str(c).endswith("_pTPM")]
    scaler = extract_scaler(bundle)
    scaler_cols = extract_scaler_cols(bundle, protein_cols)

    if input_is_raw_ptpm and protein_cols:
        for col in protein_cols:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0.0))

        if scaler is not None:
            try:
                use_cols = [c for c in scaler_cols if c in df.columns]
                if use_cols:
                    df.loc[:, use_cols] = scaler.transform(df[use_cols].astype(float).values)
            except Exception as e:
                st.warning(f"Scaler found in bundle, but transform failed: {e}. Proceeding without scaling.")
        else:
            st.warning(
                "This model bundle does not include a saved scaler. "
                "Your model was trained on log1p + z-scored proteins. "
                "For best fidelity, re-save the bundle with the fitted scaler, or upload inputs already normalized."
            )

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            df[c] = np.nan
        st.warning(
            "Input file is missing some model features. They will be treated as NA and imputed where possible. "
            f"Missing (first 10): {missing_cols[:10]}"
        )

    X = df.reindex(columns=features).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    med = bundle.get("feature_medians", {}) or {}
    if med:
        for col in X.columns:
            if col in med:
                X[col] = X[col].fillna(med[col])

    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(0.0)

    return X

def survival_prob_at_times(step_fn, times):
    return [float(step_fn(t)) for t in times]

def classify_risk(score: float, risk_ref: dict):
    if not risk_ref:
        return None, None
    q33 = risk_ref.get("q33", None)
    q66 = risk_ref.get("q66", None)
    if q33 is None or q66 is None:
        return None, None
    if score <= q33:
        return "Low", q33
    if score <= q66:
        return "Intermediate", q66
    return "High", q66

def get_risk_color(risk_group):
    """Return color based on risk group for styled metrics."""
    colors = {
        "Low": "#10b981",
        "Intermediate": "#f59e0b",
        "High": "#f43f5e"
    }
    return colors.get(risk_group, "#06b6d4")

# ---------------------------
# UI
# ---------------------------

# ═══════════════ HERO SECTION ═══════════════
st.markdown("""
<div class="hero-container">
    <span class="hero-badge">🧬 &nbsp;MACHINE LEARNING &nbsp;•&nbsp; SURVIVAL ANALYSIS</span>
</div>
""", unsafe_allow_html=True)

st.title("Random Survival Forest — Survival Prediction")

st.markdown("""
<p class="hero-subtitle" style="text-align:center; margin-top:-6px; margin-bottom:2rem;">
    Upload your trained model bundle and patient data to generate individualized
    survival probability curves and risk stratification.
</p>
""", unsafe_allow_html=True)

# ═══════════════ UPLOAD INFO CARD ═══════════════
st.markdown("""
<div class="glass-card">
    <div class="section-header">
        <div class="section-icon">📋</div>
        <p class="section-title">What You Need to Upload</p>
    </div>
    <p class="section-desc">Provide a trained model bundle and patient data file to get started with predictions.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("View detailed upload requirements", expanded=False):
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        **🔮 Model Bundle** (.joblib)
        - `model`: Fitted RSF model
        - `features`: Feature names list
        - `feature_medians`: For imputation
        - `risk_ref`: Risk quantiles (optional)
        - `scaler`: Z-score scaler (optional)
        """)
    with col_info2:
        st.markdown("""
        **📊 Patient Data** (.xlsx/.csv)
        - One row per patient
        - Matches training schema
        - Missing values as NA/blank
        - Protein expression columns (_pTPM)
        """)

st.markdown("<br>", unsafe_allow_html=True)

# File uploaders
colA, colB = st.columns(2)
with colA:
    st.markdown("##### 🔮 Model Bundle")
    model_file = st.file_uploader("Upload model bundle", type=["joblib"], accept_multiple_files=False, label_visibility="collapsed")
with colB:
    st.markdown("##### 📊 Patient Data")
    data_file = st.file_uploader("Upload patient data", type=["xlsx", "xls", "csv"], accept_multiple_files=False, label_visibility="collapsed")

if not model_file:
    st.info("👆 Upload a model bundle to get started")
    st.stop()

# Load model bundle
try:
    bundle = joblib.load(model_file)
except Exception as e:
    st.error(f"Could not load model bundle: {e}")
    st.stop()

if not isinstance(bundle, dict) or "model" not in bundle:
    st.error("The uploaded .joblib file is not a valid model bundle dictionary (missing key 'model').")
    st.stop()

model = bundle["model"]
features = extract_features(bundle)
risk_ref = bundle.get("risk_ref", {}) or {}

# ═══════════════ SIDEBAR ═══════════════
st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0 18px;">
    <span style="font-size:1.8rem;">⚙️</span>
    <h2 style="margin:4px 0 0; font-size:1.15rem; font-weight:700;">Inference Options</h2>
</div>
""", unsafe_allow_html=True)

input_is_raw = st.sidebar.toggle("🧪 Raw pTPM inputs (apply log1p + scaling)", value=True)
timepoints_years = st.sidebar.multiselect(
    "📅 Survival timepoints (years):",
    options=[1, 2, 3, 5, 10],
    default=[1, 2, 3, 5]
)
timepoints_days = [y * 365.25 for y in timepoints_years]

st.sidebar.divider()

# ── Features expected ──
EXPECTED_FEATURES = [
    "ADAM15", "ADAMTS8", "MMP7", "MMP15", "ADAMTSL1", "MMP13",
    "MMP1", "MMP12", "MMP23", "MMP26", "ADAMTS7", "MMP28",
    "MMP9", "MMP25"
]

st.sidebar.markdown("""
<div style="text-align:center;">
    <span style="font-size:1.3rem;">📦</span>
    <h3 style="margin:4px 0 8px; font-size:1rem; font-weight:700;">Features Expected</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"**Total features:** `{len(features)}`")

# Display the known feature chips
chips_html = "".join([f'<span class="feature-chip">{f}</span>' for f in EXPECTED_FEATURES])
st.sidebar.markdown(f'<div style="margin-top:6px;">{chips_html}</div>', unsafe_allow_html=True)

if features:
    with st.sidebar.expander("All bundle features", expanded=False):
        feature_preview = "\n".join(features[:30])
        if len(features) > 30:
            feature_preview += f"\n... (+{len(features) - 30} more)"
        st.code(feature_preview, language=None)

if not data_file:
    st.info("📊 Upload a patient data file to proceed with predictions")
    st.stop()

# Read data
try:
    df_in = read_table(data_file)
    df_in = ensure_columns(df_in)
except Exception as e:
    st.error(f"Could not read patient data: {e}")
    st.stop()

# Choose patient id column
id_candidates = [c for c in ["Sample", "Patient_ID", "patient_id", "id"] if c in df_in.columns]
id_col = id_candidates[0] if id_candidates else None

# Initialize session state for predictions
if "_predictions" not in st.session_state:
    st.session_state["_predictions"] = None

# ═══════════════ DATA EDITOR + RUN BUTTON ═══════════════
left, right = st.columns([1, 1])

with left:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">✏️</div>
        <p class="section-title">Input Data</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Edit values directly below. Use NA or blank for missing values.")
    df_edit = st.data_editor(df_in, num_rows="dynamic", use_container_width=True, key="data_editor")

with right:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🚀</div>
        <p class="section-title">Run Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    if id_col:
        st.caption(f"Patient identifier: `{id_col}`")
    else:
        st.caption("No patient ID column detected — using row index")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔬 Predict Survival", type="primary", use_container_width=True)

# Process predictions when button is clicked
if run_btn:
    try:
        X = preprocess_for_model(df_edit, bundle=bundle, input_is_raw_ptpm=input_is_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.session_state["_predictions"] = None
    else:
        try:
            surv_funcs = model.predict_survival_function(X.values, return_array=False)
            risk_scores = model.predict(X.values)
            st.session_state["_predictions"] = {
                "X": X,
                "surv_funcs": surv_funcs,
                "risk_scores": risk_scores,
                "df": df_edit.copy(),
                "id_col": id_col,
            }
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.session_state["_predictions"] = None

# Get predictions from session state
pred = st.session_state.get("_predictions")

if pred is None:
    st.info("📊 Upload data, edit if needed, then click **Predict Survival** to see results.")
    st.stop()

# Extract stored predictions
X = pred["X"]
surv_funcs = pred["surv_funcs"]
risk_scores = pred["risk_scores"]
df_used = pred["df"]
id_col = pred["id_col"]

# Results table
rows = []
for i in range(len(X)):
    pid = df_used.iloc[i][id_col] if id_col else i
    sf = surv_funcs[i]
    probs = survival_prob_at_times(sf, timepoints_days) if timepoints_days else []
    risk = float(risk_scores[i])
    risk_group, _ = classify_risk(risk, risk_ref)
    row = {"Patient": pid, "Risk_Score": round(risk, 4), "Risk_Group": risk_group if risk_group else "—"}
    for y, p in zip(timepoints_years, probs):
        row[f"S(t={y}y)"] = round(p, 3)
    rows.append(row)

res_df = pd.DataFrame(rows)

st.divider()

# ═══════════════ RESULTS ═══════════════
st.markdown("""
<div class="results-header">
    <h2>📈 &nbsp;Prediction Results</h2>
</div>
""", unsafe_allow_html=True)

st.dataframe(res_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════ SURVIVAL CURVES ═══════════════
st.markdown("""
<div class="section-header">
    <div class="section-icon">📊</div>
    <p class="section-title">Survival Curve Analysis</p>
</div>
""", unsafe_allow_html=True)

# Plot mode selection
plot_mode = st.radio(
    "Plot mode:",
    options=["Single Patient", "All Patients"],
    horizontal=True,
    key="plot_mode_selector"
)

PATIENT_COLORS = [
    '#a78bfa', '#7c3aed', '#c4b5fd', '#6d28d9', '#ddd6fe',
    '#b0b0b0', '#808080', '#e0e0e0', '#5b21b6', '#8b5cf6',
    '#f8f8f8', '#4c1d95', '#ede9fe', '#606060', '#9333ea',
    '#d8d8d8', '#7e22ce', '#a3a3a3', '#c084fc', '#404040'
]

if plot_mode == "Single Patient":
    sel_options = list(range(len(X)))
    if id_col:
        sel_label = df_used[id_col].astype(str).tolist()
        sel = st.selectbox(
            "Select patient for detailed analysis:",
            options=sel_options,
            format_func=lambda i: f"🧬 {sel_label[i]}",
            key="patient_selector"
        )
    else:
        sel = st.selectbox(
            "Select patient (row index):",
            options=sel_options,
            format_func=lambda i: f"🧬 Patient {i}",
            key="patient_selector"
        )

    sf = surv_funcs[sel]
    try:
        xs = sf.x
        ys = sf.y
    except Exception:
        xs = np.linspace(0, np.nanmax(df_used.get("days", pd.Series([3650]))), 200)
        ys = np.array([sf(t) for t in xs])

    # Modern light-theme matplotlib plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafbfe')

    # Gradient fill under curve
    ax.fill_between(xs, ys, alpha=0.12, color='#7c3aed', step='post')
    ax.step(xs, ys, where="post", color='#7c3aed', linewidth=2.5, label='Survival Probability')
    # Soft glow
    ax.step(xs, ys, where="post", color='#a78bfa', linewidth=6, alpha=0.12)

    ax.set_xlabel("Time (days)", fontsize=12, color='#374151', fontweight='600', labelpad=10)
    ax.set_ylabel("Survival Probability", fontsize=12, color='#374151', fontweight='600', labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(xs) if len(xs) > 0 else 3650)

    ax.grid(True, alpha=0.25, color='#d1d5db', linestyle='-', linewidth=0.5)
    ax.spines['bottom'].set_color('#d1d5db')
    ax.spines['left'].set_color('#d1d5db')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#6b7280', which='both', labelsize=10)

    # Timepoint markers
    for y, t in zip(timepoints_years, timepoints_days):
        if t <= max(xs):
            prob = float(sf(t))
            ax.axvline(x=t, color='#ddd6fe', linestyle='--', alpha=0.8, linewidth=1)
            ax.scatter([t], [prob], color='#7c3aed', s=70, zorder=5, edgecolors='white', linewidths=2.5)
            ax.annotate(f'{y}y: {prob:.1%}', (t, prob), textcoords="offset points",
                        xytext=(12, 12), fontsize=9, color='#1e1b4b', fontweight='600',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#c4b5fd', alpha=0.95,
                                  boxshadow='2px 2px 6px rgba(0,0,0,0.08)'))

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Detail panel
    st.markdown("""
    <div class="section-header" style="margin-top:1.25rem;">
        <div class="section-icon">🎯</div>
        <p class="section-title">Selected Patient Details</p>
    </div>
    """, unsafe_allow_html=True)

    pid = df_used.iloc[sel][id_col] if id_col else sel
    risk = float(risk_scores[sel])
    risk_group, _ = classify_risk(risk, risk_ref)

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.metric("🧬 Patient", str(pid))
    with detail_cols[1]:
        st.metric("⚡ Risk Score", f"{risk:.4f}")
    with detail_cols[2]:
        risk_display = risk_group if risk_group else "—"
        st.metric("🎯 Risk Group", risk_display)

    if timepoints_years:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📅</div>
            <p class="section-title">Survival Probabilities at Key Timepoints</p>
        </div>
        """, unsafe_allow_html=True)
        probs = survival_prob_at_times(sf, timepoints_days)

        prob_cols = st.columns(len(timepoints_years))
        for i, (y, p) in enumerate(zip(timepoints_years, probs)):
            with prob_cols[i]:
                if p >= 0.7:
                    emoji = "🟢"
                elif p >= 0.4:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                st.metric(f"{emoji} Year {y}", f"{p:.1%}")

else:
    # All Patients plot
    st.caption("Comparing survival curves for all patients with color-coded legend")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafbfe')

    max_time = 3650

    import matplotlib.cm as cm
    num_patients = len(X)
    colormap = cm.get_cmap('gist_rainbow', num_patients)

    for i in range(num_patients):
        sf = surv_funcs[i]
        color = colormap(i / max(num_patients - 1, 1))

        if id_col:
            label = str(df_used.iloc[i][id_col])
        else:
            label = f"Patient {i}"

        try:
            xs = sf.x
            ys = sf.y
        except Exception:
            xs = np.linspace(0, 3650, 200)
            ys = np.array([sf(t) for t in xs])

        if len(xs) > 0:
            max_time = max(max_time, max(xs))

        ax.step(xs, ys, where="post", color=color, linewidth=4, alpha=0.12)  # Glow
        ax.step(xs, ys, where="post", color=color, linewidth=2, label=label, alpha=0.9)

    ax.set_xlabel("Time (days)", fontsize=12, color='#374151', fontweight='600', labelpad=10)
    ax.set_ylabel("Survival Probability", fontsize=12, color='#374151', fontweight='600', labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_time)

    ax.grid(True, alpha=0.25, color='#d1d5db', linestyle='-', linewidth=0.5)
    ax.spines['bottom'].set_color('#d1d5db')
    ax.spines['left'].set_color('#d1d5db')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#6b7280', which='both', labelsize=10)

    legend = ax.legend(
        loc='upper right',
        fontsize=9,
        framealpha=0.95,
        facecolor='white',
        edgecolor='#e5e7eb',
        labelcolor='#374151',
        title='Patients',
        title_fontsize=10,
        ncol=min(3, (len(X) + 9) // 10),
        shadow=True
    )
    legend.get_title().set_color('#1e1b4b')
    legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Summary statistics
    st.markdown("""
    <div class="section-header" style="margin-top:1.25rem;">
        <div class="section-icon">📊</div>
        <p class="section-title">Summary Statistics</p>
    </div>
    """, unsafe_allow_html=True)

    avg_risk = np.mean(risk_scores)
    min_risk = np.min(risk_scores)
    max_risk = np.max(risk_scores)

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("👥 Total Patients", len(X))
    with stat_cols[1]:
        st.metric("📉 Min Risk Score", f"{min_risk:.4f}")
    with stat_cols[2]:
        st.metric("📈 Max Risk Score", f"{max_risk:.4f}")
    with stat_cols[3]:
        st.metric("📊 Avg Risk Score", f"{avg_risk:.4f}")

# ═══════════════ DOWNLOAD ═══════════════
st.markdown("<br>", unsafe_allow_html=True)
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv_data = res_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Predictions (CSV)",
        data=csv_data,
        file_name="survival_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
with col_dl2:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        "📥 Download Plot (PNG)",
        data=buf,
        file_name="survival_curves.png",
        mime="image/png",
        use_container_width=True
    )

# ═══════════════ FOOTER ═══════════════
st.markdown("""
<div class="footer-bar">
    <span>🧬 RSF Survival Predictor</span>
    <span class="footer-dot"></span>
    <span>Built with Streamlit</span>
    <span class="footer-dot"></span>
    <span>Powered by scikit-survival</span>
</div>
""", unsafe_allow_html=True)
