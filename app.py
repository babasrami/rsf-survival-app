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
# Custom CSS — Production-Grade Dark Theme
# ---------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* =============================================
       CSS VARIABLES
    ============================================= */
    :root {
        --bg-primary: #0f1117;
        --bg-secondary: #161b22;
        --bg-card: #1c2333;
        --bg-card-hover: #222b3d;
        --border-subtle: rgba(124, 58, 237, 0.12);
        --border-accent: rgba(124, 58, 237, 0.35);
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
        --cyan-400: #22d3ee;
        --cyan-500: #06b6d4;
        --text-primary: #f0f2f6;
        --text-secondary: #c0c6d4;
        --text-muted: #8b92a5;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 14px rgba(0,0,0,0.35);
        --shadow-lg: 0 10px 30px rgba(0,0,0,0.45);
        --shadow-glow: 0 0 25px rgba(124,58,237,0.12);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    /* =============================================
       GLOBAL — APP BACKGROUND & FONT
    ============================================= */
    .stApp,
    .stApp > header,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(170deg, var(--bg-primary) 0%, #12141f 50%, var(--bg-primary) 100%) !important;
        background-attachment: fixed !important;
    }

    /* =============================================
       MAIN CONTENT AREA
    ============================================= */
    .stMainBlockContainer, .main .block-container,
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        max-width: 1100px;
        padding-top: 2rem !important;
    }

    /* =============================================
       SIDEBAR
    ============================================= */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #13162b 0%, #0d0f1e 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        background: none !important;
    }
    section[data-testid="stSidebar"] code {
        background: rgba(124, 58, 237, 0.12) !important;
        color: var(--purple-300) !important;
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 6px;
        padding: 2px 6px;
        font-size: 0.82rem;
    }
    section[data-testid="stSidebar"] .stDivider,
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.25), transparent) !important;
    }

    /* =============================================
       TYPOGRAPHY
    ============================================= */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    h1 {
        font-size: 2.4rem !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(135deg, #ffffff 0%, var(--purple-400) 55%, var(--cyan-400) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    h2 {
        font-size: 1.45rem !important;
        letter-spacing: -0.3px !important;
    }
    h3 {
        font-size: 1.15rem !important;
        color: var(--text-secondary) !important;
    }
    p, li, span, div, label {
        color: var(--text-secondary) !important;
    }
    a {
        color: var(--purple-400) !important;
        text-decoration: none !important;
    }
    a:hover {
        color: var(--purple-300) !important;
    }
    small, .stCaption, caption {
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
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
       BUTTONS
    ============================================= */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--purple-600) 0%, var(--purple-800) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(167,139,250,0.25) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.65rem 1.6rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.2px;
        box-shadow: var(--shadow-md), 0 0 18px rgba(124,58,237,0.15);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg), 0 0 30px rgba(124,58,237,0.25);
        border-color: rgba(167,139,250,0.45) !important;
        background: linear-gradient(135deg, var(--purple-500) 0%, var(--purple-700) 100%) !important;
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: var(--shadow-sm);
    }

    /* =============================================
       FILE UPLOADERS
    ============================================= */
    div[data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed rgba(124, 58, 237, 0.22) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.2rem !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--border-accent) !important;
        background: var(--bg-card-hover) !important;
        box-shadow: var(--shadow-glow);
    }
    div[data-testid="stFileUploader"] label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button {
        background: rgba(124, 58, 237, 0.15) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* =============================================
       SELECTBOX, MULTISELECT, INPUTS
    ============================================= */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: var(--border-accent) !important;
        box-shadow: var(--shadow-glow);
    }
    .stSelectbox label,
    .stMultiSelect label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* Radio buttons */
    .stRadio > div {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md);
        padding: 0.5rem 0.75rem;
        border: 1px solid var(--border-subtle);
    }
    .stRadio label span {
        color: var(--text-secondary) !important;
    }

    /* Toggle */
    .stToggle label span {
        color: var(--text-primary) !important;
    }

    /* =============================================
       DATA TABLE / DATAFRAME
    ============================================= */
    .stDataFrame, div[data-testid="stDataFrame"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-md);
    }
    .stDataFrame th {
        background: rgba(124, 58, 237, 0.12) !important;
        color: var(--purple-300) !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(124,58,237,0.2) !important;
    }
    .stDataFrame td {
        background: transparent !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    }
    .stDataFrame tr:hover td {
        background: rgba(124,58,237,0.06) !important;
    }

    /* =============================================
       METRIC CARDS
    ============================================= */
    div[data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.15rem 1.3rem !important;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        border-color: var(--border-accent) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.7px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* =============================================
       EXPANDER
    ============================================= */
    .streamlit-expanderHeader,
    details > summary {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: all 0.25s ease;
        box-shadow: var(--shadow-sm);
    }
    .streamlit-expanderHeader:hover,
    details > summary:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--border-accent) !important;
        box-shadow: var(--shadow-md), var(--shadow-glow);
    }
    .streamlit-expanderContent,
    details > div {
        background: rgba(28, 35, 51, 0.8) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }

    /* =============================================
       ALERTS (info / warning / success / error)
    ============================================= */
    .stAlert, div[data-testid="stAlert"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stAlert"] > div,
    div[role="alert"] {
        color: var(--text-primary) !important;
    }

    /* =============================================
       PLOT / CHART CONTAINERS
    ============================================= */
    div.stPyplot,
    div[data-testid="stPlotlyChart"],
    .stPlotlyChart {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-md);
        transition: border-color 0.3s;
    }
    div.stPyplot:hover {
        border-color: var(--border-accent) !important;
    }

    /* =============================================
       TABLE (plain st.table)
    ============================================= */
    .stTable {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    .stTable th {
        background: rgba(124,58,237,0.12) !important;
        color: var(--purple-300) !important;
    }
    .stTable td {
        color: var(--text-primary) !important;
        border-color: var(--border-subtle) !important;
    }

    /* =============================================
       SCROLLBAR
    ============================================= */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--purple-600), var(--purple-800));
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--purple-400), var(--purple-600));
    }

    /* =============================================
       CUSTOM UTILITY CLASSES
    ============================================= */
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(34,211,238,0.10));
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 50px;
        padding: 6px 18px;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--purple-300) !important;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    .hero-subtitle {
        color: var(--text-muted) !important;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.75rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.25rem;
    }
    .section-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.6rem;
    }
    .section-icon {
        font-size: 1.4rem;
        line-height: 1;
    }
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin: 0;
    }
    .section-desc {
        font-size: 0.85rem;
        color: var(--text-muted) !important;
        margin: 0;
    }
    .results-header {
        text-align: center;
        margin-bottom: 1rem;
    }
    .results-header h2 {
        display: inline-block;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.5rem !important;
        background: linear-gradient(135deg, #ffffff 0%, var(--purple-400) 60%, var(--cyan-400) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 0;
    }
    .feature-chip {
        display: inline-block;
        background: rgba(124,58,237,0.10);
        border: 1px solid rgba(124,58,237,0.18);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: var(--purple-300) !important;
        margin: 2px 3px;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .footer-bar {
        text-align: center;
        padding: 2rem 0 1rem;
        font-size: 0.8rem;
        color: var(--text-muted) !important;
        border-top: 1px solid var(--border-subtle);
        margin-top: 2rem;
    }
    .footer-bar span {
        color: var(--text-muted) !important;
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
    # Common training file uses: status (event indicator), days (follow-up time)
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

    # Apply protein preprocessing if user inputs raw pTPM
    # We treat columns ending with '_pTPM' as protein columns.
    protein_cols = [c for c in df.columns if str(c).endswith("_pTPM")]
    scaler = extract_scaler(bundle)
    scaler_cols = extract_scaler_cols(bundle, protein_cols)

    if input_is_raw_ptpm and protein_cols:
        # Log1p on proteins (fill missing with 0 as in training notebook)
        for col in protein_cols:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0.0))

        # Apply z-score scaler if available (recommended)
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

    # Build X in correct order
    # If the uploaded patient file is missing some expected columns, add them as NA.
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            df[c] = np.nan
        st.warning(
            "Input file is missing some model features. They will be treated as NA and imputed where possible. "
            f"Missing (first 10): {missing_cols[:10]}"
        )

    X = df.reindex(columns=features).copy()

    # Coerce numeric where possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Median imputation from training (if provided)
    med = bundle.get("feature_medians", {}) or {}
    if med:
        for col in X.columns:
            if col in med:
                X[col] = X[col].fillna(med[col])

    # Fallback filling if medians are not provided (keeps inference robust)
    # - proteins: fill NA with 0 (same as training pre-log1p)
    # - other numeric features: fill NA with 0
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
        "Low": "#00ff88",
        "Intermediate": "#ffb700",
        "High": "#ff006e"
    }
    return colors.get(risk_group, "#00d4ff")

# ---------------------------
# UI
# ---------------------------

# ═══════════════ HERO SECTION ═══════════════
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.75rem;">
    <span class="hero-badge">🧬 &nbsp;MACHINE LEARNING &nbsp;•&nbsp; SURVIVAL ANALYSIS</span>
</div>
""", unsafe_allow_html=True)

st.title("Random Survival Forest — Survival Prediction")

st.markdown("""
<p class="hero-subtitle" style="text-align:center; margin-top:-8px; margin-bottom:1.75rem;">
    Upload your trained model bundle and patient data to generate individualized
    survival probability curves and risk stratification.
</p>
""", unsafe_allow_html=True)

# ═══════════════ UPLOAD SECTION (card) ═══════════════
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <span class="section-icon">📋</span>
        <p class="section-title">What you need to upload</p>
    </div>
    <p class="section-desc">Provide a trained model bundle and patient data file to get started.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("View upload requirements", expanded=False):
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
<div style="text-align:center; padding: 8px 0 16px;">
    <span style="font-size:1.8rem;">⚙️</span>
    <h2 style="margin:4px 0 0; font-size:1.2rem; font-weight:700;">Inference Options</h2>
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
        <span class="section-icon">✏️</span>
        <p class="section-title">Input Data</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Edit values directly below. Use NA or blank for missing values.")
    df_edit = st.data_editor(df_in, num_rows="dynamic", use_container_width=True, key="data_editor")

with right:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🚀</span>
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
            # Store in session state
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
    <span class="section-icon">📊</span>
    <p class="section-title">Survival Curve Analysis</p>
</div>
""", unsafe_allow_html=True)

# Plot mode selection - Single patient vs All patients
plot_mode = st.radio(
    "Plot mode:",
    options=["Single Patient", "All Patients"],
    horizontal=True,
    key="plot_mode_selector"
)

# Color palette for multi-patient plot - Purple/Grey theme
PATIENT_COLORS = [
    '#a78bfa', '#7c3aed', '#c4b5fd', '#6d28d9', '#ddd6fe',
    '#b0b0b0', '#808080', '#e0e0e0', '#5b21b6', '#8b5cf6',
    '#f8f8f8', '#4c1d95', '#ede9fe', '#606060', '#9333ea',
    '#d8d8d8', '#7e22ce', '#a3a3a3', '#c084fc', '#404040'
]

if plot_mode == "Single Patient":
    # Single patient selector
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
    # StepFunction has x and y
    try:
        xs = sf.x
        ys = sf.y
    except Exception:
        xs = np.linspace(0, np.nanmax(df_used.get("days", pd.Series([3650]))), 200)
        ys = np.array([sf(t) for t in xs])

    # Enhanced matplotlib styling for dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set figure and axes background - dark theme
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    # Plot survival curve with purple gradient effect
    ax.fill_between(xs, ys, alpha=0.25, color='#7c3aed', step='post')
    ax.step(xs, ys, where="post", color='#a78bfa', linewidth=2.5, label='Survival Probability')

    # Add glow effect
    ax.step(xs, ys, where="post", color='#7c3aed', linewidth=6, alpha=0.15)

    # Styling
    ax.set_xlabel("Time (days)", fontsize=12, color='white', fontweight='500')
    ax.set_ylabel("Survival Probability", fontsize=12, color='white', fontweight='500')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(xs) if len(xs) > 0 else 3650)

    # Grid styling - subtle grey
    ax.grid(True, alpha=0.1, color='#606060', linestyle='--')
    ax.spines['bottom'].set_color('#404040')
    ax.spines['left'].set_color('#404040')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick styling - light grey
    ax.tick_params(colors='#b0b0b0', which='both')

    # Add timepoint markers if available
    for y, t in zip(timepoints_years, timepoints_days):
        if t <= max(xs):
            prob = float(sf(t))
            ax.axvline(x=t, color='#4a1d6a', linestyle=':', alpha=0.6)
            ax.scatter([t], [prob], color='#e0e0e0', s=80, zorder=5, edgecolors='#7c3aed', linewidths=2)
            ax.annotate(f'{y}y: {prob:.1%}', (t, prob), textcoords="offset points",
                        xytext=(10, 10), fontsize=9, color='#f8f8f8',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#4a1d6a', edgecolor='#7c3aed', alpha=0.85))

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Detail panel with enhanced metrics
    st.markdown("""
    <div class="section-header" style="margin-top:1.25rem;">
        <span class="section-icon">🎯</span>
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
            <span class="section-icon">📅</span>
            <p class="section-title">Survival Probabilities at Key Timepoints</p>
        </div>
        """, unsafe_allow_html=True)
        probs = survival_prob_at_times(sf, timepoints_days)

        # Display as styled columns
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

    # Use default style for light background
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set figure and axes background - light theme
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    max_time = 3650  # Default max time

    # Generate unique colors for all patients using colormap (no limit)
    import matplotlib.cm as cm
    num_patients = len(X)
    colormap = cm.get_cmap('gist_rainbow', num_patients)  # Use rainbow for max variety

    # Plot each patient with a unique color
    for i in range(num_patients):
        sf = surv_funcs[i]
        color = colormap(i / max(num_patients - 1, 1))  # Get unique color from colormap

        # Get patient label
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

        # Plot with slight glow effect
        ax.step(xs, ys, where="post", color=color, linewidth=4, alpha=0.15)  # Glow
        ax.step(xs, ys, where="post", color=color, linewidth=2, label=label, alpha=0.9)

    # Styling - black text for light background
    ax.set_xlabel("Time (days)", fontsize=12, color='black', fontweight='500')
    ax.set_ylabel("Survival Probability", fontsize=12, color='black', fontweight='500')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_time)

    # Grid styling - subtle grey on light background
    ax.grid(True, alpha=0.3, color='#cccccc', linestyle='--')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick styling - black
    ax.tick_params(colors='black', which='both')

    # Legend on top right with light style
    legend = ax.legend(
        loc='upper right',
        fontsize=9,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#333333',
        labelcolor='black',
        title='Patients',
        title_fontsize=10,
        ncol=min(3, (len(X) + 9) // 10)  # Adaptive columns
    )
    legend.get_title().set_color('black')

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Summary statistics for all patients
    st.markdown("""
    <div class="section-header" style="margin-top:1.25rem;">
        <span class="section-icon">📊</span>
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
    # Save current plot to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='#0f1117', bbox_inches='tight')
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
    <span>🧬 RSF Survival Predictor</span> &nbsp;•&nbsp;
    <span>Built with Streamlit</span> &nbsp;•&nbsp;
    <span>Powered by scikit-survival</span>
</div>
""", unsafe_allow_html=True)
