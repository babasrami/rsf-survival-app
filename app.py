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
# Custom CSS for 3D Modern UI
# ---------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ============================================
       ROOT VARIABLES & GLOBAL STYLES
    ============================================ */
    :root {
        --bg-gradient-start: #0f0c29;
        --bg-gradient-mid: #302b63;
        --bg-gradient-end: #24243e;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-shadow: rgba(0, 0, 0, 0.3);
        --accent-cyan: #00d4ff;
        --accent-purple: #7b2cbf;
        --accent-pink: #ff006e;
        --accent-green: #00ff88;
        --accent-gold: #ffb700;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --text-muted: rgba(255, 255, 255, 0.5);
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-mid) 50%, var(--bg-gradient-end) 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated background overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 20% 80%, rgba(123, 44, 191, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 40% 40%, rgba(255, 0, 110, 0.05) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* ============================================
       GLASSMORPHISM CONTAINERS
    ============================================ */
    .stMainBlockContainer, .main .block-container {
        background: transparent !important;
        position: relative;
        z-index: 1;
    }
    
    /* Glass card effect for major sections */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.9) 100%) !important;
        border-right: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-primary) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.15);
        transform: translateY(-2px);
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.2) !important;
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
        border-radius: 16px 16px 0 0;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 8px 16px rgba(0, 212, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* ============================================
       BUTTONS - NEON GLOW EFFECT
    ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        box-shadow: 
            0 4px 15px rgba(0, 212, 255, 0.4),
            0 2px 4px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 8px 30px rgba(0, 212, 255, 0.6),
            0 4px 10px rgba(123, 44, 191, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Primary button with extra glow */
    button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4), 0 2px 4px rgba(0, 0, 0, 0.2); }
        50% { box-shadow: 0 6px 25px rgba(0, 212, 255, 0.6), 0 4px 8px rgba(123, 44, 191, 0.3); }
    }
    
    /* ============================================
       FILE UPLOADERS
    ============================================ */
    div[data-testid="stFileUploader"] {
        background: var(--glass-bg);
        border: 2px dashed var(--glass-border);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--accent-cyan);
        background: rgba(0, 212, 255, 0.05);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
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
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
    }
    
    /* ============================================
       DATA TABLES
    ============================================ */
    .stDataFrame, div[data-testid="stDataFrame"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .stDataFrame table {
        background: transparent !important;
    }
    
    .stDataFrame th {
        background: rgba(0, 212, 255, 0.1) !important;
        color: var(--accent-cyan) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--glass-border) !important;
    }
    
    .stDataFrame tr:hover td {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* ============================================
       HEADERS & TYPOGRAPHY
    ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    h1 {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 50%, var(--accent-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-shadow: none;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        color: var(--text-primary) !important;
        position: relative;
        padding-bottom: 10px;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        border-radius: 2px;
    }
    
    p, li, span, div {
        color: var(--text-secondary);
    }
    
    .stMarkdown a {
        color: var(--accent-cyan) !important;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    
    .stMarkdown a:hover {
        color: var(--accent-pink) !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* ============================================
       SELECTBOX & INPUT STYLING
    ============================================ */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    }
    
    .stSelectbox label,
    .stMultiSelect label {
        color: var(--text-primary) !important;
    }
    
    /* Toggle styling */
    .stToggle label span {
        color: var(--text-primary) !important;
    }
    
    /* ============================================
       MATPLOTLIB CHART CONTAINER
    ============================================ */
    .stPlotlyChart, div[data-testid="stPlotlyChart"],
    div.stPyplot {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stPyplot:hover {
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    /* ============================================
       DIVIDERS
    ============================================ */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--glass-border), var(--accent-cyan), var(--glass-border), transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* ============================================
       SCROLLBAR
    ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-purple), var(--accent-pink));
    }
    
    /* ============================================
       INFO/WARNING/SUCCESS BOXES
    ============================================ */
    .stAlert {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stAlert"] > div {
        color: var(--text-primary) !important;
    }
    
    /* Info alert */
    div[role="alert"]:has(svg[data-testid="stInfoIcon"]) {
        border-left: 4px solid var(--accent-cyan) !important;
    }
    
    /* Warning alert */
    div[role="alert"]:has(svg[data-testid="stWarningIcon"]) {
        border-left: 4px solid var(--accent-gold) !important;
    }
    
    /* Success alert */
    div[role="alert"]:has(svg[data-testid="stSuccessIcon"]) {
        border-left: 4px solid var(--accent-green) !important;
    }
    
    /* ============================================
       SIDEBAR EXTRAS
    ============================================ */
    section[data-testid="stSidebar"] .stDivider {
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        background: none !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }
    
    section[data-testid="stSidebar"] code {
        background: rgba(0, 212, 255, 0.1) !important;
        color: var(--accent-cyan) !important;
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 6px;
    }
    
    /* ============================================
       CAPTION STYLING
    ============================================ */
    .stCaption, small {
        color: var(--text-muted) !important;
        font-size: 0.85rem;
    }
    
    /* ============================================
       TABLE STYLING
    ============================================ */
    .stTable {
        background: var(--glass-bg) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stTable th {
        background: rgba(0, 212, 255, 0.15) !important;
        color: var(--accent-cyan) !important;
    }
    
    .stTable td {
        color: var(--text-primary) !important;
        border-color: var(--glass-border) !important;
    }
    
    /* ============================================
       ANIMATIONS
    ============================================ */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Subtle floating animation for hero */
    h1 {
        animation: float 6s ease-in-out infinite;
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
# Hero header with icon
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <span style="font-size: 3.5rem;">🧬</span>
</div>
""", unsafe_allow_html=True)

st.title("Random Survival Forest — Survival Prediction")

# Subtitle
st.markdown("""
<p style="text-align: center; font-size: 1.1rem; color: rgba(255,255,255,0.6); margin-top: -10px; margin-bottom: 30px;">
    Advanced Machine Learning for Personalized Survival Analysis
</p>
""", unsafe_allow_html=True)

with st.expander("📋 What you need to upload", expanded=True):
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
        - Protein columns ending in `_pTPM`
        """)

st.markdown("<br>", unsafe_allow_html=True)

# File uploaders in styled columns
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

# Sidebar with enhanced styling
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <span style="font-size: 2rem;">⚙️</span>
    <h2 style="margin: 5px 0 0 0; font-size: 1.3rem;">Inference Options</h2>
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
st.sidebar.markdown("""
<div style="text-align: center;">
    <span style="font-size: 1.5rem;">📦</span>
    <h3 style="margin: 5px 0; font-size: 1.1rem;">Bundle Summary</h3>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown(f"**Features expected:** `{len(features)}`")
if features:
    feature_preview = "\n".join(features[:15])
    if len(features) > 15:
        feature_preview += f"\n... (+{len(features) - 15} more)"
    st.sidebar.code(feature_preview, language=None)

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

left, right = st.columns([1, 1])

with left:
    st.markdown("### ✏️ Input Data")
    st.caption("Edit values directly below. Use NA or blank for missing values.")
    df_edit = st.data_editor(df_in, num_rows="dynamic", use_container_width=True)

with right:
    st.markdown("### 🚀 Run Predictions")
    if id_col:
        st.caption(f"Patient identifier: `{id_col}`")
    else:
        st.caption("No patient ID column detected — using row index")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔬 Predict Survival", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

# Build X
try:
    X = preprocess_for_model(df_edit, bundle=bundle, input_is_raw_ptpm=input_is_raw)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# Predict
try:
    surv_funcs = model.predict_survival_function(X.values, return_array=False)
    risk_scores = model.predict(X.values)
except Exception as e:
    st.error(f"Model prediction failed: {e}")
    st.stop()

# Results table
rows = []
for i in range(len(X)):
    pid = df_edit.iloc[i][id_col] if id_col else i
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

# Results section with enhanced styling
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <span style="font-size: 2rem;">📈</span>
    <h2 style="display: inline-block; margin-left: 10px; background: linear-gradient(135deg, #00d4ff, #7b2cbf); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Prediction Results
    </h2>
</div>
""", unsafe_allow_html=True)

st.dataframe(res_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Survival curve section
st.markdown("### 📊 Survival Curve Analysis")

sel_options = list(range(len(X)))
if id_col:
    sel_label = df_edit[id_col].astype(str).tolist()
    sel = st.selectbox("Select patient for detailed analysis:", options=sel_options, format_func=lambda i: f"🧬 {sel_label[i]}")
else:
    sel = st.selectbox("Select patient (row index):", options=sel_options, format_func=lambda i: f"🧬 Patient {i}")

sf = surv_funcs[sel]
# StepFunction has x and y
try:
    xs = sf.x
    ys = sf.y
except Exception:
    # fallback: sample along a grid
    xs = np.linspace(0, np.nanmax(df_edit.get("days", pd.Series([3650]))), 200)
    ys = np.array([sf(t) for t in xs])

# Enhanced matplotlib styling for dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 5))

# Set figure and axes background
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Plot survival curve with gradient-like effect
ax.fill_between(xs, ys, alpha=0.3, color='#00d4ff', step='post')
ax.step(xs, ys, where="post", color='#00d4ff', linewidth=2.5, label='Survival Probability')

# Add glow effect
ax.step(xs, ys, where="post", color='#00d4ff', linewidth=6, alpha=0.2)

# Styling
ax.set_xlabel("Time (days)", fontsize=12, color='white', fontweight='500')
ax.set_ylabel("Survival Probability", fontsize=12, color='white', fontweight='500')
ax.set_ylim(0, 1.05)
ax.set_xlim(0, max(xs) if len(xs) > 0 else 3650)

# Grid styling
ax.grid(True, alpha=0.15, color='white', linestyle='--')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tick styling
ax.tick_params(colors='white', which='both')

# Add timepoint markers if available
for y, t in zip(timepoints_years, timepoints_days):
    if t <= max(xs):
        prob = float(sf(t))
        ax.axvline(x=t, color='#7b2cbf', linestyle=':', alpha=0.5)
        ax.scatter([t], [prob], color='#ff006e', s=80, zorder=5, edgecolors='white', linewidths=1.5)
        ax.annotate(f'{y}y: {prob:.1%}', (t, prob), textcoords="offset points", 
                    xytext=(10, 10), fontsize=9, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#7b2cbf', alpha=0.7))

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# Detail panel with enhanced metrics
st.markdown("### 🎯 Selected Patient Details")

pid = df_edit.iloc[sel][id_col] if id_col else sel
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
    st.markdown("#### 📅 Survival Probabilities at Key Timepoints")
    probs = survival_prob_at_times(sf, timepoints_days)
    
    # Display as styled columns
    prob_cols = st.columns(len(timepoints_years))
    for i, (y, p) in enumerate(zip(timepoints_years, probs)):
        with prob_cols[i]:
            # Color based on probability
            if p >= 0.7:
                emoji = "🟢"
            elif p >= 0.4:
                emoji = "🟡"
            else:
                emoji = "🔴"
            st.metric(f"{emoji} Year {y}", f"{p:.1%}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.4); font-size: 0.85rem;">
    🧬 RSF Survival Predictor • Built with Streamlit • Powered by scikit-survival
</div>
""", unsafe_allow_html=True)
