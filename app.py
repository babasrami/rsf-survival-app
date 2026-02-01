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
    page_icon="",
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
       ROOT VARIABLES - BLACK, WHITE, GREY, PURPLE
    ============================================ */
    :root {
        --bg-dark: #0a0a0f;
        --bg-mid: #121218;
        --bg-light: #1a1a24;
        --glass-bg: rgba(255, 255, 255, 0.02);
        --glass-border: rgba(255, 255, 255, 0.06);
        --glass-shadow: rgba(0, 0, 0, 0.5);
        --purple-dark: #4a1d6a;
        --purple-main: #7c3aed;
        --purple-light: #a78bfa;
        --purple-glow: rgba(124, 58, 237, 0.3);
        --grey-100: #f8f8f8;
        --grey-200: #e0e0e0;
        --grey-300: #b0b0b0;
        --grey-400: #808080;
        --grey-500: #606060;
        --grey-600: #404040;
        --grey-700: #2a2a2a;
        --grey-800: #1a1a1a;
        --text-primary: rgba(255, 255, 255, 0.95);
        --text-secondary: rgba(255, 255, 255, 0.65);
        --text-muted: rgba(255, 255, 255, 0.40);
    }
    
    /* ============================================
       MAIN APP BACKGROUND - FADED DARK
    ============================================ */
    .stApp {
        background: linear-gradient(160deg, 
            var(--bg-dark) 0%, 
            var(--bg-mid) 40%, 
            #0d0d14 70%,
            var(--bg-dark) 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }
    
    /* Subtle purple ambient glow overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 10% 90%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 10%, rgba(124, 58, 237, 0.05) 0%, transparent 45%),
            radial-gradient(ellipse at 50% 50%, rgba(60, 60, 80, 0.03) 0%, transparent 60%);
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
    
    /* ============================================
       SIDEBAR - DARK GLASS
    ============================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(18, 18, 26, 0.98) 0%, 
            rgba(12, 12, 18, 0.99) 100%) !important;
        border-right: 1px solid rgba(124, 58, 237, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.5);
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-primary) !important;
    }
    
    section[data-testid="stSidebar"] .stDivider {
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.4), transparent) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        background: none !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }
    
    section[data-testid="stSidebar"] code {
        background: rgba(124, 58, 237, 0.15) !important;
        color: var(--purple-light) !important;
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 6px;
    }
    
    /* ============================================
       EXPANDER STYLING
    ============================================ */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(124, 58, 237, 0.3) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px var(--purple-glow);
        transform: translateY(-2px);
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.25) !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ============================================
       METRIC CARDS WITH SHADOWS
    ============================================ */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.4),
            0 4px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.5),
            0 10px 25px rgba(124, 58, 237, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(124, 58, 237, 0.25);
    }
    
    div[data-testid="stMetric"] label {
        color: var(--grey-400) !important;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.75rem !important;
        font-weight: 700;
    }
    
    /* ============================================
       3D BUTTONS WITH SHADOWS
    ============================================ */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(145deg, 
            var(--purple-main) 0%, 
            var(--purple-dark) 100%) !important;
        color: white !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 14px !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
        box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.4),
            0 4px 10px rgba(124, 58, 237, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            inset 0 -2px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before,
    .stDownloadButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.08), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
    }
    
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 15px 40px rgba(0, 0, 0, 0.5),
            0 8px 20px rgba(124, 58, 237, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            inset 0 -2px 0 rgba(0, 0, 0, 0.2);
        border-color: rgba(167, 139, 250, 0.5) !important;
    }
    
    .stButton > button:hover::before,
    .stDownloadButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: 
            0 4px 15px rgba(0, 0, 0, 0.4),
            0 2px 8px rgba(124, 58, 237, 0.2),
            inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* ============================================
       FILE UPLOADERS
    ============================================ */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.015);
        border: 2px dashed rgba(124, 58, 237, 0.25);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.25);
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(124, 58, 237, 0.5);
        background: rgba(124, 58, 237, 0.03);
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35), 0 0 25px var(--purple-glow);
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
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
    }
    
    /* ============================================
       DATA TABLES
    ============================================ */
    .stDataFrame, div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.015) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
    }
    
    .stDataFrame table {
        background: transparent !important;
    }
    
    .stDataFrame th {
        background: rgba(124, 58, 237, 0.12) !important;
        color: var(--purple-light) !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(124, 58, 237, 0.25) !important;
    }
    
    .stDataFrame td {
        background: transparent !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    .stDataFrame tr:hover td {
        background: rgba(124, 58, 237, 0.05) !important;
    }
    
    /* ============================================
       HEADERS & TYPOGRAPHY
    ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    h1 {
        background: linear-gradient(135deg, 
            var(--grey-100) 0%, 
            var(--purple-light) 50%, 
            var(--grey-200) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-shadow: 0 4px 30px rgba(124, 58, 237, 0.3);
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        color: var(--text-primary) !important;
        position: relative;
        padding-bottom: 12px;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 50px;
        height: 2px;
        background: linear-gradient(90deg, var(--purple-main), transparent);
        border-radius: 2px;
    }
    
    h3 {
        color: var(--grey-200) !important;
    }
    
    p, li, span, div {
        color: var(--text-secondary);
    }
    
    .stMarkdown a {
        color: var(--purple-light) !important;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    
    .stMarkdown a:hover {
        color: var(--grey-100) !important;
        text-shadow: 0 0 10px var(--purple-glow);
    }
    
    /* ============================================
       SELECTBOX & INPUT STYLING
    ============================================ */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: rgba(124, 58, 237, 0.3) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), 0 0 15px var(--purple-glow);
    }
    
    .stSelectbox label,
    .stMultiSelect label {
        color: var(--text-primary) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.015);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
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
        background: rgba(255, 255, 255, 0.015) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 12px 45px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    div.stPyplot:hover {
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.5), 0 0 30px var(--purple-glow);
        border-color: rgba(124, 58, 237, 0.2);
    }
    
    /* ============================================
       DIVIDERS
    ============================================ */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(124, 58, 237, 0.3), 
            rgba(255, 255, 255, 0.1), 
            rgba(124, 58, 237, 0.3), 
            transparent) !important;
        margin: 2.5rem 0 !important;
    }
    
    /* ============================================
       SCROLLBAR
    ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--purple-main), var(--purple-dark));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--purple-light), var(--purple-main));
    }
    
    /* ============================================
       INFO/WARNING/SUCCESS BOXES
    ============================================ */
    .stAlert {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.25);
    }
    
    div[data-testid="stAlert"] > div {
        color: var(--text-primary) !important;
    }
    
    /* Info alert */
    div[role="alert"]:has(svg[data-testid="stInfoIcon"]) {
        border-left: 4px solid var(--purple-main) !important;
    }
    
    /* Warning alert */
    div[role="alert"]:has(svg[data-testid="stWarningIcon"]) {
        border-left: 4px solid var(--grey-400) !important;
    }
    
    /* Success alert */
    div[role="alert"]:has(svg[data-testid="stSuccessIcon"]) {
        border-left: 4px solid var(--purple-light) !important;
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
        background: rgba(255, 255, 255, 0.015) !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    .stTable th {
        background: rgba(124, 58, 237, 0.12) !important;
        color: var(--purple-light) !important;
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
        50% { transform: translateY(-8px); }
    }
    
    @keyframes subtle-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
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
    <span style="font-size: 3.5rem;"></span>
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
        - Protein expression columns (_pTPM)
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

# Initialize session state for predictions
if "_predictions" not in st.session_state:
    st.session_state["_predictions"] = None

left, right = st.columns([1, 1])

with left:
    st.markdown("### ✏️ Input Data")
    st.caption("Edit values directly below. Use NA or blank for missing values.")
    df_edit = st.data_editor(df_in, num_rows="dynamic", use_container_width=True, key="data_editor")

with right:
    st.markdown("### 🚀 Run Predictions")
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

# Results section with enhanced styling
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <span style="font-size: 2rem;">📈</span>
    <h2 style="display: inline-block; margin-left: 10px; background: linear-gradient(135deg, #f8f8f8, #a78bfa, #e0e0e0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Prediction Results
    </h2>
</div>
""", unsafe_allow_html=True)

st.dataframe(res_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Survival curve section
st.markdown("### 📊 Survival Curve Analysis")

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
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')

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
    st.markdown("### 🎯 Selected Patient Details")

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
        st.markdown("#### 📅 Survival Probabilities at Key Timepoints")
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
    st.markdown("### 📊 Summary Statistics")
    
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

# Download buttons
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
    fig.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        "📥 Download Plot (PNG)",
        data=buf,
        file_name="survival_curves.png",
        mime="image/png",
        use_container_width=True
    )

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.4); font-size: 0.85rem;">
    🧬 RSF Survival Predictor • Built with Streamlit • Powered by scikit-survival
</div>
""", unsafe_allow_html=True)
