
import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go

st.set_page_config(page_title="RSF Survival Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Custom CSS for 3D Design
# ---------------------------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container with gradient background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* 3D Card effect for containers */
    .stApp > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.3),
            0 0 40px rgba(102, 126, 234, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
    }
    
    /* Title styling with 3D text effect */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem 0;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #667eea;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Button 3D effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 
            0 8px 16px rgba(102, 126, 234, 0.4),
            0 4px 8px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        transform: translateY(0);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 24px rgba(102, 126, 234, 0.5),
            0 6px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 
            0 4px 8px rgba(102, 126, 234, 0.3),
            0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(102, 126, 234, 0.05);
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Data editor styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.15),
            0 0 20px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric cards with 3D effect */
    .stMetric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
    }
    
    .css-1d391kg *, [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Number input and slider styling */
    .stNumberInput > div > div > input,
    .stSlider > div > div > div > div {
        border-radius: 8px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Divider with gradient */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Download button styling */
    .download-btn {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

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

# ---------------------------
# UI
# ---------------------------
st.title("🔬 Random Survival Forest — Advanced Survival Prediction")

with st.expander("📋 What you upload", expanded=True):
    st.markdown(
        """
- **Model bundle (.joblib)**: a `joblib.dump()` dictionary containing at least:
  - `model` (a fitted `sksurv.ensemble.RandomSurvivalForest`)
  - `features` (ordered list of feature names used in training)
  - `feature_medians` (dict of medians for imputation)
  - `risk_ref` (optional; contains risk-score quantiles for risk-group labeling)
  - `scaler` and `scaler_cols` (optional but strongly recommended if you trained with z-scored proteins)
- **Patient input (.xlsx or .csv)**: one or more rows matching your training schema (you may place missing values as `NA`/blank).
        """
    )

colA, colB = st.columns(2)
with colA:
    model_file = st.file_uploader("📦 Upload model bundle (.joblib)", type=["joblib"], accept_multiple_files=False)
with colB:
    data_file = st.file_uploader("📊 Upload patient data (.xlsx or .csv)", type=["xlsx", "xls", "csv"], accept_multiple_files=False)

if not model_file:
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

st.sidebar.header("⚙️ Inference Options")
input_is_raw = st.sidebar.toggle("Inputs are raw pTPM (apply log1p + scaling if available)", value=True)
timepoints_years = st.sidebar.multiselect("Report survival probability at years:",
                                         options=[1,2,3,5,10],
                                         default=[1,2,3,5])
timepoints_days = [y * 365.25 for y in timepoints_years]

st.sidebar.divider()
st.sidebar.subheader("📦 Bundle Summary")
st.sidebar.write(f"Features expected: {len(features)}")
if features:
    st.sidebar.code("\n".join(features[:20]) + ("\n..." if len(features) > 20 else ""))

if not data_file:
    st.info("📤 Upload a patient data file to proceed.")
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

st.subheader("📝 Input Data Editor")
st.caption("Edit values directly below. Use interactive controls for numeric columns or edit cells manually.")

# Create tabs for different editing modes
edit_tab1, edit_tab2 = st.tabs(["📊 Table View", "🎛️ Interactive Controls"])

with edit_tab1:
    df_edit = st.data_editor(
        df_in, 
        num_rows="dynamic", 
        use_container_width=True,
        height=400
    )

with edit_tab2:
    st.caption("Adjust numeric values using sliders and number inputs")
    
    # Store edited dataframe
    if 'df_interactive' not in st.session_state:
        st.session_state.df_interactive = df_in.copy()
    
    # Select row to edit
    row_options = list(range(len(st.session_state.df_interactive)))
    if id_col:
        row_labels = st.session_state.df_interactive[id_col].astype(str).tolist()
        selected_row = st.selectbox("Select patient/row to edit:", options=row_options, format_func=lambda i: f"Row {i}: {row_labels[i]}")
    else:
        selected_row = st.selectbox("Select row to edit:", options=row_options)
    
    # Create interactive controls for numeric columns
    numeric_cols = st.session_state.df_interactive.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        cols_per_row = 2
        for i in range(0, len(numeric_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                with cols[j]:
                    current_val = st.session_state.df_interactive.loc[selected_row, col_name]
                    if pd.isna(current_val):
                        current_val = 0.0
                    
                    # Determine reasonable min/max for slider
                    col_min = float(st.session_state.df_interactive[col_name].min()) if not st.session_state.df_interactive[col_name].isna().all() else 0.0
                    col_max = float(st.session_state.df_interactive[col_name].max()) if not st.session_state.df_interactive[col_name].isna().all() else 100.0
                    
                    # Expand range slightly
                    col_range = col_max - col_min
                    col_min = col_min - col_range * 0.1
                    col_max = col_max + col_range * 0.1
                    
                    if col_min == col_max:
                        col_min = 0.0
                        col_max = 100.0
                    
                    new_val = st.slider(
                        f"{col_name}",
                        min_value=float(col_min),
                        max_value=float(col_max),
                        value=float(current_val),
                        key=f"slider_{selected_row}_{col_name}"
                    )
                    st.session_state.df_interactive.loc[selected_row, col_name] = new_val
    
    # Use the interactively edited dataframe
    df_edit = st.session_state.df_interactive.copy()
    
    st.dataframe(df_edit, use_container_width=True, height=200)

st.divider()

col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("🚀 Run Predictions")
    if id_col:
        st.caption(f"Patient identifier column detected: `{id_col}`")
    else:
        st.caption("No patient identifier column detected. Predictions will be displayed by row index.")

    run_btn = st.button("🔮 Predict Survival", type="primary", use_container_width=True)

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
    row = {"Patient": pid, "Risk_Score": risk, "Risk_Group": risk_group}
    for y, p in zip(timepoints_years, probs):
        row[f"S(t={y}y)"] = p
    rows.append(row)

res_df = pd.DataFrame(rows)

st.divider()
st.subheader("📊 Prediction Results")

# Add download button for CSV
csv_buffer = io.StringIO()
res_df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

col1, col2 = st.columns([3, 1])
with col1:
    st.dataframe(res_df, use_container_width=True)
with col2:
    st.download_button(
        label="💾 Download as CSV",
        data=csv_data,
        file_name="survival_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

# Plot per-patient survival curve with Plotly for zoom functionality
st.subheader("📈 Interactive Survival Curve (with Zoom)")
sel_options = list(range(len(X)))
sel_label = None
if id_col:
    sel_label = df_edit[id_col].astype(str).tolist()
    sel = st.selectbox("Select patient", options=sel_options, format_func=lambda i: sel_label[i])
else:
    sel = st.selectbox("Select patient (row index)", options=sel_options)

sf = surv_funcs[sel]
# StepFunction has x and y
try:
    xs = sf.x
    ys = sf.y
except Exception:
    # fallback: sample along a grid
    xs = np.linspace(0, np.nanmax(df_edit.get("days", pd.Series([3650]))), 200)
    ys = np.array([sf(t) for t in xs])

# Create interactive Plotly chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=xs,
    y=ys,
    mode='lines',
    name='Survival Probability',
    line=dict(color='#667eea', width=3, shape='hv'),
    fill='tozeroy',
    fillcolor='rgba(102, 126, 234, 0.2)'
))

fig.update_layout(
    title=dict(
        text=f"Survival Curve - Patient: {df_edit.iloc[sel][id_col] if id_col else sel}",
        font=dict(size=20, color='#667eea', family='Inter')
    ),
    xaxis_title="Time (days)",
    yaxis_title="Survival Probability",
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', size=12),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.1)',
        zeroline=False
    ),
    yaxis=dict(
        range=[0, 1.05],
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.1)',
        zeroline=False
    ),
    height=500
)

# Add zoom and pan tools
fig.update_xaxes(fixedrange=False)
fig.update_yaxes(fixedrange=False)

st.plotly_chart(fig, use_container_width=True, config={
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
})

# Detail panel
st.subheader("🔍 Selected Patient Details")
pid = df_edit.iloc[sel][id_col] if id_col else sel
risk = float(risk_scores[sel])
risk_group, _ = classify_risk(risk, risk_ref)

detail_cols = st.columns(3)
detail_cols[0].metric("👤 Patient", str(pid))
detail_cols[1].metric("⚠️ Risk Score", f"{risk:.4f}")
detail_cols[2].metric("📊 Risk Group", risk_group if risk_group else "—")

if timepoints_years:
    probs = survival_prob_at_times(sf, timepoints_days)
    prob_df = pd.DataFrame({"Year": timepoints_years, "Survival Probability": probs})
    st.table(prob_df)

st.divider()
st.caption("✨ Powered by Random Survival Forest | Enhanced 3D UI Design")
