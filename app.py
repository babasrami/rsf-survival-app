import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config with custom theme
st.set_page_config(
    page_title="RSF Survival Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card-like containers */
    .main .block-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Headers with gradient text */
    h1 {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #667eea;
        font-weight: 700;
        margin-top: 2rem;
    }
    
    h3 {
        color: #764ba2;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Data editor */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
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
    for k in ["features", "feature_cols", "feature_columns", "X_columns", "columns", "ptpm_cols"]:
        v = bundle.get(k, None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return list(v)
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
    """
    df = ensure_columns(df)
    df = compute_time_event(df)

    if "stage_ordinal" not in df.columns:
        if "stage" in df.columns:
            df["stage_ordinal"] = df["stage"].apply(parse_stage_ordinal)
        else:
            df["stage_ordinal"] = 0.0

    features = extract_features(bundle)
    if not features:
        raise ValueError("Model bundle is missing the feature list.")

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
                st.warning(f"Scaler transform failed: {e}")
        else:
            st.warning("Model bundle does not include a saved scaler.")

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        for c in missing_cols:
            df[c] = np.nan
        st.warning(f"Missing features (first 10): {missing_cols[:10]}")

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

def create_3d_surface_plot(surv_funcs, timepoints_days, patient_labels):
    """Create a 3D surface plot of survival probabilities across patients and time"""
    n_patients = len(surv_funcs)
    time_grid = np.linspace(0, max(timepoints_days) if timepoints_days else 3650, 50)
    
    Z = []
    for sf in surv_funcs:
        probs = [float(sf(t)) for t in time_grid]
        Z.append(probs)
    
    Z = np.array(Z)
    
    fig = go.Figure(data=[go.Surface(
        x=time_grid / 365.25,  # Convert to years
        y=list(range(n_patients)),
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title="Survival Probability"),
    )])
    
    fig.update_layout(
        title="3D Survival Probability Landscape",
        scene=dict(
            xaxis_title="Time (years)",
            yaxis_title="Patient Index",
            zaxis_title="Survival Probability",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_interactive_survival_curve(sf, patient_id, risk_score, risk_group, timepoints_years, timepoints_days):
    """Create an interactive survival curve with Plotly"""
    try:
        xs = sf.x
        ys = sf.y
    except Exception:
        xs = np.linspace(0, 3650, 200)
        ys = np.array([sf(t) for t in xs])
    
    # Convert days to years
    xs_years = xs / 365.25
    
    fig = go.Figure()
    
    # Main survival curve
    fig.add_trace(go.Scatter(
        x=xs_years,
        y=ys,
        mode='lines',
        name='Survival Probability',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    # Add markers for specific timepoints
    if timepoints_years and timepoints_days:
        probs = survival_prob_at_times(sf, timepoints_days)
        fig.add_trace(go.Scatter(
            x=timepoints_years,
            y=probs,
            mode='markers',
            name='Key Timepoints',
            marker=dict(size=12, color='#764ba2', symbol='diamond'),
            text=[f'{y}y: {p:.2%}' for y, p in zip(timepoints_years, probs)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    # Risk group color
    risk_colors = {'Low': '#4ade80', 'Intermediate': '#fbbf24', 'High': '#f87171'}
    risk_color = risk_colors.get(risk_group, '#667eea')
    
    fig.update_layout(
        title=f"Survival Curve - Patient {patient_id}<br><sub>Risk Score: {risk_score:.4f} | Risk Group: {risk_group or 'N/A'}</sub>",
        xaxis_title="Time (years)",
        yaxis_title="Survival Probability",
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', range=[0, 1.05])
    
    return fig

# ---------------------------
# UI
# ---------------------------
st.title("🧬 Random Survival Forest Predictor")
st.markdown("### Advanced Survival Analysis with 3D Visualization")

with st.expander("📚 Documentation", expanded=False):
    st.markdown("""
    **Model Bundle Requirements** (.joblib):
    - `model`: Fitted RandomSurvivalForest
    - `features`: Ordered feature names
    - `feature_medians`: Median values for imputation
    - `risk_ref`: Risk quantiles (optional)
    - `scaler` & `scaler_cols`: Z-score normalization (recommended)
    
    **Patient Data Format** (.xlsx or .csv):
    - Columns matching your training schema
    - Missing values as NA/blank are supported
    """)

# File uploads
st.markdown("### 📁 Upload Files")
colA, colB = st.columns(2)
with colA:
    model_file = st.file_uploader("🤖 Model Bundle (.joblib)", type=["joblib"])
with colB:
    data_file = st.file_uploader("👥 Patient Data (.xlsx or .csv)", type=["xlsx", "xls", "csv"])

if not model_file:
    st.info("👆 Please upload a model bundle to begin")
    st.stop()

# Load model bundle
try:
    bundle = joblib.load(model_file)
except Exception as e:
    st.error(f"❌ Could not load model bundle: {e}")
    st.stop()

if not isinstance(bundle, dict) or "model" not in bundle:
    st.error("❌ Invalid model bundle (missing 'model' key)")
    st.stop()

model = bundle["model"]
features = extract_features(bundle)
risk_ref = bundle.get("risk_ref", {}) or {}

# Sidebar
st.sidebar.markdown("## ⚙️ Settings")
input_is_raw = st.sidebar.toggle("🧪 Raw pTPM Input", value=True, help="Apply log1p + scaling transformation")
timepoints_years = st.sidebar.multiselect(
    "📊 Survival Timepoints (years)",
    options=[1, 2, 3, 5, 10],
    default=[1, 2, 3, 5]
)
timepoints_days = [y * 365.25 for y in timepoints_years]

st.sidebar.divider()
st.sidebar.markdown("### 📋 Model Info")
st.sidebar.metric("Features", len(features))
if features:
    with st.sidebar.expander("View Features"):
        st.code("\n".join(features[:20]) + ("\n..." if len(features) > 20 else ""))

if not data_file:
    st.info("👆 Please upload patient data to continue")
    st.stop()

# Read data
try:
    df_in = read_table(data_file)
    df_in = ensure_columns(df_in)
except Exception as e:
    st.error(f"❌ Could not read patient data: {e}")
    st.stop()

# Choose patient id column
id_candidates = [c for c in ["Sample", "Patient_ID", "patient_id", "id"] if c in df_in.columns]
id_col = id_candidates[0] if id_candidates else None

st.markdown("### 📝 Input Data")
left, right = st.columns([3, 1])

with left:
    st.caption("✏️ Edit values directly (use NA for missing)")
    df_edit = st.data_editor(df_in, num_rows="dynamic", use_container_width=True)

with right:
    if id_col:
        st.info(f"🆔 ID Column: `{id_col}`")
    else:
        st.warning("⚠️ No ID column detected")
    
    run_btn = st.button("🚀 Run Predictions", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

# Build X
with st.spinner("🔄 Preprocessing data..."):
    try:
        X = preprocess_for_model(df_edit, bundle=bundle, input_is_raw_ptpm=input_is_raw)
    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        st.stop()

# Predict
with st.spinner("🤖 Running predictions..."):
    try:
        surv_funcs = model.predict_survival_function(X.values, return_array=False)
        risk_scores = model.predict(X.values)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
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
        row[f"S(t={y}y)"] = f"{p:.2%}"
    rows.append(row)

res_df = pd.DataFrame(rows)

st.divider()
st.markdown("### 📊 Prediction Results")

# Color-code risk groups
def color_risk_group(val):
    if val == "Low":
        return 'background-color: #d1fae5; color: #065f46'
    elif val == "Intermediate":
        return 'background-color: #fef3c7; color: #92400e'
    elif val == "High":
        return 'background-color: #fee2e2; color: #991b1b'
    return ''

styled_df = res_df.style.applymap(color_risk_group, subset=['Risk_Group'])
st.dataframe(styled_df, use_container_width=True)

# 3D Visualization
if len(surv_funcs) > 1:
    st.markdown("### 🌐 3D Survival Landscape")
    patient_labels = [df_edit.iloc[i][id_col] if id_col else i for i in range(len(X))]
    fig_3d = create_3d_surface_plot(surv_funcs, timepoints_days, patient_labels)
    st.plotly_chart(fig_3d, use_container_width=True)

# Individual patient analysis
st.markdown("### 📈 Individual Patient Analysis")
sel_options = list(range(len(X)))
if id_col:
    sel_label = df_edit[id_col].astype(str).tolist()
    sel = st.selectbox("Select Patient", options=sel_options, format_func=lambda i: sel_label[i])
else:
    sel = st.selectbox("Select Patient (Row Index)", options=sel_options)

sf = surv_funcs[sel]
pid = df_edit.iloc[sel][id_col] if id_col else sel
risk = float(risk_scores[sel])
risk_group, _ = classify_risk(risk, risk_ref)

# Interactive plot
fig_interactive = create_interactive_survival_curve(sf, pid, risk, risk_group, timepoints_years, timepoints_days)
st.plotly_chart(fig_interactive, use_container_width=True)

# Detail panel with metrics
st.markdown("### 📋 Patient Details")
col1, col2, col3 = st.columns(3)
col1.metric("🆔 Patient ID", str(pid))
col2.metric("⚠️ Risk Score", f"{risk:.4f}")

risk_emoji = {"Low": "🟢", "Intermediate": "🟡", "High": "🔴"}
col3.metric("🎯 Risk Group", f"{risk_emoji.get(risk_group, '⚪')} {risk_group if risk_group else 'N/A'}")

if timepoints_years:
    st.markdown("#### Survival Probabilities")
    probs = survival_prob_at_times(sf, timepoints_days)
    prob_df = pd.DataFrame({
        "Time Point": [f"{y} year{'s' if y > 1 else ''}" for y in timepoints_years],
        "Survival Probability": [f"{p:.2%}" for p in probs],
        "Value": probs
    })
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        fig_bar = px.bar(
            prob_df,
            x="Time Point",
            y="Value",
            text="Survival Probability",
            color="Value",
            color_continuous_scale="Viridis"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False,
            yaxis_range=[0, 1.1],
            yaxis_title="Survival Probability",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_right:
        st.dataframe(prob_df[["Time Point", "Survival Probability"]], use_container_width=True, hide_index=True)

st.success("✅ Analysis complete!")
