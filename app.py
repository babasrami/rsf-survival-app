
import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import hashlib

st.set_page_config(page_title="RSF Survival Predictor", layout="wide")

st.markdown(
    """
<style>
/* Dark sidebar */
[data-testid="stSidebar"]{
  background: radial-gradient(1200px 600px at 30% 10%, rgba(90,92,255,0.25), rgba(0,0,0,0)) ,
              linear-gradient(180deg, #0b1220 0%, #070b12 100%);
}
[data-testid="stSidebar"] * { color: #e8eefc; }

/* 3D-ish cards */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}

/* Make main page background subtle */
.stApp {
  background: radial-gradient(1100px 700px at 15% 10%, rgba(88,101,242,0.18), rgba(0,0,0,0)),
              radial-gradient(900px 600px at 85% 25%, rgba(16,185,129,0.14), rgba(0,0,0,0));
}

/* Tighter data editor */
[data-testid="stDataFrame"]{
  border-radius: 12px;
  overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------

# Protein panel to expose in the UI (editable in the input table)
PROTEIN_PANEL = [
    "ADAM15",
    "ADAMTS8",
    "MMP7",
    "MMP15",
    "ADAMTSL1",
    "MMP13",
    "MMP1",
    "MMP12",
    "MMP23",
    "MMP26",
    "ADAMTS7",
    "MMP28",
    "MMP9",
    "MMP25",
]
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

def hash_dataframe(df: pd.DataFrame) -> str:
    """Stable hash for detecting when the editable input changes."""
    # Use CSV bytes for deterministic hashing
    data = df.to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.md5(data).hexdigest()

def ensure_ui_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the editable table exposes the expected clinical columns and protein panel."""
    df = ensure_columns(df)
    df = df.copy()

    # Create a stable id column if missing
    id_col = detect_id_column(df)
    if id_col is None:
        df.insert(0, "patient_id", [f"P{i+1}" for i in range(len(df))])
        id_col = "patient_id"

    clinical_cols = [
        "age",
        "stage",
        "tumor_size",
        "lymph_nodes",
        "grade",
        "er_status",
        "pr_status",
        "her2_status",
    ]
    required = [c for c in clinical_cols if c not in [id_col]] + PROTEIN_PANEL

    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # Reorder: id, clinical, proteins, then everything else
    ordered = [id_col] + [c for c in clinical_cols if c in df.columns and c != id_col] + [p for p in PROTEIN_PANEL if p in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    return df

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
st.title("Random Survival Forest (RSF) — Survival Prediction Interface")

with st.expander("What you upload", expanded=True):
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
    model_file = st.file_uploader("Upload model bundle (.joblib)", type=["joblib"], accept_multiple_files=False)
with colB:
    data_file = st.file_uploader("Upload patient data (.xlsx or .csv)", type=["xlsx", "xls", "csv"], accept_multiple_files=False)

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

st.sidebar.header("Inference options")
input_is_raw = st.sidebar.toggle("Inputs are raw pTPM (apply log1p + scaling if available)", value=True)
timepoints_years = st.sidebar.multiselect(
    "Report survival probability at years:",
    options=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30],
    default=[1, 2, 3, 5],
)
timepoints_days = [y * 365.25 for y in timepoints_years]

st.sidebar.divider()
st.sidebar.subheader("Bundle summary")
st.sidebar.write("Protein panel")
st.sidebar.code("\n".join(PROTEIN_PANEL))

with st.sidebar.expander("Model feature count", expanded=False):
    st.write(f"Total features in uploaded model: {len(features)}")

if not data_file:
    st.info("Upload a patient data file to proceed.")
    st.stop()

# Read data
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.last_pred_hash = None
    st.session_state.pred_store = {}

try:
    file_bytes = data_file.getvalue()
    file_sig = (data_file.name, len(file_bytes), hashlib.md5(file_bytes).hexdigest())
except Exception:
    file_sig = (getattr(data_file, "name", "uploaded"), None, None)

if st.session_state.get("data_sig") != file_sig:
    try:
        df_in = read_table(data_file)
        df_in = ensure_ui_columns(df_in)
    except Exception as e:
        st.error(f"Could not read patient data: {e}")
        st.stop()
    st.session_state.data_sig = file_sig
    st.session_state.df_edit = df_in
    st.session_state.predicted = False
    st.session_state.last_pred_hash = None
    st.session_state.pred_store = {}

df_current = st.session_state.df_edit.copy()
id_col = detect_id_column(df_current)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input data (editable)")
    st.caption("Edit values directly. If you change the input data, click Predict again.")

    df_edit = st.data_editor(
        df_current,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

    # Optional bulk edit tools (does not affect model logic; only edits the table)
    with st.expander("Bulk edits", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            bulk_col = st.selectbox("Column", options=list(df_edit.columns), index=0, key="bulk_col")
        with c2:
            bulk_op = st.selectbox("Operation", options=["Set", "Add", "Multiply"], key="bulk_op")
        with c3:
            bulk_val = st.number_input("Value", value=0.0, step=0.1, key="bulk_val")

        apply_bulk = st.button("Apply to all rows", use_container_width=True, key="bulk_apply")
        if apply_bulk and bulk_col:
            try:
                s = pd.to_numeric(df_edit[bulk_col], errors="coerce")
                if bulk_op == "Set":
                    df_edit[bulk_col] = bulk_val
                elif bulk_op == "Add":
                    df_edit[bulk_col] = (s.fillna(0) + bulk_val)
                else:
                    df_edit[bulk_col] = (s.fillna(0) * bulk_val)
            except Exception:
                df_edit[bulk_col] = bulk_val

            # Persist and refresh the editor view
            st.session_state.df_edit = df_edit
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Run predictions")
    if id_col:
        st.caption(f"Patient identifier column detected: `{id_col}`")
    else:
        st.caption("No patient identifier column detected. Predictions will be displayed by row index.")

    edited_csv = df_edit.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download edited input (CSV)",
        data=edited_csv,
        file_name="edited_input.csv",
        mime="text/csv",
        use_container_width=True,
    )

    current_hash = hash_dataframe(df_edit)
    if st.session_state.predicted and st.session_state.last_pred_hash != current_hash:
        st.session_state.predicted = False
        st.session_state.pred_store = {}
        st.info("Input data changed. Click Predict to refresh results.")

    run_btn = st.button("Predict survival", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.session_state.df_edit = df_edit

# Compute predictions only when requested. Plot settings update live after prediction.
if run_btn:
    try:
        X = preprocess_for_model(df_edit, bundle=bundle, input_is_raw_ptpm=input_is_raw)
        surv_funcs = model.predict_survival_function(X.values, return_array=False)
        risk_scores = np.asarray(model.predict(X.values), dtype=float)

        if id_col:
            id_values = df_edit[id_col].astype(str).tolist()
        else:
            id_values = [str(i) for i in range(len(df_edit))]

        st.session_state.pred_store = {
            "surv_funcs": surv_funcs,
            "risk_scores": risk_scores,
            "id_values": id_values,
            "id_col": id_col,
        }
        st.session_state.predicted = True
        st.session_state.last_pred_hash = current_hash
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.session_state.predicted = False
        st.session_state.pred_store = {}

if not st.session_state.get("predicted"):
    st.divider()
    st.info("Click **Predict survival** to compute results. Plot settings update live after prediction.")
    st.stop()

# Pull stored outputs (so plot controls do not reset the app)
surv_funcs = st.session_state.pred_store["surv_funcs"]
risk_scores = st.session_state.pred_store["risk_scores"]
id_values = st.session_state.pred_store["id_values"]

# Results table (timepoints can change live)
rows = []
for i in range(len(id_values)):
    sf = surv_funcs[i]
    probs = survival_prob_at_times(sf, timepoints_days) if timepoints_days else []
    risk = float(risk_scores[i])
    risk_group, _ = classify_risk(risk, risk_ref)
    row = {"Patient": id_values[i], "Risk_Score": risk, "Risk_Group": risk_group}
    for y, p in zip(timepoints_years, probs):
        row[f"S(t={y}y)"] = p
    rows.append(row)

res_df = pd.DataFrame(rows)

st.divider()
st.subheader("Predictions")
st.dataframe(res_df, use_container_width=True)

csv_preds = res_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions (CSV)",
    data=csv_preds,
    file_name="survival_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)

# Survival plot (interactive zoom)
st.subheader("Survival curve")

plot_scope = "Single patient"
if len(id_values) > 1:
    plot_scope = st.radio(
        "Plot scope",
        ["Single patient", "All patients"],
        horizontal=True,
        key="plot_scope",
    )

show_legend = st.toggle("Show legend", value=True, key="show_legend") if plot_scope == "All patients" else False
legend_limit = None
if plot_scope == "All patients":
    legend_limit = st.slider("Legend limit (patients)", 1, min(200, len(id_values)), min(20, len(id_values)), key="legend_limit")

if plot_scope == "Single patient":
    sel = st.selectbox("Select patient", options=list(range(len(id_values))), format_func=lambda i: id_values[i], key="sel_patient")
    to_plot = [(sel, id_values[sel])]
else:
    to_plot = list(enumerate(id_values))

fig = go.Figure()
for idx, label in to_plot:
    sf = surv_funcs[idx]
    # StepFunction has x/y; if not, sample a grid
    try:
        xs = np.asarray(sf.x, dtype=float)
        ys = np.asarray(sf.y, dtype=float)
    except Exception:
        xs = np.linspace(0, 365.25 * max(timepoints_years + [10]), 400)
        ys = np.array([sf(t) for t in xs], dtype=float)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=label,
            line_shape="hv",
            showlegend=(plot_scope == "All patients" and show_legend and (legend_limit is None or idx < legend_limit)),
        )
    )

fig.update_layout(
    height=520,
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis_title="Time (days)",
    yaxis_title="Survival probability",
    yaxis=dict(range=[0, 1.02]),
    legend=dict(orientation="h") if plot_scope == "All patients" else dict(),
)

st.plotly_chart(fig, use_container_width=True)

# Detail panel (single patient)
if plot_scope == "Single patient":
    st.subheader("Selected patient details")
    risk = float(risk_scores[sel])
    risk_group, _ = classify_risk(risk, risk_ref)
    detail_cols = st.columns(3)
    detail_cols[0].metric("Patient", str(id_values[sel]))
    detail_cols[1].metric("Risk score", f"{risk:.4f}")
    detail_cols[2].metric("Risk group", risk_group if risk_group else "—")

    if timepoints_years:
        probs = survival_prob_at_times(surv_funcs[sel], timepoints_days)
        prob_df = pd.DataFrame({"Year": timepoints_years, "Survival probability": probs})
        st.table(prob_df)
