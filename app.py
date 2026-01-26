
import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="RSF Survival Predictor", layout="wide")

# -----------------------------
# Fixed protein/ADAM features (UI display)
# -----------------------------
PROTEIN_FEATURES_DISPLAY = [
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

# Session state keys
if "_pred" not in st.session_state:
    st.session_state["_pred"] = None  # stores computed predictions so plot settings update without re-click

# -----------------------------
# UI theme / styling
# -----------------------------
st.markdown(
    """
    <style>
      /* Dark sidebar */
      section[data-testid="stSidebar"] > div {
        background: radial-gradient(900px 600px at 10% 10%, rgba(255,255,255,0.08), rgba(0,0,0,0.0)),
                    linear-gradient(180deg, #0b1020 0%, #070a12 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }
      section[data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.92);
      }
      /* Cards */
      .rsf-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
      }
      .rsf-muted { color: rgba(255,255,255,0.70); }
    </style>
    """,
    unsafe_allow_html=True,
)

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
timepoints_years = st.sidebar.multiselect("Report survival probability at years:",
                                         options=[1,2,3,5,10],
                                         default=[1,2,3,5])
timepoints_days = [y * 365.25 for y in timepoints_years]

st.sidebar.divider()
st.sidebar.subheader("Bundle summary")
st.sidebar.write(f"Model features in bundle: {len(features)}")
st.sidebar.subheader("Protein/ADAM features expected")
st.sidebar.code("\n".join(PROTEIN_FEATURES_DISPLAY))

if not data_file:
    st.info("Upload a patient data file to proceed.")
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
    st.subheader("Input data (editable)")
    st.caption("You can edit values directly below; use NA/blank for missing values.")
    df_edit = st.data_editor(df_in, num_rows="dynamic", use_container_width=True)

with right:
    st.subheader("Run predictions")
    if id_col:
        st.caption(f"Patient identifier column detected: `{id_col}`")
    else:
        st.caption("No patient identifier column detected. Predictions will be displayed by row index.")

    run_btn = st.button("Predict survival", type="primary", use_container_width=True)

# Compute predictions only when the button is clicked. After that, keep results in
# session_state so changing plot/table settings updates instantly.
if run_btn:
    try:
        X = preprocess_for_model(df_edit, bundle=bundle, input_is_raw_ptpm=input_is_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.session_state["_pred"] = None
    else:
        try:
            surv_funcs = model.predict_survival_function(X.values, return_array=False)
            risk_scores = model.predict(X.values)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.session_state["_pred"] = None
        else:
            st.session_state["_pred"] = {
                "X": X,
                "surv_funcs": surv_funcs,
                "risk_scores": risk_scores,
                "id_col": id_col,
                "df": df_edit.copy(),
            }

pred = st.session_state.get("_pred")
if pred is None:
    st.info("Upload data, edit if needed, then click **Predict survival**.")
    st.stop()

X = pred["X"]
surv_funcs = pred["surv_funcs"]
risk_scores = pred["risk_scores"]
id_col = pred["id_col"]
df_used = pred["df"]

# Results table (recomputed from stored survival functions so it updates with timepoint settings)
rows = []
for i in range(len(X)):
    pid = df_used.iloc[i][id_col] if id_col else i
    sf_i = surv_funcs[i]
    probs = survival_prob_at_times(sf_i, timepoints_days) if timepoints_days else []
    risk = float(risk_scores[i])
    risk_group, _ = classify_risk(risk, risk_ref)
    row = {"Patient": pid, "Risk_Score": risk, "Risk_Group": risk_group}
    for y, p in zip(timepoints_years, probs):
        row[f"S(t={y}y)"] = p
    rows.append(row)
res_df = pd.DataFrame(rows)

st.divider()
st.subheader("Predictions")
st.dataframe(res_df, use_container_width=True)
st.download_button(
    "Download predictions as CSV",
    data=res_df.to_csv(index=False).encode("utf-8"),
    file_name="survival_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)

# Survival curves
st.subheader("Survival curve")

plot_scope = "Single patient"
if len(X) > 1:
    st.caption("Plot scope")
    plot_scope = st.radio(
        "Plot scope",
        options=["Single patient", "All patients"],
        horizontal=True,
        label_visibility="collapsed",
    )

show_legend = True
legend_limit = 20
if plot_scope == "All patients":
    # Keep the legend always on, but allow limiting how many patient labels are shown.
    max_leg = min(200, len(X))
    legend_limit = st.slider(
        "Legend limit (patients)",
        min_value=1,
        max_value=max_leg,
        value=min(20, max_leg),
    )

def _step_xy(step_fn, fallback_days=3650):
    try:
        return step_fn.x, step_fn.y
    except Exception:
        xs_ = np.linspace(0, fallback_days, 200)
        ys_ = np.array([step_fn(t) for t in xs_])
        return xs_, ys_

fig, ax = plt.subplots()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival probability")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

selected_index = 0
if plot_scope == "Single patient":
    sel_options = list(range(len(X)))
    if id_col:
        labels = df_used[id_col].astype(str).tolist()
        selected_index = st.selectbox("Select patient", options=sel_options, format_func=lambda i: labels[i])
    else:
        selected_index = st.selectbox("Select patient (row index)", options=sel_options)

    xs, ys = _step_xy(surv_funcs[selected_index], fallback_days=int(max(timepoints_days + [3650])))
    ax.step(xs, ys, where="post")

else:
    # Plot all patients with default matplotlib color cycle
    fallback = int(max(timepoints_days + [3650]))
    for i in range(len(X)):
        xs, ys = _step_xy(surv_funcs[i], fallback_days=fallback)
        label = None
        if show_legend and i < legend_limit:
            label = str(df_used.iloc[i][id_col]) if id_col else f"{i}"
        ax.step(xs, ys, where="post", label=label)
    if show_legend:
        ax.legend(loc="best", fontsize=8)

st.pyplot(fig, use_container_width=True)

# Download plot
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
buf.seek(0)
st.download_button(
    "Download plot (PNG)",
    data=buf.getvalue(),
    file_name="survival_curve.png",
    mime="image/png",
    use_container_width=True,
)

# Detail panel (single-patient only)
if plot_scope == "Single patient":
    st.subheader("Selected patient details")
    pid = df_used.iloc[selected_index][id_col] if id_col else selected_index
    risk = float(risk_scores[selected_index])
    risk_group, _ = classify_risk(risk, risk_ref)

    detail_cols = st.columns(3)
    detail_cols[0].metric("Patient", str(pid))
    detail_cols[1].metric("Risk score", f"{risk:.4f}")
    detail_cols[2].metric("Risk group", risk_group if risk_group else "—")

    if timepoints_years:
        probs = survival_prob_at_times(surv_funcs[selected_index], timepoints_days)
        prob_df = pd.DataFrame({"Year": timepoints_years, "Survival probability": probs})
        st.table(prob_df)
