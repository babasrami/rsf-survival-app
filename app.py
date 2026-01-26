
import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="RSF Survival Predictor", layout="wide")

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
st.sidebar.write(f"Features expected: {len(features)}")
if features:
    st.sidebar.code("\n".join(features[:20]) + ("\n..." if len(features) > 20 else ""))

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
st.subheader("Predictions")
st.dataframe(res_df, use_container_width=True)

# Plot per-patient survival curve
st.subheader("Survival curve")
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

fig, ax = plt.subplots()
ax.step(xs, ys, where="post")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival probability")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
st.pyplot(fig, clear_figure=True)

# Detail panel
st.subheader("Selected patient details")
pid = df_edit.iloc[sel][id_col] if id_col else sel
risk = float(risk_scores[sel])
risk_group, _ = classify_risk(risk, risk_ref)

detail_cols = st.columns(3)
detail_cols[0].metric("Patient", str(pid))
detail_cols[1].metric("Risk score", f"{risk:.4f}")
detail_cols[2].metric("Risk group", risk_group if risk_group else "—")

if timepoints_years:
    probs = survival_prob_at_times(sf, timepoints_days)
    prob_df = pd.DataFrame({"Year": timepoints_years, "Survival probability": probs})
    st.table(prob_df)
