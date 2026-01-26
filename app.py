import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional (for zoomable charts). If unavailable, the app falls back gracefully.
try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

st.set_page_config(page_title="RSF Survival Predictor", layout="wide")

# ---------------------------
# Expected proteins list (shown in sidebar)
# ---------------------------
EXPECTED_PROTEINS = [
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

# Fixed suffix behavior (no UI)
DEFAULT_PROTEIN_SUFFIXES = ("_pTPM",)

# ---------------------------
# Modern UI styling (no external deps)
# ---------------------------
st.markdown(
    """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 600px at 20% 10%, rgba(99, 102, 241, 0.18), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(16, 185, 129, 0.16), transparent 55%),
              radial-gradient(900px 700px at 40% 90%, rgba(236, 72, 153, 0.10), transparent 60%),
              linear-gradient(180deg, rgba(15, 23, 42, 0.04), rgba(15, 23, 42, 0.00));
}

/* Buttons */
.stButton button {
  border-radius: 14px !important;
  border: 1px solid rgba(2, 6, 23, 0.10) !important;
  box-shadow: 0 10px 26px rgba(2, 6, 23, 0.10), 0 2px 10px rgba(2, 6, 23, 0.06) !important;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 16px 34px rgba(2, 6, 23, 0.14), 0 6px 16px rgba(2, 6, 23, 0.09) !important;
}

/* Dataframes/editors */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
  border-radius: 16px !important;
  border: 1px solid rgba(2, 6, 23, 0.10);
  box-shadow: 0 12px 30px rgba(2, 6, 23, 0.08);
  overflow: hidden;
}

/* Sidebar (dark) */
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(180deg, rgba(2, 6, 23, 0.94), rgba(2, 6, 23, 0.82));
  color: rgba(255,255,255,0.92) !important;
}
section[data-testid="stSidebar"] * {
  color: rgba(255,255,255,0.92) !important;
}
section[data-testid="stSidebar"] .stSlider > div,
section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stTextInput > div {
  background: rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] code {
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; }
</style>
""",
    unsafe_allow_html=True,
)

ROMAN_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

def parse_stage_ordinal(stage_val) -> float:
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

def _detect_protein_cols(df: pd.DataFrame, suffixes: tuple[str, ...]) -> list[str]:
    """
    Detect protein columns by either:
    - exact match of EXPECTED_PROTEINS (e.g., 'MMP9', 'ADAM15')
    - suffix match (default: '_pTPM')
    """
    cols = []
    exp_set = set(EXPECTED_PROTEINS)
    for c in df.columns:
        cs = str(c)
        if cs in exp_set:
            cols.append(cs)
            continue
        if any(cs.endswith(sfx) for sfx in suffixes):
            cols.append(cs)
    return cols

def preprocess_for_model(df: pd.DataFrame, bundle: dict, input_is_raw_ptpm: bool, protein_suffixes: tuple[str, ...]) -> pd.DataFrame:
    df = ensure_columns(df)
    df = compute_time_event(df)

    if "stage_ordinal" not in df.columns:
        if "stage" in df.columns:
            df["stage_ordinal"] = df["stage"].apply(parse_stage_ordinal)
        else:
            df["stage_ordinal"] = 0.0

    features = extract_features(bundle)
    if not features:
        raise ValueError(
            "Model bundle is missing the feature list. Please re-save the bundle to include one of: "
            "'features' or 'feature_cols'."
        )

    protein_cols = _detect_protein_cols(df, protein_suffixes)
    scaler = extract_scaler(bundle)
    scaler_cols = extract_scaler_cols(bundle, protein_cols)

    # Only apply log1p + scaling to suffix-based columns by default.
    # (Keeps prior behavior safe; exact-name proteins may already be in correct scale depending on your pipeline.)
    if input_is_raw_ptpm:
        suffix_based_cols = [c for c in protein_cols if any(str(c).endswith(sfx) for sfx in protein_suffixes)]
        if suffix_based_cols:
            for col in suffix_based_cols:
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
                    "If your model was trained on log1p + z-scored proteins, results may differ unless you re-save "
                    "the bundle with the fitted scaler or upload already-normalized inputs."
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

def _parse_years_text(text: str) -> list[float]:
    if not text:
        return []
    parts = re.split(r"[,\s]+", text.strip())
    out = []
    for p in parts:
        if not p:
            continue
        try:
            v = float(p)
            if v > 0:
                out.append(v)
        except Exception:
            continue
    return sorted(set(out))

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.title("Random Survival Forest (RSF) — Survival Prediction Interface")
st.caption("Upload your model bundle (.joblib) and a patient table (.xlsx/.csv). Edit values, run inference, and export outputs.")

with st.expander("What you upload", expanded=True):
    st.markdown(
        """
- **Model bundle (.joblib)**: a `joblib.dump()` dictionary containing at least:
  - `model` (a fitted `sksurv.ensemble.RandomSurvivalForest`)
  - `features` (ordered list of feature names used in training)
  - `feature_medians` (dict of medians for imputation)
  - `risk_ref` (optional; contains risk-score quantiles for risk-group labeling)
  - `scaler` and `scaler_cols` (optional but recommended if you trained with z-scored proteins)
- **Patient input (.xlsx or .csv)**: one or more rows matching your training schema (missing values allowed).
        """
    )

colA, colB = st.columns(2)
with colA:
    model_file = st.file_uploader("Upload model bundle (.joblib)", type=["joblib"], accept_multiple_files=False)
with colB:
    data_file = st.file_uploader("Upload patient data (.xlsx or .csv)", type=["xlsx", "xls", "csv"], accept_multiple_files=False)

if not model_file:
    st.stop()

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

st.sidebar.divider()
st.sidebar.subheader("Report survival probability at years")

preset_years = [0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20]
timepoints_years = st.sidebar.multiselect(
    "Presets",
    options=preset_years,
    default=[1, 2, 3, 5, 10],
)
custom_years_text = st.sidebar.text_input(
    "Custom years (comma/space separated)",
    value="",
    placeholder="e.g. 0.25, 6, 8, 13",
)
custom_years = _parse_years_text(custom_years_text)
all_years = sorted(set(timepoints_years + custom_years))
timepoints_days = [y * 365.25 for y in all_years]

st.sidebar.divider()
st.sidebar.subheader("Outputs")
table_height = st.sidebar.slider("Table height (px)", 260, 900, 420, 20)
plot_mode = st.sidebar.radio(
    "Survival curve view",
    options=(["Interactive (zoom)", "Static (matplotlib)"] if _PLOTLY_OK else ["Static (matplotlib)"]),
    index=0,
    help="Interactive mode supports zoom/pan if Plotly is available.",
)
plot_height = st.sidebar.slider("Plot height (px)", 320, 900, 520, 20)

st.sidebar.divider()
st.sidebar.subheader("Features expected (proteins)")
st.sidebar.code("\n".join(EXPECTED_PROTEINS))

if not data_file:
    st.info("Upload a patient data file to proceed.")
    st.stop()

try:
    df_in = read_table(data_file)
    df_in = ensure_columns(df_in)
except Exception as e:
    st.error(f"Could not read patient data: {e}")
    st.stop()

id_candidates = [c for c in ["Sample", "Patient_ID", "patient_id", "id"] if c in df_in.columns]
id_col = id_candidates[0] if id_candidates else None

st.subheader("Input data")
st.caption("Edit values directly, then run predictions. Use NA/blank for missing values.")

if "df_edit" not in st.session_state:
    st.session_state.df_edit = df_in.copy()

tools = st.container()
with tools:
    tcol1, tcol2, tcol3, tcol4 = st.columns([1.1, 1.1, 1.1, 0.9])
    with tcol1:
        st.markdown("**Quick edit (single cell)**")
    with tcol2:
        sel_row = st.number_input("Row index", min_value=0, max_value=max(len(st.session_state.df_edit)-1, 0), value=0, step=1)
    with tcol3:
        sel_col = st.selectbox("Column", options=list(st.session_state.df_edit.columns))
    with tcol4:
        new_val = st.text_input("New value", value="")
    apply_cell = st.button("Apply cell edit", use_container_width=True)

    bcol1, bcol2, bcol3 = st.columns([1.1, 1.1, 0.8])
    with bcol1:
        st.markdown("**Bulk edit (column)**")
        bulk_col = st.selectbox("Bulk column", options=list(st.session_state.df_edit.columns), key="bulk_col")
    with bcol2:
        bulk_mode = st.selectbox("Operation", options=["Set value", "Add number", "Multiply by"], key="bulk_mode")
        bulk_value = st.text_input("Value", value="", key="bulk_value")
    with bcol3:
        apply_scope = st.selectbox("Apply to", options=["All rows", "Only empty cells"], key="apply_scope")
    apply_bulk = st.button("Apply bulk edit", use_container_width=True)

    rcol1, rcol2 = st.columns([1, 1])
    with rcol1:
        reset_btn = st.button("Reset edits", use_container_width=True)
    with rcol2:
        st.download_button(
            "Download edited input (CSV)",
            data=_df_to_csv_bytes(st.session_state.df_edit),
            file_name="edited_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

if reset_btn:
    st.session_state.df_edit = df_in.copy()

if apply_cell:
    df_tmp = st.session_state.df_edit.copy()
    try:
        if new_val.strip() == "":
            df_tmp.at[int(sel_row), sel_col] = np.nan
        else:
            v = pd.to_numeric(pd.Series([new_val]), errors="coerce").iloc[0]
            df_tmp.at[int(sel_row), sel_col] = v if not np.isnan(v) else new_val
        st.session_state.df_edit = df_tmp
    except Exception as e:
        st.warning(f"Cell edit failed: {e}")

if apply_bulk:
    df_tmp = st.session_state.df_edit.copy()
    col = bulk_col
    raw = (bulk_value or "").strip()

    if bulk_mode in ("Add number", "Multiply by"):
        try:
            num = float(raw)
        except Exception:
            st.warning("Bulk operation requires a numeric value.")
            num = None
        if num is not None:
            series = pd.to_numeric(df_tmp[col], errors="coerce")
            mask = series.isna() if apply_scope == "Only empty cells" else pd.Series([True] * len(df_tmp), index=df_tmp.index)

            if bulk_mode == "Add number":
                series.loc[mask] = series.loc[mask].fillna(0.0) + num
            else:
                series.loc[mask] = series.loc[mask].fillna(1.0) * num
            df_tmp[col] = series
            st.session_state.df_edit = df_tmp
    else:
        mask = (df_tmp[col].isna() | (df_tmp[col].astype(str).str.strip() == "")) if apply_scope == "Only empty cells" else pd.Series([True] * len(df_tmp), index=df_tmp.index)
        if raw == "":
            df_tmp.loc[mask, col] = np.nan
        else:
            v = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            df_tmp.loc[mask, col] = v if not np.isnan(v) else raw
        st.session_state.df_edit = df_tmp

col_cfg = {}
for c in st.session_state.df_edit.columns:
    if c in ("time", "days"):
        col_cfg[c] = st.column_config.NumberColumn(c, step=1, format="%.0f")
    elif c in ("event", "status"):
        col_cfg[c] = st.column_config.NumberColumn(c, step=1, format="%.0f", help="Event indicator (0/1).")
    else:
        if pd.api.types.is_numeric_dtype(st.session_state.df_edit[c]):
            col_cfg[c] = st.column_config.NumberColumn(c, step=0.01)

df_edit = st.data_editor(
    st.session_state.df_edit,
    num_rows="dynamic",
    use_container_width=True,
    height=table_height,
    column_config=col_cfg,
    key="data_editor",
)
st.session_state.df_edit = df_edit

st.subheader("Run predictions")
if id_col:
    st.caption(f"Patient identifier column detected: `{id_col}`")
else:
    st.caption("No patient identifier column detected. Predictions will be displayed by row index.")

run_btn = st.button("Predict survival", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

try:
    X = preprocess_for_model(
        df_edit,
        bundle=bundle,
        input_is_raw_ptpm=input_is_raw,
        protein_suffixes=DEFAULT_PROTEIN_SUFFIXES,
    )
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

try:
    surv_funcs = model.predict_survival_function(X.values, return_array=False)
    risk_scores = model.predict(X.values)
except Exception as e:
    st.error(f"Model prediction failed: {e}")
    st.stop()

rows = []
for i in range(len(X)):
    pid = df_edit.iloc[i][id_col] if id_col else i
    sf = surv_funcs[i]
    probs = survival_prob_at_times(sf, timepoints_days) if timepoints_days else []
    risk = float(risk_scores[i])
    risk_group, _ = classify_risk(risk, risk_ref)
    row = {"Patient": pid, "Risk_Score": risk, "Risk_Group": risk_group}
    for y, p in zip(all_years, probs):
        row[f"S(t={y}y)"] = p
    rows.append(row)

res_df = pd.DataFrame(rows)

st.divider()
st.subheader("Predictions")
st.dataframe(res_df, use_container_width=True, height=table_height)

dcol1, dcol2 = st.columns([1, 1])
with dcol1:
    st.download_button(
        "Download predictions (CSV)",
        data=_df_to_csv_bytes(res_df),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dcol2:
    st.download_button(
        "Download X used for inference (CSV)",
        data=_df_to_csv_bytes(pd.DataFrame(X, columns=X.columns)),
        file_name="inference_matrix_X.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.subheader("Survival curve")

n_patients = len(X)
plot_scope = "Single patient" if n_patients == 1 else st.radio(
    "Plot scope",
    options=["Single patient", "All patients"],
    horizontal=True,
)

def _get_xy(sf_obj):
    try:
        xs_ = np.asarray(sf_obj.x, dtype=float)
        ys_ = np.asarray(sf_obj.y, dtype=float)
        return xs_, ys_
    except Exception:
        xs_ = np.linspace(0, 3650, 200)
        ys_ = np.array([sf_obj(t) for t in xs_])
        return xs_, ys_

if plot_scope == "Single patient":
    sel_options = list(range(n_patients))
    if id_col:
        sel_label = df_edit[id_col].astype(str).tolist()
        sel = st.selectbox("Select patient", options=sel_options, format_func=lambda i: sel_label[i])
    else:
        sel = st.selectbox("Select patient (row index)", options=sel_options)

    sf = surv_funcs[sel]
    xs, ys = _get_xy(sf)

    if _PLOTLY_OK and plot_mode.startswith("Interactive"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(shape="hv"), name=str(sel)))
        fig.update_layout(
            height=plot_height,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time (days)",
            yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.02]),
            legend_title_text="Patient",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(xs, ys, where="post")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Selected patient details")
    pid = df_edit.iloc[sel][id_col] if id_col else sel
    risk = float(risk_scores[sel])
    risk_group, _ = classify_risk(risk, risk_ref)

    detail_cols = st.columns(3)
    detail_cols[0].metric("Patient", str(pid))
    detail_cols[1].metric("Risk score", f"{risk:.4f}")
    detail_cols[2].metric("Risk group", risk_group if risk_group else "—")

    if all_years:
        probs = survival_prob_at_times(sf, timepoints_days)
        prob_df = pd.DataFrame({"Year": all_years, "Survival probability": probs})
        st.table(prob_df)

else:
    # All patients in one plot
    show_legend = st.toggle("Show legend", value=False)
    max_legend = st.slider("Legend limit (patients)", 5, 80, 20, 5)

    if _PLOTLY_OK and plot_mode.startswith("Interactive"):
        fig = go.Figure()
        for i in range(n_patients):
            pid = df_edit.iloc[i][id_col] if id_col else i
            xs, ys = _get_xy(surv_funcs[i])
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(shape="hv"),
                    name=str(pid),
                    showlegend=show_legend and (i < max_legend),
                )
            )
        fig.update_layout(
            height=plot_height,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time (days)",
            yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.02]),
            legend_title_text="Patient",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(n_patients):
            pid = df_edit.iloc[i][id_col] if id_col else i
            xs, ys = _get_xy(surv_funcs[i])
            # different colors automatically; legend optional
            if show_legend and (i < max_legend):
                ax.step(xs, ys, where="post", label=str(pid))
            else:
                ax.step(xs, ys, where="post")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(loc="best", fontsize=8)
        st.pyplot(fig, clear_figure=True)

