import io
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="RSF Survival Predictor", layout="wide")

# ---------------------------
# Styling (modern + subtle 3D)
# ---------------------------
st.markdown(
    """
<style>
/* Soft neon-on-dark look with depth */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 800px at 20% 10%, rgba(90,120,255,0.18), rgba(0,0,0,0) 60%),
              radial-gradient(900px 600px at 90% 30%, rgba(255,90,190,0.16), rgba(0,0,0,0) 55%),
              linear-gradient(180deg, rgba(10,12,18,1) 0%, rgba(6,8,12,1) 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Main containers */
.block-container{padding-top: 1.2rem; padding-bottom: 2rem;}

/* Cards */
.rsf-card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 18px 48px rgba(0,0,0,0.45);
  backdrop-filter: blur(12px);
}
.rsf-card:hover{
  transform: translateY(-2px);
  box-shadow: 0 22px 60px rgba(0,0,0,0.55);
  transition: 140ms ease;
}

/* Buttons */
.stButton>button{
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.16);
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.stButton>button:hover{
  transform: translateY(-1px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.45);
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input{
  border-radius: 12px !important;
}

/* Reduce heavy whitespace */
div[data-testid="stVerticalBlock"] > div:has(> .rsf-card){ margin-bottom: 14px; }

/* Small label */
.rsf-muted{opacity:0.8; font-size: 0.92rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Defaults (fallback list if bundle has none)
# ---------------------------
DEFAULT_FEATURE_COLS = [
    # Clinical
    "Age",
    "Stage",
    "Grade",
    "Tumor_Size",
    "Lymph_Nodes",
    "ER",
    "PR",
    "HER2",
    # MMPs
    "MMP1_pTPM",
    "MMP2_pTPM",
    "MMP3_pTPM",
    "MMP7_pTPM",
    "MMP9_pTPM",
    "MMP11_pTPM",
    "MMP14_pTPM",
    # ADAMs
    "ADAM8_pTPM",
    "ADAM9_pTPM",
    "ADAM10_pTPM",
    "ADAM12_pTPM",
    "ADAM17_pTPM",
    # ADAMTS
    "ADAMTS1_pTPM",
    "ADAMTS4_pTPM",
    "ADAMTS5_pTPM",
]


# ---------------------------
# Helpers
# ---------------------------

def _looks_like_model(obj) -> bool:
    return hasattr(obj, "predict") and (
        hasattr(obj, "predict_survival_function")
        or hasattr(obj, "predict_cumulative_hazard_function")
        or hasattr(obj, "predict_proba")
    )


def _load_bundle(uploaded_file) -> Dict:
    bundle = joblib.load(uploaded_file)
    if isinstance(bundle, dict):
        return bundle
    # If a bare model object was saved
    return {"model": bundle}


def _pick_model(bundle: Dict):
    # Common keys
    for k in ["model", "estimator", "final_model", "best_estimator", "rsf", "pipeline"]:
        if k in bundle and _looks_like_model(bundle[k]):
            return bundle[k], k

    # Try to find a model-like object anywhere in dict
    for k, v in bundle.items():
        if _looks_like_model(v):
            return v, k

    # Nothing found
    return None, None


def _protein_cols(cols: List[str]) -> List[str]:
    # Heuristic: treat MMP/ADAM/ADAMTS as proteins if present
    patt = re.compile(r"^(MMP\d+|ADAM\d+|ADAMTS\d+)_pTPM$", re.IGNORECASE)
    return [c for c in cols if patt.search(str(c))]


def _ensure_columns(df: pd.DataFrame, feature_cols: List[str], defaults: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    defaults = defaults or {}
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = defaults.get(c, np.nan)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_preprocess(
    df: pd.DataFrame,
    feature_cols: List[str],
    bundle: Dict,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return X matrix and a cleaned df view."""
    # Defaults/medians for missing values
    fill_values = {}
    for k in ["feature_medians", "train_medians", "medians", "impute_values"]:
        if k in bundle and isinstance(bundle[k], dict):
            fill_values.update(bundle[k])

    df = _ensure_columns(df, feature_cols, defaults=fill_values)
    df = _coerce_numeric(df, feature_cols)

    # Protein transform: log1p if bundle requests it, else keep old behavior (log1p on *_pTPM)
    preprocess = bundle.get("preprocess", {}) if isinstance(bundle.get("preprocess", {}), dict) else {}
    log1p_cols = preprocess.get("log1p_cols")
    if not log1p_cols:
        log1p_cols = _protein_cols(feature_cols)

    for c in log1p_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    # Fill missing after transform
    df = df.fillna({c: fill_values.get(c, df[c].median(skipna=True)) for c in feature_cols})

    # Scaling
    scaler = bundle.get("scaler")
    scaler_cols = bundle.get("scaler_cols") or preprocess.get("scaler_cols") or feature_cols
    if scaler is not None:
        try:
            df.loc[:, scaler_cols] = scaler.transform(df[scaler_cols])
        except Exception:
            # If scaler_cols mismatch, attempt safe intersection
            inter = [c for c in scaler_cols if c in df.columns]
            if inter:
                df.loc[:, inter] = scaler.transform(df[inter])

    # Order columns
    X = df[feature_cols].to_numpy(dtype=float)
    return X, df[feature_cols]


def _default_timepoints(max_years: int = 15) -> List[float]:
    # 0.5y steps up to 5y then 1y steps
    pts = [0.5, 1, 2, 3, 4, 5]
    pts.extend(list(range(6, max_years + 1)))
    return [float(x) for x in sorted(set(pts))]


def _predict_survival(model, X: np.ndarray, years: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times_years, surv_probs) for each row in X.

    - times_years: shape (T,)
    - surv_probs: shape (n_samples, T)
    """
    years_arr = np.array(years, dtype=float)

    if hasattr(model, "predict_survival_function"):
        sf = model.predict_survival_function(X)
        # sf can be list of step functions; each has x (times) and y (surv)
        surv = np.zeros((X.shape[0], len(years_arr)), dtype=float)
        for i, f in enumerate(sf):
            # sksurv StepFunction supports __call__(t)
            surv[i, :] = np.clip(f(years_arr), 0.0, 1.0)
        return years_arr, surv

    if hasattr(model, "predict_cumulative_hazard_function"):
        chf = model.predict_cumulative_hazard_function(X)
        surv = np.zeros((X.shape[0], len(years_arr)), dtype=float)
        for i, f in enumerate(chf):
            surv[i, :] = np.exp(-np.clip(f(years_arr), 0.0, np.inf))
        return years_arr, surv

    raise AttributeError("Loaded model does not support survival function prediction.")


def _risk_percentile(bundle: Dict, risks: np.ndarray) -> Optional[np.ndarray]:
    # If bundle provides training risk distribution, compute percentile
    dist = None
    for k in ["train_risk", "train_risks", "risk_distribution", "risk_dist"]:
        if k in bundle:
            dist = bundle[k]
            break
    if dist is None:
        return None
    dist = np.asarray(dist, dtype=float)
    if dist.size < 5:
        return None
    # Percentile rank
    return np.array([100.0 * (dist <= r).mean() for r in risks], dtype=float)


# ---------------------------
# UI
# ---------------------------

st.markdown(
    """
<div class="rsf-card">
  <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:14px; flex-wrap:wrap;">
    <div>
      <div style="font-size:1.55rem; font-weight:700;">RSF Survival Predictor</div>
      <div class="rsf-muted">Upload a trained survival model bundle (.pkl) and run batch or single-patient predictions.</div>
    </div>
    <div class="rsf-muted">Supports RSF / Cox-style models that implement <code>predict_survival_function</code>.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([0.42, 0.58], gap="large")

with st.sidebar:
    st.markdown("### Model")
    model_file = st.file_uploader("Upload model bundle (.pkl)", type=["pkl", "joblib"])

    st.markdown("### Time horizon")
    max_years = st.slider("Max years", min_value=5, max_value=25, value=15, step=1)
    default_years = _default_timepoints(max_years=max_years)
    years = st.multiselect(
        "Years to report",
        options=default_years,
        default=[y for y in default_years if y in [1.0, 3.0, 5.0, float(max_years)]],
    )
    years = sorted(set([float(y) for y in years])) if years else [1.0, 3.0, 5.0]

    st.markdown("### Output")
    show_curves = st.checkbox("Plot survival curves", value=True)
    show_table = st.checkbox("Show prediction table", value=True)


bundle = None
model = None
model_key = None
feature_cols = DEFAULT_FEATURE_COLS

if model_file is not None:
    try:
        bundle = _load_bundle(model_file)
        model, model_key = _pick_model(bundle)
        feature_cols = bundle.get("feature_cols") or bundle.get("features") or bundle.get("columns") or feature_cols
        feature_cols = list(feature_cols)
    except Exception as e:
        st.error(f"Failed to load bundle: {e}")

if bundle is not None and model is None:
    # Specific help for the common mistake: saving a pandas Series row as the "model"
    if isinstance(bundle.get("model"), pd.Series):
        st.error(
            "The uploaded bundle contains a pandas Series under key 'model' (not a trained estimator). "
            "Re-export your bundle so that 'model' points to the fitted survival model object."
        )
    else:
        st.error(
            "Could not find a fitted model object inside the uploaded bundle. "
            "Expected a key like 'model'/'final_model' holding an estimator with predict_survival_function()."
        )


with col_left:
    st.markdown('<div class="rsf-card">', unsafe_allow_html=True)
    st.markdown("#### Inputs")

    tab_batch, tab_single = st.tabs(["Batch (CSV/Excel)", "Single patient"])

    with tab_batch:
        data_file = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx"], key="data_file")
        if bundle is not None:
            with st.expander("Expected features", expanded=False):
                prot = _protein_cols(feature_cols)
                nonprot = [c for c in feature_cols if c not in prot]
                st.write(f"**Total features:** {len(feature_cols)}")
                if nonprot:
                    st.write("**Clinical / other:**")
                    st.code(", ".join(nonprot))
                if prot:
                    st.write("**Proteins (MMP/ADAM/ADAMTS):**")
                    st.code(", ".join(prot))

        st.caption("Tip: missing columns will be created and filled from bundle medians if available.")

    with tab_single:
        if bundle is None:
            st.info("Upload a model bundle to enable the single-patient form.")
        else:
            search = st.text_input("Search feature", value="")
            shown_cols = feature_cols
            if search.strip():
                s = search.strip().lower()
                shown_cols = [c for c in feature_cols if s in c.lower()]

            # Group proteins vs clinical for readability
            prot = set(_protein_cols(shown_cols))
            clinical = [c for c in shown_cols if c not in prot]
            proteins = [c for c in shown_cols if c in prot]

            single_vals: Dict[str, float] = {}

            with st.expander("Clinical / other", expanded=True):
                for c in clinical:
                    default = None
                    for k in ["feature_medians", "train_medians", "medians", "impute_values"]:
                        if isinstance(bundle.get(k), dict) and c in bundle[k]:
                            default = float(bundle[k][c])
                            break
                    if default is None:
                        default = 0.0
                    single_vals[c] = st.number_input(c, value=float(default), step=1.0 if c.lower() in ["age", "stage", "grade"] else 0.1)

            with st.expander("Proteins (MMP/ADAM/ADAMTS)", expanded=False):
                # Two columns for compactness
                pcols = st.columns(2)
                for i, c in enumerate(proteins):
                    default = None
                    for k in ["feature_medians", "train_medians", "medians", "impute_values"]:
                        if isinstance(bundle.get(k), dict) and c in bundle[k]:
                            default = float(bundle[k][c])
                            break
                    if default is None:
                        default = 0.0
                    with pcols[i % 2]:
                        single_vals[c] = st.number_input(c, value=float(default), step=0.1)

            single_df = pd.DataFrame([single_vals])

    st.markdown("</div>", unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="rsf-card">', unsafe_allow_html=True)
    st.markdown("#### Predictions")

    if bundle is None or model is None:
        st.info("Upload a valid model bundle to run predictions.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        run_batch = st.button("Run batch prediction", use_container_width=True)
        run_single = st.button("Run single-patient prediction", use_container_width=True)

        def _read_table(uploaded) -> pd.DataFrame:
            if uploaded is None:
                raise ValueError("No dataset uploaded.")
            if uploaded.name.lower().endswith(".csv"):
                return pd.read_csv(uploaded)
            return pd.read_excel(uploaded)

        out_df = None
        curves_fig = None

        try:
            if run_batch:
                df_in = _read_table(data_file)
                X, cleaned = _apply_preprocess(df_in, feature_cols, bundle)
                t, s = _predict_survival(model, X, years)

                pred = pd.DataFrame({"id": np.arange(len(df_in))})
                for j, yr in enumerate(t):
                    pred[f"S({yr:g}y)"] = s[:, j]

                # risk score: 1 - S(last)
                risks = 1.0 - s[:, -1]
                pred["risk"] = risks
                pct = _risk_percentile(bundle, risks)
                if pct is not None:
                    pred["risk_percentile"] = pct

                out_df = pd.concat([df_in.reset_index(drop=True), pred], axis=1)

                if show_curves:
                    fig, ax = plt.subplots(figsize=(8.2, 4.6))
                    n_plot = min(30, s.shape[0])
                    for i in range(n_plot):
                        ax.step(t, s[i, :], where="post", alpha=0.35)
                    ax.set_xlabel("Years")
                    ax.set_ylabel("Survival probability")
                    ax.set_ylim(0, 1.02)
                    ax.grid(True, alpha=0.2)
                    ax.set_title(f"Survival curves (showing {n_plot}/{s.shape[0]})")
                    curves_fig = fig

            if run_single:
                X, cleaned = _apply_preprocess(single_df, feature_cols, bundle)
                t, s = _predict_survival(model, X, years)

                pred = {f"S({yr:g}y)": float(s[0, j]) for j, yr in enumerate(t)}
                risk = float(1.0 - s[0, -1])
                pred["risk"] = risk
                pct = _risk_percentile(bundle, np.array([risk]))
                if pct is not None:
                    pred["risk_percentile"] = float(pct[0])

                out_df = pd.DataFrame([pred])

                if show_curves:
                    fig, ax = plt.subplots(figsize=(8.2, 4.6))
                    ax.step(t, s[0, :], where="post")
                    ax.set_xlabel("Years")
                    ax.set_ylabel("Survival probability")
                    ax.set_ylim(0, 1.02)
                    ax.grid(True, alpha=0.2)
                    ax.set_title("Single-patient survival curve")
                    curves_fig = fig

        except Exception as e:
            st.error(f"Prediction failed: {e}")

        if curves_fig is not None:
            st.pyplot(curves_fig)

        if out_df is not None and show_table:
            st.dataframe(out_df, use_container_width=True)

            # Download
            buf = io.BytesIO()
            out_df.to_csv(buf, index=False)
            st.download_button(
                "Download predictions (CSV)",
                data=buf.getvalue(),
                file_name="survival_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Bundle info", expanded=False):
            st.write(f"Model key: **{model_key}**")
            st.write(f"Features: **{len(feature_cols)}**")
            if "cv_summary" in bundle:
                st.write("cv_summary found in bundle.")
            if "feature_importance" in bundle:
                st.write("feature_importance found in bundle.")

        st.markdown("</div>", unsafe_allow_html=True)
