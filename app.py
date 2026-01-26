import io
import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🧬",
    layout="wide",
)


st.markdown(
    """
    <style>
      /* Background + typography */
      .stApp {
        background:
          radial-gradient(1200px circle at 15% 8%, rgba(124, 58, 237, 0.22), transparent 42%),
          radial-gradient(900px circle at 90% 12%, rgba(59, 130, 246, 0.18), transparent 45%),
          radial-gradient(900px circle at 70% 90%, rgba(16, 185, 129, 0.16), transparent 50%),
          linear-gradient(180deg, rgba(2, 6, 23, 1) 0%, rgba(3, 7, 18, 1) 100%);
      }

      /* Make the header area cleaner */
      header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
      }

      /* Container cards (3D) */
      .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        box-shadow:
          0 10px 30px rgba(0,0,0,0.30),
          inset 0 1px 0 rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transform: translateZ(0);
      }

      .card:hover {
        border-color: rgba(255,255,255,0.16);
        box-shadow:
          0 14px 40px rgba(0,0,0,0.38),
          inset 0 1px 0 rgba(255,255,255,0.10);
      }

      .hero-title {
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.1rem 0 0.35rem 0;
      }

      .hero-sub {
        color: rgba(255,255,255,0.75);
        margin: 0 0 0.6rem 0;
        line-height: 1.35;
      }

      .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.82);
        font-size: 0.80rem;
        margin-right: 8px;
      }

      /* Sidebar tweaks */
      section[data-testid="stSidebar"] {
        background: rgba(2,6,23,0.65);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Reduce top padding in main body */
      .block-container { padding-top: 1.0rem; }

      /* Data editor styling */
      div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
      }

      /* Metric cards */
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 10px 12px;
        box-shadow:
          0 10px 26px rgba(0,0,0,0.26),
          inset 0 1px 0 rgba(255,255,255,0.08);
      }

      /* Buttons */
      .stDownloadButton button, .stButton button {
        border-radius: 12px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Defaults + helpers
# ----------------------------


DEFAULT_FEATURE_COLS = [
    "Age",
    "NPI",
    "ER_Expr",
    "PR_Expr",
    "HER2_Expr",
    "Histological_Grade",
    "Tumor_Size",
    "Lymph_Nodes",
    "Survival_Days",
    "Survival_Status",
    "MMP1_pTPM",
    "MMP2_pTPM",
    "MMP7_pTPM",
    "MMP9_pTPM",
    "MMP11_pTPM",
    "MMP14_pTPM",
    "ADAM8_pTPM",
    "ADAM9_pTPM",
    "ADAM10_pTPM",
    "ADAM12_pTPM",
    "ADAM17_pTPM",
]


EXTRA_PROTEIN_LIBRARY = [
    # MMPs
    "MMP3_pTPM",
    "MMP8_pTPM",
    "MMP10_pTPM",
    "MMP12_pTPM",
    "MMP13_pTPM",
    "MMP15_pTPM",
    "MMP16_pTPM",
    "MMP19_pTPM",
    "MMP20_pTPM",
    "MMP21_pTPM",
    "MMP23A_pTPM",
    "MMP23B_pTPM",
    "MMP24_pTPM",
    "MMP25_pTPM",
    "MMP26_pTPM",
    "MMP27_pTPM",
    "MMP28_pTPM",
    # ADAMs
    "ADAM15_pTPM",
    "ADAM19_pTPM",
    "ADAM28_pTPM",
    "ADAM33_pTPM",
]


def _parse_years(text: str) -> list[float]:
    """Parse a comma/space separated list of years."""
    if not text:
        return []
    parts = re.split(r"[\s,;]+", text.strip())
    years: list[float] = []
    for p in parts:
        if not p:
            continue
        try:
            years.append(float(p))
        except ValueError:
            continue
    years = sorted({y for y in years if y > 0})
    return years


@st.cache_resource
def load_bundle(bundle_file) -> dict:
    return joblib.load(bundle_file)


def get_feature_cols(bundle: dict) -> list[str]:
    if isinstance(bundle, dict) and "feature_cols" in bundle and isinstance(bundle["feature_cols"], (list, tuple)):
        cols = [str(c) for c in bundle["feature_cols"]]
        return cols
    return DEFAULT_FEATURE_COLS


def ensure_columns(df: pd.DataFrame, required: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in required if c not in df.columns]
    out = df.copy()
    for c in missing:
        out[c] = np.nan
    return out, missing


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fill_missing_with_medians(df: pd.DataFrame, medians: dict[str, float], cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in medians and pd.notna(medians[c]):
            out[c] = out[c].fillna(medians[c])
        else:
            out[c] = out[c].fillna(0)
    return out


def risk_score(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        return model.predict(x)
    raise ValueError("Model does not support prediction.")


def survival_function(rsf_model, x: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Return matrix (n_samples, len(times)) of survival probabilities."""
    surv_funcs = rsf_model.predict_survival_function(x)
    mat = np.zeros((x.shape[0], len(times)))
    for i, sf in enumerate(surv_funcs):
        mat[i, :] = sf(times)
    return mat


def times_in_days(years: float, n: int = 400) -> np.ndarray:
    return np.linspace(0, years * 365.0, n)


@dataclass
class AppState:
    feature_cols: list[str]
    medians: dict[str, float]


def build_state(bundle: dict) -> AppState:
    fcols = get_feature_cols(bundle)
    med = bundle.get("medians", {}) if isinstance(bundle, dict) else {}
    if not isinstance(med, dict):
        med = {}
    return AppState(feature_cols=fcols, medians=med)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------
# Header
# ----------------------------


st.markdown(
    """
    <div class="card">
      <span class="pill">Random Survival Forest</span>
      <span class="pill">Interactive curves</span>
      <span class="pill">Editable dataset</span>
      <div class="hero-title">Breast Cancer Survival Predictor</div>
      <div class="hero-sub">
        Upload your trained model bundle and a patient dataset. Edit values in-app, run predictions, zoom into curves, and export results.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ----------------------------
# Sidebar: inputs
# ----------------------------


with st.sidebar:
    st.markdown("### Inputs")
    model_file = st.file_uploader("Model bundle (.pkl)", type=["pkl"], help="joblib dump containing RSF and scaler")
    data_file = st.file_uploader("Patient dataset (.csv)", type=["csv"], help="CSV file with clinical + protein features")

    st.markdown("---")
    st.markdown("### Time horizon")
    years_max = st.slider("Plot range (years)", min_value=1, max_value=50, value=10, step=1)
    st.caption("Curve time axis is in years (internally converted to days).")

    st.markdown("---")
    st.markdown("### Output years")
    default_years = [1, 2, 3, 5, 10]
    years_selected = st.multiselect("Compute survival at (years)", options=sorted(set(default_years + [15, 20, 25, 30])), default=default_years)
    extra_years_text = st.text_input("Add more years (comma/space separated)", value="")
    extra_years = _parse_years(extra_years_text)
    years_selected = sorted(set([float(y) for y in years_selected] + extra_years))

    st.markdown("---")
    st.markdown("### Add columns (optional)")
    st.caption("You can add extra protein/ADAM columns to your dataset for tracking/editing. Only columns used by the model affect predictions.")
    add_cols = st.multiselect(
        "Add protein features",
        options=sorted(set(EXTRA_PROTEIN_LIBRARY)),
        default=[],
    )


# ----------------------------
# Load model bundle
# ----------------------------


if model_file is None:
    st.info("Upload a model bundle (.pkl) to begin.")
    st.stop()

bundle = load_bundle(model_file)
state = build_state(bundle)

rsf_model = bundle.get("rsf_model")
scaler = bundle.get("scaler")

if rsf_model is None or scaler is None:
    st.error("Model bundle must contain 'rsf_model' and 'scaler'.")
    st.stop()


# ----------------------------
# Load patient data
# ----------------------------


if data_file is None:
    st.info("Upload a patient dataset (.csv).")
    st.stop()

try:
    raw_df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

df = raw_df.copy()
if add_cols:
    for c in add_cols:
        if c not in df.columns:
            df[c] = np.nan

df, missing = ensure_columns(df, state.feature_cols)

# Keep a stable editable copy in session state
if "editable_df" not in st.session_state:
    st.session_state.editable_df = df.copy()
else:
    # if a new file is uploaded, refresh editable_df (best-effort)
    if getattr(st.session_state, "_last_upload_shape", None) != df.shape:
        st.session_state.editable_df = df.copy()
st.session_state._last_upload_shape = df.shape


# ----------------------------
# Main tabs
# ----------------------------


tab_edit, tab_predict, tab_about = st.tabs(["Edit dataset", "Predict & export", "Notes"])


with tab_edit:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Edit your uploaded dataset")
    if missing:
        st.warning(
            "Missing required columns were added as empty fields: " + ", ".join(missing)
        )

    st.caption(
        "Edit values directly in the table. Use the buttons below to reset, download the edited data, or continue to prediction."
    )

    # Column config: make numeric columns editable with proper formatting
    col_cfg = {}
    for c in st.session_state.editable_df.columns:
        if c in state.feature_cols or c in add_cols:
            col_cfg[c] = st.column_config.NumberColumn(c, format="%.4f")

    edited = st.data_editor(
        st.session_state.editable_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config=col_cfg,
        key="data_editor",
        height=520,
    )
    st.session_state.editable_df = edited

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("Reset to uploaded", use_container_width=True):
            st.session_state.editable_df = df.copy()
            st.rerun()
    with c2:
        st.download_button(
            "Download edited CSV",
            data=df_to_csv_bytes(st.session_state.editable_df),
            file_name="edited_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Download template (model columns)",
            data=df_to_csv_bytes(pd.DataFrame(columns=state.feature_cols)),
            file_name="template_model_columns.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c4:
        st.caption(
            "Tip: Columns not used by the model are kept for tracking, but will not change predictions unless they are in the model's feature list."
        )

    st.markdown("</div>", unsafe_allow_html=True)


with tab_predict:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Run survival predictions")

    work_df = st.session_state.editable_df.copy()
    work_df = ensure_columns(work_df, state.feature_cols)[0]
    work_df = coerce_numeric(work_df, state.feature_cols)
    work_df = fill_missing_with_medians(work_df, state.medians, state.feature_cols)

    x = work_df[state.feature_cols]
    x_scaled = scaler.transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=state.feature_cols, index=work_df.index)

    # Run predictions
    risk = risk_score(rsf_model, x_scaled_df)
    times = times_in_days(years_max, n=500)
    surv_mat = survival_function(rsf_model, x_scaled_df, times)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Patients", f"{len(work_df):,}")
    with m2:
        st.metric("Features used", f"{len(state.feature_cols):,}")
    with m3:
        st.metric("Plot horizon", f"{years_max} years")
    with m4:
        st.metric("Output years", f"{len(years_selected)}")

    st.write("")

    # Interactive curve plot (Plotly) with zoom
    try:
        import plotly.graph_objects as go

        st.markdown("#### Survival curves (zoom/pan enabled)")
        n_plot = st.slider("Number of patients to plot", min_value=1, max_value=min(25, len(work_df)), value=min(5, len(work_df)))
        idxs = list(work_df.index[:n_plot])

        fig = go.Figure()
        t_years = times / 365.0
        for i in idxs:
            row_pos = list(work_df.index).index(i)
            fig.add_trace(
                go.Scatter(
                    x=t_years,
                    y=surv_mat[row_pos, :],
                    mode="lines",
                    line_shape="hv",
                    name=f"Patient {i}",
                )
            )

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis_title="Time (years)",
            yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.02]),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "toImageButtonOptions": {"format": "png", "filename": "survival_curves"},
            },
        )

        # Download curve data
        curve_df = pd.DataFrame({"time_years": t_years})
        for i in idxs:
            row_pos = list(work_df.index).index(i)
            curve_df[f"patient_{i}"] = surv_mat[row_pos, :]
        st.download_button(
            "Download plotted curve data (CSV)",
            data=df_to_csv_bytes(curve_df),
            file_name="survival_curve_data.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Interactive plot unavailable (Plotly error: {e}).")

    st.write("")
    st.markdown("#### Per-patient survival at selected years")

    # Survival at specified years
    years_selected = [float(y) for y in years_selected]
    year_times = np.array([y * 365.0 for y in years_selected], dtype=float)
    year_surv = survival_function(rsf_model, x_scaled_df, year_times)

    out = pd.DataFrame({"patient_index": work_df.index, "risk_score": risk})
    for j, y in enumerate(years_selected):
        out[f"survival_{y:g}y"] = year_surv[:, j]

    # Sorting and filtering
    c1, c2 = st.columns([1, 2])
    with c1:
        sort_by = st.selectbox("Sort by", options=["risk_score"] + [c for c in out.columns if c.startswith("survival_")], index=0)
    with c2:
        ascending = st.toggle("Ascending", value=False)

    out_sorted = out.sort_values(by=sort_by, ascending=ascending)
    st.dataframe(out_sorted, use_container_width=True, height=360)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        st.download_button(
            "Download results (CSV)",
            data=df_to_csv_bytes(out_sorted),
            file_name="survival_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b2:
        st.download_button(
            "Download scaled features (CSV)",
            data=df_to_csv_bytes(x_scaled_df),
            file_name="scaled_features.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b3:
        st.caption(
            "Exports include: edited dataset, scaled features, prediction table, and curve data for the plotted patients."
        )

    st.markdown("</div>", unsafe_allow_html=True)


with tab_about:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Notes")
    st.markdown(
        """
        - **Editing:** the "Edit dataset" tab lets you change any uploaded values, add rows, and then predict using the edited table.
        - **Missing values:** required model features are coerced to numeric; missing values are filled using bundle medians when available, otherwise 0.
        - **Extra proteins/ADAMs:** you can add extra columns to track them in the dataset editor. Predictions only use columns present in the model bundle's feature list.
        - **Zoom:** survival curves are plotted with Plotly for pan/zoom and image export from the chart toolbar.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
