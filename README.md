# RSF Survival Predictor — Simple Upload-and-Predict UI (Streamlit)

This app provides a clinician-friendly interface for a **trained Random Survival Forest (RSF)** survival model:
- Upload a **model bundle** (`.joblib`)
- Upload a **patient input file** (`.xlsx` or `.csv`) in the *same schema as training*
- Edit values (including **NA / blanks**) directly in the UI
- Get per-patient:
  - Survival curve
  - Survival probabilities at selected timepoints (e.g., 1/2/3/5 years)
  - Risk score and optional risk-group classification

## 1) Files included
- `app.py` — Streamlit application
- `patient_input_template.xlsx` / `patient_input_template.csv` — input templates (same columns as `Data.xlsx`)
- `requirements.txt` — Python dependencies

## 2) Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3) What the model bundle must contain
The app expects a `joblib.dump()` dictionary with at least:

- `model`: fitted `sksurv.ensemble.RandomSurvivalForest`
- `features`: **ordered** list of feature names used for training
- `feature_medians`: dict `{feature_name: median_value}` for NA imputation
- `risk_ref` (optional): dict with `q33`, `q66`, and optionally the training risk scores

### Strongly recommended (if you trained with z-scored proteins)
In your training notebook you did:
1) `log1p` on protein columns  
2) `StandardScaler().fit_transform(...)` on protein columns

For faithful inference on **raw pTPM input**, you should also save:
- `scaler`: the fitted `StandardScaler`
- `scaler_cols`: list of protein columns the scaler was fit on (e.g., `ptpm_cols`)

Example save snippet (matches your notebook logic):

```python
import joblib, os, numpy as np

feature_list = list(best_features)
feature_medians = tmp[feature_list].median(numeric_only=True).to_dict()

train_risk_scores = final_model_1.predict(tmp[feature_list].values)
risk_ref = {
    "risk_scores": train_risk_scores.tolist(),
    "q33": float(np.quantile(train_risk_scores, 0.33)),
    "q66": float(np.quantile(train_risk_scores, 0.66)),
}

bundle = {
    "model": final_model_1,
    "features": feature_list,
    "feature_medians": feature_medians,
    "risk_ref": risk_ref,
    "scaler": scaler,          # <-- IMPORTANT
    "scaler_cols": ptpm_cols,  # <-- IMPORTANT
}

os.makedirs("results", exist_ok=True)
joblib.dump(bundle, "results/survival_model_bundle.joblib")
```

## 4) Input file rules (patient data)
- You may upload **one or multiple rows** (patients).
- Use `NA` / blank cells for missing values.
- If your file contains `stage` (e.g., `"Stage II"`), the app will derive `stage_ordinal`.
- If your bundle contains `scaler`, the app can accept raw pTPM values and apply the same preprocessing.

If your bundle **does not** include `scaler`, either:
- Upload **already-normalized** protein values (log1p + z-scored to match training), or
- Re-save the bundle including the scaler (recommended).
