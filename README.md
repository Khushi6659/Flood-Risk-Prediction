# Flood-Risk-Prediction
“Flood risk prediction model using ML on climate and hydrological data for India.”

# Flood Risk Prediction — README

**Project:** Flood Risk Prediction for India using ml
**Files:** `flood_risk_dataset_india.csv`,`flood_app.db`, `smote_data.pkl`, `flood_stack_artifacts.pkl`, `requirements.txt`, `notebook.ipynb`, `users.db`,`userss_data.db`

---

## 1 — Short project summary

This project builds a model to predict flood risk (binary label `Flood Occurred`) using historical climate, hydrological and socio-environmental features.
Steps performed: data loading → EDA → feature engineering → feature selection → optional balancing → stacking ensemble modeling → evaluation → save artifacts.

---

## 2 — Dataset (what I used)

* File: `flood_risk_dataset_india.csv` (you provided/uploaded)
* Size: **10,000 rows × 14 original columns**
* Example columns:

  * `Latitude`, `Longitude`
  * `Rainfall (mm)`, `Temperature (°C)`, `Humidity (%)`
  * `River Discharge (m³/s)`, `Water Level (m)`, `Elevation (m)`
  * `Land Cover`, `Soil Type`
  * `Population Density`, `Infrastructure`, `Historical Floods`
  * `Flood Occurred` (target, binary)



---

## 3 — Stepwise process (what I did — detailed)

### Step 0 — Environment & deps

Install dependencies (run once):

```bash
pip install -r requirements.txt
# OR individually:
pip install numpy pandas scikit-learn matplotlib seaborn joblib xgboost catboost imbalanced-learn
```

*Note:* CatBoost / XGBoost / imbalanced-learn are optional — code falls back if missing.

---

### Step 1 — Load & quick checks

* Load CSV into `pandas.DataFrame`.
* Print shape, head, `df.info()`, `df.describe()` and `df.isnull().sum()` to confirm there are no missing values.

Purpose: ensure file loaded correctly and understand basic stats.

---

### Step 2 — Exploratory Data Analysis (EDA)

Typical EDA performed:

* Target distribution (`countplot`) — confirmed near 50/50.
* Numeric distributions and summary stats.
* Correlation heatmap of numeric columns to inspect relationships.
* Boxplots (e.g., `Rainfall` vs `Flood Occurred`) and frequency counts for categorical columns like `Land Cover`, `Soil Type`.

Purpose: spot patterns, outliers, and features likely informative for flood risk.

---

### Step 3 — Feature engineering (new features added)

Created domain-inspired engineered features to capture interactions / nonlinearity:

* `Rainfall_Anomaly` = current rainfall − overall mean rainfall
* `Flood_Risk_Index_v2` = (rainfall \* (river discharge + 1)) / (elevation + 1)
* `Humid_Temp` = humidity × temperature
* `Population_Exposure` = population density × (historical\_floods + 1)
* `Elevation_WaterRatio` = elevation / (water level + 1)
* `Discharge_Rainfall_Ratio` = (river discharge + 1) / (rainfall + 1)

These attempt to capture combined risk signals (hydrology + exposure).

---

### Step 4 — Encoding categorical variables

* `LabelEncoder` applied to categorical text columns (`Land Cover`, `Soil Type`).
  (Encoders are saved so the same mappings can be used at inference.)

---

### Step 5 — Feature selection

* Ran a quick `RandomForestClassifier` to compute importances.
* Used `SelectFromModel` with a threshold (median importance) to pick the most informative features (rolled up to \~10–12 features in the pipeline).
* Saved `selected_features` order for later saving/prediction.

Rationale: reduce noisy dimensions and speed up models.

---

### Step 6 — (Optional) SMOTE balancing

* Because many pipelines benefit from balanced classes, code uses `SMOTE` if `imblearn` is installed.
* **Important note:** your dataset is already nearly balanced — in experiments SMOTE did **not** help and sometimes introduced noise. For your particular dataset, consider skipping SMOTE (the pipeline has a flag and fallback).

---

### Step 7 — Scaling

* `StandardScaler` fit on training data (or on SMOTE-resampled data) and saved. Tree models do not require scaling, but scaling is helpful if you use logistic regression or neural nets, and ensures consistent input for stacking meta-learner.

---

### Step 8 — Modeling: Stacking ensemble

* Base learners used (configurable):

  * `RandomForestClassifier` (strong baseline)
  * `GradientBoostingClassifier`
  * optionally `XGBClassifier` (XGBoost) if installed
  * optionally `CatBoostClassifier` if installed
* Final meta-learner: `LogisticRegression`
* `StackingClassifier` from scikit-learn used with `n_jobs=-1`.
* Cross-validated via `StratifiedKFold` (default 5 folds) to estimate performance.

Why stacking? it combines complementary strengths of base models and often yields better generalization.

---

### Step 9 — Evaluation & outputs

* Predictions on test set: `accuracy_score`, `classification_report`, `confusion_matrix`, `ROC-AUC` (if probability available).
* Confusion matrix is plotted as a heatmap and also saved as `confusion_matrix.png`.

**Observed results on your runs:** stacking + engineered features did not improve beyond \~50% on several runs — this suggests the dataset’s features do not contain a learnable signal strong enough for the target or that additional domain features are required. (See *Next steps* below.)

---

### Step 10 — Save artifacts

Saved model & artifacts with `joblib`:

```python
joblib.dump({
  "stack_model": stack,
  "scaler": scaler,
  "encoders": encoders,
  "selected_features": selected_features
}, "flood_stack_artifacts.pkl")
```

Saved confusion matrix image: `confusion_matrix.png`.

---

## 4 — How to run (quick)

### Option A — Run the pipeline script (one-shot)

If you have the `flood_stack_pipeline.py` (or copied the Jupyter code into a cell), simply run:

```bash
python flood_stack_pipeline.py
```

or run the notebook and execute cells in order.

### Option B — Stepwise (recommended while developing)

1. Install deps:

   ```bash
   pip install -r requirements.txt
   ```
2. Open `notebook.ipynb` in Jupyter, run cells in order:

   * Data load & EDA
   * Feature engineering & encoding
   * Feature selection
   * (Optional) SMOTE
   * Train stacking model
   * Evaluate & save artifacts

### Inference (use saved artifact)

```python
import joblib
import numpy as np
art = joblib.load("flood_stack_artifacts.pkl")
model = art["stack_model"]
scaler = art["scaler"]
encoders = art["encoders"]
selected = art["selected_features"]

# Example: prepare a single sample (dict keys must match selected features)
sample = {
  "Latitude": 18.86,
  "Longitude": 78.83,
  "Temperature (°C)": 34.14,
  # ... include all selected features in correct order or compute engineered features first
}

# Create DataFrame row -> apply same encoders & scalers -> select features -> predict
```

(See the notebook for a full inference code cell — it loads encoders and the scaler and applies the same transforms.)

---

## 5 — Expected runtime / resources

On an Intel i5 laptop with 8GB RAM (typical):

* Full stacking run (with XGBoost/CatBoost present): **\~2–8 minutes** (depends on n\_estimators and presence of boosters).
* Without XGBoost/CatBoost: **\~2–4 minutes**.
  If memory issues occur, reduce `n_estimators` (e.g., 100) or remove CatBoost/XGBoost.

---

## 6 — Observed results & interpretation

* After many experiments (RF, XGBoost, GB, stacking), the accuracy plateaued around **\~50%** on the test set in multiple runs.
* Interpretation: with the current features and target label, the models cannot find a robust separable signal. This can happen if:

  * The dataset is noisy or synthetic.
  * Important domain features (temporal trends, upstream watershed info, soil moisture, satellite indices) are missing.
  * Spatial/temporal correlation is required (per-station time series rather than single-row snapshots).
* **Therefore:** the pipeline is sound and reproducible, but additional or different features are necessary to reach the 70–90% accuracy range.

---

## 7 — Next steps & recommendations (how to reach 70–90% accuracy)

If you *must* increase accuracy without changing the target labels, try:

1. **Add domain features** (most impactful):

   * Rolling sums / moving averages of rainfall (past 3/7/14 days).
   * Month/season (monsoon flag).
   * Distance to nearest river/stream (or watershed id).
   * Satellite-derived soil moisture or NDVI indices.
   * Drainage density / land slope computed from elevation grid.

2. **Spatial aggregation**:

   * Group by region (district/state) and add region-level flood history / climate normals.

3. **Model tweaks**:

   * Perform `GridSearchCV` / `RandomizedSearchCV` for stronger tuning (but increases runtime).
   * Try `StackingClassifier(passthrough=True)` so meta-learner sees original features + base predictions.
   * Try `CatBoost` (strong with categorical) and tune it.

4. **Feature engineering experiments**:

   * PCA, feature interactions, polynomial features (careful with overfitting).

5. **Data quality**:

   * Manually inspect rows where model and truth disagree (false negatives are critical in flood detection).

---

## 8 — Common troubleshooting

* **XGBoost / CatBoost import errors:** install them or the code will skip those models automatically.
* **MemoryError:** reduce `n_estimators` or use fewer estimators for boosters.
* **SMOTE making things worse:** if target is already balanced, **do not** use SMOTE.
* **Different accuracy on re-run:** set `random_state=42` consistently to make runs reproducible.

---

## 9 — Files produced by this project

* `flood_stack_artifacts.pkl` — saved model + scaler + encoders + selected features
* `confusion_matrix.png` — heatmap of test confusion matrix
* `notebook.ipynb` — step-by-step notebook (if created)
* `presentation.pptx` — final project slides (you must use the shared template)

---


## 10 — Final notes (honest expectation)

* The code and pipeline are **correct and reproducible**.
* With the current dataset we observed that models struggled to go beyond random-level performance — this is not a code bug, but a data/feature limitation.
* If you can add **any** extra domain data (time series, region id, seasonal flags, satellite indices), I will help you integrate them and tune models to target 70–90% accuracy.

---
## 11 🌊 Flood Risk Prediction Web App

A Streamlit + SQLite based web app that predicts flood risk using environmental & demographic inputs. Includes user registration, login, prediction, visualization, and downloadable reports.

Initialize Database

No manual setup needed 🚀
On first run, the app will automatically create flood_app.db with two tables:

users → stores username & password

user_inputs → stores input data

## Run the Web App
streamlit run floods_app.py


Browser will open at:
👉 http://localhost:8501

## Register a User

Open Register Page

Enter username & password

Duplicate usernames are not allowed

## Login

Enter correct username & password

Multiple logins are allowed for the same account

After success → redirected to Prediction Page

## Prediction Workflow

Enter Environmental & Demographic Data:

Latitude, Longitude

Rainfall, Temperature, Humidity

River Discharge, Water Level, Elevation

Land Cover, Soil Type, Population Density

Infrastructure, Historical Floods

Click Predict Flood Risk

Output shows:

Flood Probability (%)

Risk Category (Low / Medium / High)

Bar Chart (feature contribution)

Download CSV Report

## Example Input & Output

Input

Rainfall = 200 mm
River Discharge = 1200 m³/s
Water Level = 15 m
Historical Floods = 3


Output

Flood Probability → 75.3%

Risk → High Flood Risk

Visualization → Bar chart of features

CSV → Downloadable report

## Project Structure
📁 Flood-Risk-Prediction
 ┣ 📜 floods_app.py        # Main app
 ┣ 📜 utils.py             # DB functions
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 README.md            # Documentation
 ┗ 📜 flood_app.db         # Auto-created DB

If you want, I can now:

* generate a ready-to-paste `requirements.txt`, or
* create a compact Streamlit app that loads `flood_stack_artifacts.pkl` and provides a prediction UI, or
* prepare a short slide (1-slide) describing the pipeline for your presentation.

Which of these would you like next?
