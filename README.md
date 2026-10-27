# Car Insurance Claim Prediction

A machine learning project that predicts whether a policyholder will file a car insurance claim in the next policy period, based on policy, demographic, and vehicle features.

**Domain:** Insurance / Finance / Risk Analytics
**Target:** `is_claim` (binary classification)
**Stack:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly, Streamlit

---

## Business Problem

Insurers need to estimate how likely each customer is to claim. A reliable claim-probability model supports:

- **Risk assessment / fraud prevention:** flag high-risk policies early.
- **Pricing optimization:** adjust premiums to predicted claim likelihood.
- **Customer targeting:** direct marketing at low-risk customers.
- **Operational efficiency:** forecast claim volumes to plan claims-department resources.

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | 58,592 policies, 44 columns, including the target `is_claim` |
| `test.csv` | Policies without the target, used to generate predictions |
| `sample_submission.csv` | Required submission format (`policy_id`, `is_claim`) |

Key points:

- **Class imbalance:** only about 6.4% of policies have a claim (about 93.6% do not). Accuracy alone is misleading, so ROC-AUC, recall, precision, and F1 are used.
- `max_torque` (e.g. `60Nm@3500rpm`) and `max_power` (e.g. `40.36bhp@6000rpm`) are text fields with two numbers each. They are split into `torque_nm`, `torque_rpm`, `power_bhp`, and `power_rpm`.
- No missing values were found in the training data, but the pipeline still handles them.

## Project Structure

```
Car_Insurance_Claim_Prediction/
├── data/              # train.csv, test.csv, sample_submission.csv (raw data, never modified)
├── notebooks/         # car_insurance_claim_prediction.ipynb (full pipeline)
├── models/            # final_model.joblib, scaler.joblib, encoders.joblib,
│                      # feature_meta.joblib, submission.csv
├── dashboard/         # EDA plots, confusion matrices, ROC curves,
│                      # feature importance, metrics_summary.json, dashboard.html
├── app/               # app.py (Streamlit app)
├── requirements.txt
└── README.md
```

## Setup

1. Install Python 3.9+ (Anaconda works well).
2. From the project root, install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the model (notebook)

1. Put `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`.
2. Start Jupyter and open `notebooks/car_insurance_claim_prediction.ipynb`.
3. Run all cells top to bottom (Shift+Enter).

The notebook reads from `../data` and writes to `../models` and `../dashboard`.

### 2. Launch the Streamlit app

```bash
cd app
streamlit run app.py
```

Open the local URL it prints (usually `http://localhost:8501`). The app supports:

- **Batch CSV upload:** upload a file with the same columns as `test.csv`, get claim probabilities and predictions, and download the results.
- **Single record:** enter one policy's details in a form and get its claim probability.

The app loads the trained artifacts from `../models`, so the notebook must be run first.

### 3. View the dashboard

Open `dashboard/dashboard.html` in any browser for the interactive Plotly dashboard.

## Methodology

1. **EDA:** class balance, missing values, correlation heatmap, claim rates by key features.
2. **Preprocessing:** split torque/power text fields; fill missing values (median for numeric, mode for categorical); label-encode categoricals; standardize numeric features. All transformations are fit on training data only and applied identically to the test set.
3. **Train/validation split:** 80/20, stratified on `is_claim`, fixed random seed (42).
4. **Baseline models:** Logistic Regression, Decision Tree (with class weighting for the imbalance).
5. **Advanced models:** Random Forest, XGBoost (with class weighting / `scale_pos_weight`).
6. **Hyperparameter tuning:** `RandomizedSearchCV` (10 iterations, 3-fold stratified CV, scored on ROC-AUC) on the better of Random Forest and XGBoost.
7. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Log-Loss, plus confusion matrices and ROC curves for every model.
8. **Feature importance:** top 15 predictive features from the tuned model.
9. **Predictions:** the tuned model scores `test.csv` and writes `models/submission.csv`.

## Results

Validation-split results (20% stratified holdout), from `dashboard/metrics_summary.json`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log-Loss |
|-------|----------|-----------|--------|----|---------|----------|
| Logistic Regression | 0.564 | 0.080 | 0.553 | 0.140 | 0.589 | 0.673 |
| Decision Tree | 0.509 | 0.086 | 0.692 | 0.153 | 0.633 | 0.770 |
| Random Forest | 0.713 | 0.097 | 0.419 | 0.157 | 0.631 | 0.511 |
| XGBoost | 0.632 | 0.095 | 0.557 | 0.162 | 0.637 | 0.580 |
| XGBoost (tuned), 0.5 cutoff | 0.936 | 0.000 | 0.000 | 0.000 | 0.662 | 0.229 |
| XGBoost (tuned), F1-optimal cutoff | _fill in_ | _fill in_ | _fill in_ | _fill in_ | 0.662 | 0.229 |

The tuned XGBoost ranks policies best (highest ROC-AUC), but at the default 0.5 cutoff it predicts no claims at all, so its 0.936 accuracy is just the share of non-claims. A cutoff chosen to maximize F1 on the validation split is used for the final predictions (last row); because it is picked on the same split it is scored on, those numbers are slightly optimistic.

**Best model:** tuned XGBoost (highest ROC-AUC, 0.662). Best hyperparameters: _paste `search.best_params_` from the notebook_.

**Top predictive features** (from `dashboard/top_features.csv`): `age_of_car`, `policy_tenure`, `gross_weight`, `model`, `segment`, followed by `is_day_night_rear_view_mirror`, `cylinder`, `displacement`, `is_front_fog_lights`, and `power_rpm`. Importance scores are fairly flat (about 0.077 for the top feature down to 0.026 for the 15th), so no single feature dominates.

**Why these metrics:** with only about 6% positive cases, a model that never predicts a claim would still reach about 94% accuracy. ROC-AUC measures ranking quality independent of threshold, while recall shows how many real claims are caught and precision shows how many flagged policies actually claim.

## Business Insights

> Add claim-rate findings from `dashboard/dashboard.html` (e.g. which segments or tenure ranges claim more) alongside the points below.

- Vehicle age and policy tenure are the two most influential features, so both are worth considering in pricing and renewal decisions.
- Vehicle size and engine features (`gross_weight`, `cylinder`, `displacement`, `segment`, `model`) fill several of the top 10 slots, suggesting vehicle class matters for risk.
- With a ROC-AUC of about 0.66 and flat feature importances, no single factor separates claimers from non-claimers. The model is better suited to ranking relative risk than to confident individual predictions. Feature importance shows what the model uses, not the direction of the effect or its cause; equipment flags such as `is_day_night_rear_view_mirror` are likely proxies for car trim level.

## Limitations

- Heavy class imbalance: at the default 0.5 probability threshold the tuned model flagged no claims at all, so the final predictions use a cutoff chosen on the validation split. The threshold can be lowered to trade precision for recall depending on business cost.
- Label encoding is used for categorical features, which can imply an order that does not exist for some columns.
- LightGBM and a more extensive hyperparameter search were not included.
- The app runs locally; it is not yet deployed to a public URL.
- Model performance reflects one train/validation split plus 3-fold CV during tuning.

## Future Improvements

- Tune the decision threshold against a business cost for missed claims versus false alarms.
- Try SMOTE or other resampling, probability calibration, and LightGBM/CatBoost.
- Use target or one-hot encoding for high-cardinality categoricals such as `area_cluster` and `model`.
- Wrap the model in a FastAPI service and containerize it with Docker for deployment (Streamlit Community Cloud, Heroku, or AWS).

## Reproducibility

- Random seed fixed to 42 in the split and all models.
- Dependencies listed in `requirements.txt`.
- Raw data in `data/` is never overwritten; all transformations happen inside the notebook pipeline.
