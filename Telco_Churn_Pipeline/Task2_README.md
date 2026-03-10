# Task 2: End-to-End ML Pipeline — Telco Customer Churn Prediction

## 📌 Objective

Build a **reusable, production-ready machine learning pipeline** to predict customer churn in a telecom company using Scikit-learn's `Pipeline` API.

Customer churn is when a customer stops using a service. Predicting churn in advance allows a business to take proactive retention steps — which is far cheaper than acquiring new customers.

**This is a binary classification problem:**
- `1` → Customer will churn
- `0` → Customer will stay

---

## 📂 Files in This Folder

| File | Description |
|------|-------------|
| `Telco_Churn_ML_Pipeline.ipynb` | Fully executed Jupyter Notebook with all code, outputs, and visualizations |
| `telco_churn.csv` | Telco Customer Churn dataset (7,043 rows × 21 columns) |
| `requirements.txt` | Python package dependencies |
| `README.md` | This documentation file |

> **Dataset Source:** [Telco Customer Churn — IBM / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> The dataset used here mirrors the structure, column names, distributions, and class balance (~24% churn rate) of the original.

---

## ✅ Project Checklist

### 1. Jupyter Notebook
- [x] **Problem Statement & Objective** — Section 1: Business context, goal, ML approach
- [x] **Dataset Loading & Preprocessing** — Sections 3–5: Load CSV, fix dtypes, encode target, split data
- [x] **Model Development & Training** — Sections 6–7: Pipeline construction, training LR and RF
- [x] **Evaluation with Relevant Metrics** — Sections 7–9: Accuracy, Precision, Recall, F1, ROC-AUC, CV
- [x] **Visualizations** — Sections 4, 8, 9, 10, 11: 10 charts including ROC curves, confusion matrices, feature importance
- [x] **Final Summary / Insights** — Section 13: Results table, key findings, business recommendations

### 2. Code Quality
- [x] Clear structure with 13 named sections
- [x] Logical flow: EDA → Preprocess → Pipeline → Train → Evaluate → Tune → Export
- [x] Comments explaining every major step
- [x] Markdown cells between code sections for readability
- [x] Descriptive variable names
- [x] No data leakage (all transforms inside Pipeline)

---

## 🔄 Methodology / Approach

### Step 1 — Exploratory Data Analysis
- Analyzed class distribution: **~24.4% churn rate** (imbalanced dataset)
- Visualized numeric feature distributions by churn class
- Plotted churn rates by key categorical features (Contract, InternetService, PaymentMethod, Tenure)
- Generated correlation heatmap for numeric features

### Step 2 — Preprocessing (Inside Pipeline — No Leakage)
```
ColumnTransformer
├── Numeric columns  → StandardScaler         (tenure, MonthlyCharges, TotalCharges, SeniorCitizen)
└── Categorical cols → OneHotEncoder          (Contract, InternetService, PaymentMethod, etc.)
```
- Stratified 80/20 train-test split preserves churn ratio in both sets

### Step 3 — Models Trained
| Model | Notes |
|-------|-------|
| Logistic Regression | `class_weight='balanced'` to handle class imbalance |
| Random Forest | Baseline + GridSearchCV tuning |

### Step 4 — Hyperparameter Tuning (GridSearchCV)
- **Scoring:** ROC-AUC (better than accuracy for imbalanced data)
- **CV:** 5-fold stratified cross-validation
- **Grid:** 24 parameter combinations × 5 folds = 120 model fits

```python
param_grid = {
    'classifier__n_estimators':    [100, 200],
    'classifier__max_depth':       [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight':    [None, 'balanced']
}
```

### Step 5 — Export with Joblib
The complete trained pipeline (preprocessor + model) is exported as a single `.joblib` file for production use.

---

## 📊 Key Results & Observations

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression (`balanced`) | 65.9% | 39.9% | **78.8%** | 53.0% | **0.762** |
| Random Forest (Baseline) | 74.5% | 44.2% | 16.6% | 24.1% | 0.737 |
| Random Forest (GridSearch Tuned) | **75.7%** | **51.2%** | 6.1% | 10.9% | 0.756 |

### Top 10 Feature Importances (Tuned Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Contract — Month-to-month | 0.1482 |
| 2 | TotalCharges | 0.1269 |
| 3 | tenure | 0.1087 |
| 4 | MonthlyCharges | 0.1087 |
| 5 | Contract — Two year | 0.0886 |
| 6 | Contract — One year | 0.0405 |
| 7 | PaymentMethod — Electronic check | 0.0166 |
| 8 | SeniorCitizen | 0.0161 |
| 9 | InternetService — Fiber optic | 0.0148 |
| 10 | Partner — No | 0.0148 |

---

## 🔍 Key Observations

### 1. Class Imbalance
Only ~24.4% of customers churned. A naive "always predict No" model gets 75.6% accuracy but 0% recall. This is why we use `class_weight='balanced'` and prioritize **ROC-AUC** and **Recall** over accuracy.

### 2. Best Model Depends on Business Goal
- **Maximize recall (catch all churners)** → Use Logistic Regression (Recall: 78.8%, ROC-AUC: 0.762)
- **Maximize precision (targeted interventions)** → Use Tuned Random Forest (Precision: 51.2%)

### 3. Contract Type is the #1 Churn Driver
Month-to-month customers churn at ~3× the rate of two-year contract customers. Converting customers to longer contracts is the single highest-impact retention strategy.

### 4. Short Tenure = High Risk
Customers who have been with the company for less than 12 months are significantly more likely to churn. Early engagement programs in the first 3–6 months are critical.

### 5. Fiber Optic & Electronic Check Customers Are At Risk
These customer segments show elevated churn, possibly due to higher expectations (Fiber) or lower brand engagement (Electronic check = less friction to leave).

---

## 💼 Business Recommendations

| Observation | Action |
|-------------|--------|
| Month-to-month = highest churn risk | Offer loyalty discounts to convert to 1–2 year contracts |
| Short tenure customers churn most | Invest in onboarding programs (first 90 days) |
| Fiber optic customers have high churn | Proactive satisfaction surveys + SLA guarantees |
| Electronic check users are at-risk | Promote auto-pay enrollment with a bill credit incentive |
| Senior citizens have elevated risk | Dedicated support line + simplified plans |

**Expected ROI:** Targeting the top 20% highest-risk customers with personalized retention offers could reduce churn by 30–40%, saving thousands in customer acquisition costs monthly.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AI_ML_Internship_p-2.git
cd AI_ML_Internship_p-2/Task_2_Telco_Churn_Pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook Telco_Churn_ML_Pipeline.ipynb
```

### 4. Run all cells
`Kernel → Restart & Run All`

---

## 🔁 Reusing the Exported Pipeline

```python
import joblib
import pandas as pd

# Load the saved pipeline (handles all preprocessing automatically)
model = joblib.load('best_model_pipeline.joblib')

# Pass raw customer data — no manual preprocessing needed
new_data = pd.DataFrame({...})  # same columns as training data
churn_probability = model.predict_proba(new_data)[:, 1]
churn_prediction  = model.predict(new_data)

print("Churn probabilities:", churn_probability)
```

---

## 🛠️ Skills Demonstrated

- ✅ ML pipeline construction with `sklearn.pipeline.Pipeline`
- ✅ Column-wise preprocessing with `ColumnTransformer`
- ✅ Handling class imbalance (`class_weight='balanced'`, stratified splits)
- ✅ Hyperparameter tuning with `GridSearchCV`
- ✅ Model evaluation with Accuracy, Precision, Recall, F1, ROC-AUC, CV
- ✅ Data visualization (10 plots: distributions, confusion matrices, ROC curves, feature importance)
- ✅ Model export and reusability with `joblib`
- ✅ Production-readiness practices (no data leakage, portable pipeline)

---

*Task 2 | AI/ML Internship Project 2 | Telco Customer Churn Prediction*
