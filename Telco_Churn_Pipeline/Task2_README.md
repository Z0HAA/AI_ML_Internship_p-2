# Task 2: Telco Customer Churn Prediction — End-to-End ML Pipeline

## 🎯 Objective
Build a production-ready machine learning pipeline to predict customer churn in a telecom company using Scikit-learn's `Pipeline` API.

Customer churn is when a customer stops using a service. Predicting it in advance allows businesses to take proactive retention steps — far cheaper than acquiring new customers.

**Binary classification problem:**
- `1` → Customer will churn
- `0` → Customer will stay

---

## 📂 Files
| File | Description |
|------|-------------|
| `Telco_Churn_ML_Pipeline.ipynb` | Jupyter Notebook with all code, outputs, and visualizations |
| `telco_churn.csv` | Dataset (7,043 rows × 21 columns) |
| `requirements.txt` | Python dependencies |

> **Dataset Source:** [Telco Customer Churn — IBM / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🔄 Approach

### 1. Exploratory Data Analysis
- Class distribution: **~24.4% churn rate** (imbalanced dataset)
- Visualized feature distributions, churn rates by contract type, internet service, payment method, and tenure
- Correlation heatmap for numeric features

### 2. Preprocessing (Inside Pipeline — No Data Leakage)
```
ColumnTransformer
├── Numeric columns  → StandardScaler
└── Categorical cols → OneHotEncoder
```
- Stratified 80/20 train-test split

### 3. Models Trained
| Model | Notes |
|-------|-------|
| Logistic Regression | `class_weight='balanced'` for class imbalance |
| Random Forest | Baseline + GridSearchCV tuning |

### 4. Hyperparameter Tuning
- GridSearchCV with 5-fold stratified cross-validation
- Scoring metric: ROC-AUC
- 120 model fits across 24 parameter combinations

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 65.9% | 39.9% | **78.8%** | 53.0% | **0.762** |
| Random Forest (Baseline) | 74.5% | 44.2% | 16.6% | 24.1% | 0.737 |
| Random Forest (Tuned) | **75.7%** | **51.2%** | 6.1% | 10.9% | 0.756 |

---

## 🔍 Key Observations

- **Best model depends on business goal** — Logistic Regression maximizes recall (catch all churners), Tuned Random Forest maximizes precision (targeted interventions)
- **Contract type is the #1 churn driver** — Month-to-month customers churn at ~3× the rate of two-year contract customers
- **Short tenure = high risk** — Customers under 12 months are significantly more likely to churn
- **Fiber optic & electronic check customers** show elevated churn rates

---

## 💼 Business Recommendations

| Observation | Action |
|-------------|--------|
| Month-to-month = highest churn risk | Offer discounts to convert to longer contracts |
| Short tenure customers churn most | Invest in onboarding programs (first 90 days) |
| Fiber optic customers have high churn | Proactive satisfaction surveys + SLA guarantees |
| Electronic check users are at-risk | Promote auto-pay enrollment with bill credit incentive |

---

## 🚀 How to Run
```bash
git clone https://github.com/Z0HAA/AI_ML_Internship_p-2.git
cd AI_ML_Internship_p-2/Task_2_Telco_Churn_Pipeline
pip install -r requirements.txt
jupyter notebook Telco_Churn_ML_Pipeline.ipynb
```

---

## 🛠️ Tech Stack
Python, Scikit-learn, Pandas, Matplotlib, Seaborn, Joblib
