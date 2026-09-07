# 💳 Credit Risk Prediction

Predicting the probability that a borrower will experience serious delinquency within two years, using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient--boosting-brightgreen)
![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-deployed-success)

---

## 📑 Table of Contents

- [💳 Credit Risk Prediction](#-credit-risk-prediction)
  - [📑 Table of Contents](#-table-of-contents)
  - [📌 Overview](#-overview)
  - [🏦 Business Problem](#-business-problem)
  - [📊 Dataset](#-dataset)
  - [🗂 Project Structure](#-project-structure)
  - [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
  - [🧹 Data Preprocessing](#-data-preprocessing)
  - [🤖 Modeling Approach](#-modeling-approach)
  - [📈 Results](#-results)
  - [🌟 Feature Importance](#-feature-importance)
  - [🎯 Threshold Optimization](#-threshold-optimization)
  - [💡 Key Takeaways](#-key-takeaways)
  - [🚀 Deployment](#-deployment)
  - [⚙️ Installation](#️-installation)
  - [▶️ Usage](#️-usage)
    - [Running the API](#running-the-api)
    - [Running the Streamlit app](#running-the-streamlit-app)
  - [🛠 Tech Stack](#-tech-stack)
  - [🚧 Roadmap](#-roadmap)
  - [📄 License](#-license)
  - [📬 Contact](#-contact)

---

## 📌 Overview

This project builds and evaluates machine learning models that predict whether a borrower will default (experience 90+ days delinquency) within the next two years, based on their credit profile. It follows an end-to-end data science workflow:

1. Exploratory Data Analysis (EDA) with business-driven interpretation
2. Data cleaning and preprocessing
3. Handling severe class imbalance (SMOTE, class weighting)
4. Training and comparing multiple models (Logistic Regression, Random Forest, XGBoost)
5. Hyperparameter tuning
6. Threshold optimization for a business-relevant precision/recall trade-off
7. Model persistence for downstream use
8. Deployment behind a REST API with an interactive web interface

The final model is a tuned **XGBoost classifier**, selected because credit risk datasets are highly imbalanced and tree-based ensembles handle correlated, non-linear features better than linear models. It's served through a **FastAPI** backend and a **Streamlit** front end, so a user can enter borrower details and get an instant default-risk prediction.

## 🏦 Business Problem

Banks need to estimate the probability that a loan applicant will default so they can make informed lending decisions. Two types of errors carry very different costs:

- **False Negative** (predicting "safe" when the customer actually defaults) → direct financial loss for the bank.
- **False Positive** (predicting "risky" when the customer is actually safe) → lost business/interest revenue, but far cheaper than a default.

Because of this asymmetry, **recall on the default class** (catching as many true defaulters as possible) is prioritized over raw accuracy, and the classification threshold is tuned accordingly rather than left at the default 0.5.

## 📊 Dataset

- **Source:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle competition dataset)
- **Rows:** 150,000 customers
- **Target:** `SeriousDlqin2yrs` — `1` = customer defaulted (experienced serious delinquency), `0` = customer did not default
- **Class balance:** ~93.3% non-default vs. ~6.7% default (highly imbalanced)

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards/lines relative to credit limits |
| `age` | Age of the borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times borrower was 30–59 days late (not worse) |
| `DebtRatio` | Monthly debt payments / monthly income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans/credit lines |
| `NumberOfTimes90DaysLate` | Times borrower was 90+ days late |
| `NumberRealEstateLoansOrLines` | Number of mortgage/real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times borrower was 60–89 days late (not worse) |
| `NumberOfDependents` | Number of dependents |

## 🗂 Project Structure

```
Credit-Risk-Prediction/
├── app/
│   ├── main.py                    # FastAPI REST API — serves predictions
│   └── streamlit_app.py           # Streamlit UI — borrower form → risk prediction
├── data/
│   └── raw/
│       └── cs-training.csv        # raw dataset (not committed — see Installation)
├── images/                        # plots and screenshots used in this README
├── models/
│   ├── xgboost_credit_risk.pkl    # final trained model
│   └── threshold.pkl              # tuned decision threshold (0.70)
├── notebooks/
│   └── credit_prediction.ipynb    # full EDA + modeling notebook
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** The raw dataset is not included in this repo due to Kaggle's terms of use. Download it directly from the [competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it under `data/raw/`.

## 🔍 Exploratory Data Analysis

The notebook walks through a structured, business-driven EDA covering univariate distributions, outlier reasoning, and correlation analysis. Highlights:

- **Target variable:** ~6.7% of customers defaulted — confirming the need for imbalance-aware modeling.
- **Age:** Roughly normal distribution (mean ≈ 52, slight right skew of 0.19), with a few implausible values (min = 0) flagged for review.
- **Monthly Income:** ~20% missing, extremely right-skewed (skew ≈ 114), with legitimate high-income outliers (executives, business owners) that were investigated rather than blindly dropped — removing them could erase a real, important customer segment.
- **Debt Ratio:** Also extremely right-skewed (skew ≈ 95); a high debt ratio signals reduced repayment capacity and higher credit risk.
- **Correlation analysis:**
  - Past delinquency features (30–59, 60–89, 90+ days late) are the strongest positive predictors of default and are highly correlated with each other (0.98–0.99), indicating multicollinearity — one reason tree-based models were favored over plain linear regression.
  - `age` has a weak negative correlation with default (older customers default slightly less often).
  - `MonthlyIncome` and `DebtRatio` show weak *linear* correlation with the target, suggesting their relationship with risk is non-linear (a good fit for tree-based models).

## 🧹 Data Preprocessing

1. Dropped the redundant index column (`Unnamed: 0`).
2. Imputed missing values with the **median** (robust to skew/outliers):
   - `MonthlyIncome` (29,731 missing)
   - `NumberOfDependents` (3,924 missing)
3. Split features (`X`) and target (`y`).
4. Train/test split: 80/20, **stratified** on the target to preserve the class ratio in both sets.
5. Addressed class imbalance via **SMOTE** oversampling (training set only) and, separately, via XGBoost's built-in `scale_pos_weight`.

## 🤖 Modeling Approach

Four modeling strategies were trained and compared on the same held-out test set (30,000 customers):

| Model | Imbalance handling |
|---|---|
| Logistic Regression | None (baseline) |
| Logistic Regression + SMOTE | Oversampling |
| Random Forest (200 trees) | None, then + SMOTE |
| XGBoost (tuned) | `scale_pos_weight` + threshold optimization |

XGBoost hyperparameters were tuned with `RandomizedSearchCV` (15 candidates, 3-fold CV, optimizing F1) over `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, and `colsample_bytree`.

## 📈 Results

Metrics on the untouched test set, default 0.5 threshold (except the final XGBoost row, evaluated at its optimized threshold):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.934 | 0.588 | 0.050 | 0.092 |
| Logistic Regression + SMOTE | 0.694 | 0.128 | 0.615 | 0.212 |
| Random Forest | 0.936 | 0.563 | 0.186 | 0.280 |
| Random Forest + SMOTE | 0.894 | 0.300 | 0.440 | 0.357 |
| XGBoost (0.5 threshold) | 0.805 | 0.223 | 0.769 | 0.346 |
| **XGBoost (tuned, threshold = 0.70)** | **0.898** | **0.344** | **0.582** | **0.432** |

Additional threshold-independent metrics for the final XGBoost model:

- **ROC-AUC:** 0.867 — strong ability to rank defaulters above non-defaulters, well above the 0.5 random baseline.
- **PR-AUC:** 0.404 — far above the ~0.067 baseline expected from a random classifier on this imbalanced dataset, and the more informative metric here since it focuses on the minority (default) class.

The plain-accuracy baseline (Logistic Regression, 93.4% accuracy) is misleading: it only caught **5%** of actual defaulters. The final tuned XGBoost model trades some accuracy for a **12x improvement in recall**, catching more than half of all defaulters while keeping precision at a workable level.

## 🌟 Feature Importance

Top predictors identified by the Random Forest model:

| Feature | Importance |
|---|---|
| RevolvingUtilizationOfUnsecuredLines | 0.191 |
| DebtRatio | 0.179 |
| MonthlyIncome | 0.146 |
| age | 0.128 |
| NumberOfTimes90DaysLate | 0.094 |
| NumberOfOpenCreditLinesAndLoans | 0.089 |
| NumberOfTime60-89DaysPastDueNotWorse | 0.050 |
| NumberOfTime30-59DaysPastDueNotWorse | 0.048 |
| NumberOfDependents | 0.042 |
| NumberRealEstateLoansOrLines | 0.033 |

## 🎯 Threshold Optimization

Because the default 0.5 cutoff is arbitrary, prediction probabilities were swept across thresholds (0.10 → 0.90) to find the point maximizing F1-score on the default class. The best F1 (0.449) occurred near **threshold ≈ 0.77**; a threshold of **0.70** was selected as the final operating point to balance a strong recall (catching risky customers) against keeping false positives manageable.

| Threshold | Precision | Recall | F1-score |
|---|---|---|---|
| 0.20 | 0.121 | 0.937 | 0.214 |
| 0.30 | 0.151 | 0.890 | 0.259 |
| 0.40 | 0.183 | 0.837 | 0.300 |
| 0.50 | 0.223 | 0.769 | 0.346 |
| 0.60 | 0.278 | 0.676 | 0.393 |
| 0.70 | 0.344 | 0.582 | 0.432 |

## 💡 Key Takeaways

- **Accuracy is a poor metric for imbalanced problems.** A 93%+ accuracy model can still miss nearly all real defaulters.
- **Recall matters more than accuracy in this business context** — missing a defaulter is more costly than flagging a safe customer for review.
- **Outliers require business judgment, not automatic removal.** High-income and high-debt-ratio outliers can represent legitimate, important customer segments.
- **Multicollinearity among delinquency features** favors tree-based ensembles (Random Forest, XGBoost) over plain linear models.
- **Threshold tuning is essential** — the same model can be pushed toward higher recall or higher precision by moving the decision threshold, without retraining.

## 🚀 Deployment

The trained XGBoost model is saved with **Joblib** and served through a **FastAPI** REST API. A **Streamlit** app sits on top of the API and provides a simple form where a user can enter borrower information and get back a default-risk prediction.

**Application architecture:**

```text
User
  ↓
Streamlit (app/streamlit_app.py)
  ↓
FastAPI REST API (app/main.py)
  ↓
XGBoost Model (models/xgboost_credit_risk.pkl)
  ↓
Probability of Default
  ↓
Default / No Default  (threshold = 0.70)
  ↓
High Risk / Low Risk
```

The API applies the same 0.70 decision threshold used during evaluation, so the risk label returned to the user matches the model's evaluated precision/recall trade-off rather than the naive 0.5 cutoff.

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Credit-Risk-Prediction.git
cd Credit-Risk-Prediction

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Minimum required packages:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
joblib
fastapi
uvicorn
streamlit
requests
```

## ▶️ Usage

1. Download `cs-training.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in `data/raw/`.
2. Open `notebooks/credit_prediction.ipynb` in Jupyter or Google Colab.
3. Run all cells to reproduce the EDA, preprocessing, training, and evaluation.
4. The final trained model and threshold are saved to `models/xgboost_credit_risk.pkl` and `models/threshold.pkl`, and can be loaded for inference:

```python
import joblib

model = joblib.load("models/xgboost_credit_risk.pkl")
threshold = joblib.load("models/threshold.pkl")

proba = model.predict_proba(X_new)[:, 1]
prediction = (proba >= threshold).astype(int)
```

### Running the API

```bash
cd app
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### Running the Streamlit app

```bash
cd app
streamlit run streamlit_app.py
```

This opens a web form where you can enter a borrower's details (age, income, debt ratio, delinquency history, etc.) and get back a **Default / No Default** prediction along with the underlying probability.

> Make sure the FastAPI server is running before starting the Streamlit app, since the UI calls the API for predictions.

## 🛠 Tech Stack

- **Language:** Python 3.13
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Model persistence:** joblib
- **API:** FastAPI, Uvicorn
- **Web interface:** Streamlit
- **Environment:** Google Colab / Jupyter Notebook

## 🚧 Roadmap

- [ ] Add SHAP-based explainability for individual predictions (surfaced in the Streamlit UI)
- [ ] Package preprocessing + inference into a reusable pipeline (`sklearn.Pipeline`)
- [ ] Containerize the API and app with Docker for easier deployment
- [ ] Add automated tests and CI
- [ ] Experiment with LightGBM / CatBoost and model stacking
- [ ] Deploy to a public cloud host (e.g. Render, Fly.io, or Streamlit Community Cloud)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📬 Contact

Questions, feedback, or collaboration ideas are welcome — feel free to open an issue or reach out.

Future Enhancements

Batch prediction through CSV file uploads
Automatic prediction for multiple borrowers
Integration with a database or external data source
Real-time credit-risk prediction
Model monitoring and performance tracking
Cloud deployment
Improved model explainability

Limitations

This project is intended as a machine learning portfolio and decision-support project rather than a production banking system.

The model should not be used as the sole basis for real-world lending decisions. Production deployment would require additional validation, monitoring, security, fairness assessment, regulatory compliance, and integration with reliable financial data sources.

Technologies
Python
Pandas
NumPy
Scikit-learn
XGBoost
FastAPI
Streamlit
Joblib