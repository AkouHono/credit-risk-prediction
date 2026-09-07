from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_credit_risk.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "API for predicting the probability of serious "
        "credit delinquency using an XGBoost model."
    ),
    version="1.0.0"
)


# ============================================================
# INPUT DATA SCHEMA
# ============================================================

class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    NumberOfTime30_59DaysPastDueNotWorse: float
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float
    NumberOfDependents: float


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running",
        "model": "XGBoost",
        "threshold": threshold
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    # Convert request data to dictionary
    data = customer.model_dump()

    # Create DataFrame using the EXACT feature names
    # and order used during model training.
    input_data = pd.DataFrame([[
        data["RevolvingUtilizationOfUnsecuredLines"],
        data["age"],
        data["NumberOfTime30_59DaysPastDueNotWorse"],
        data["DebtRatio"],
        data["MonthlyIncome"],
        data["NumberOfOpenCreditLinesAndLoans"],
        data["NumberOfTimes90DaysLate"],
        data["NumberRealEstateLoansOrLines"],
        data["NumberOfTime60_89DaysPastDueNotWorse"],
        data["NumberOfDependents"]
    ]], columns=[
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ])

    # Get probability of default
    probability = model.predict_proba(input_data)[0, 1]

    # Apply final classification threshold
    prediction = int(probability >= threshold)

    # Risk interpretation
    if prediction == 1:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return {
        "probability_of_default": round(float(probability), 4),
        "prediction": "Default" if prediction == 1 else "No Default",
        "risk": risk,
    }