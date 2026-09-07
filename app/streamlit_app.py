import streamlit as st
import requests

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Credit Risk Prediction")
st.write(
    "Enter the borrower's information to estimate the risk of serious credit delinquency."
)

st.divider()

st.subheader("Borrower Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=45.0
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.30
    )

    revolving_utilization = st.number_input(
        "Revolving Utilization",
        min_value=0.0,
        value=0.20
    )

    open_credit_lines = st.number_input(
        "Open Credit Lines & Loans",
        min_value=0.0,
        value=5.0
    )

with col2:
    late_30_59 = st.number_input(
        "30–59 Days Past Due",
        min_value=0.0,
        value=0.0
    )

    late_60_89 = st.number_input(
        "60–89 Days Past Due",
        min_value=0.0,
        value=0.0
    )

    late_90 = st.number_input(
        "90+ Days Late",
        min_value=0.0,
        value=0.0
    )

    real_estate_loans = st.number_input(
        "Real Estate Loans / Lines",
        min_value=0.0,
        value=1.0
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0.0,
        value=2.0
    )

st.divider()

if st.button("Predict Credit Risk", type="primary"):

    data = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
        "age": age,
        "NumberOfTime30_59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60_89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["probability_of_default"]
            prediction = result["prediction"]
            risk = result["risk"]

            st.subheader("Prediction Result")

            st.metric(
                "Probability of Default",
                f"{probability * 100:.2f}%"
            )

            if prediction == "Default":
                st.error(f"Prediction: {prediction}")
            else:
                st.success(f"Prediction: {prediction}")

            if risk == "High Risk":
                st.warning(f"Risk Level: {risk}")
            else:
                st.success(f"Risk Level: {risk}")

        else:
            st.error(
                f"API error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure FastAPI is running on port 8000."
        )