import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('final_logistic_model.pkl')

# Streamlit UI setup
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("📊 Credit Risk Predictor")
st.markdown("Enter applicant details below to assess credit risk.")

# Input form
with st.form("input_form"):
    annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, step=0.01)
    inquiries_last_12m = st.number_input("Inquiries in Last 12 Months", min_value=0, step=1)
    total_credit_utilized = st.number_input("Total Credit Utilized", min_value=0.0, step=100.0)
    total_credit_limit = st.number_input("Total Credit Limit", min_value=0.0, step=100.0)
    account_never_delinq_percent = st.number_input("Account Never Delinquent (%)", min_value=0.0, max_value=100.0, step=0.1)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
    term = st.number_input("Loan Term (months)", min_value=0, step=12)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_data = pd.DataFrame([[
        annual_income,
        debt_to_income,
        inquiries_last_12m,
        total_credit_utilized,
        total_credit_limit,
        account_never_delinq_percent,
        loan_amount,
        interest_rate,
        term
    ]], columns=[
        'annual_income',
        'debt_to_income',
        'inquiries_last_12m',
        'total_credit_utilized',
        'total_credit_limit',
        'account_never_delinq_percent',
        'loan_amount',
        'interest_rate',
        'term'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.success(f"🔮 Prediction: {'High Risk (1)' if prediction == 1 else 'Low Risk (0)'}")
    st.info(f"📈 Probability of High Risk: {probability:.2%}")
