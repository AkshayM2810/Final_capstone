import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('final_xgb_weighted_model.pkl')

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Predictor")

st.markdown("Please enter the applicant's details:")

# User input form
annual_income = st.number_input("Annual Income", value=60000.0)
debt_to_income = st.number_input("Debt-to-Income Ratio", value=0.3, min_value=0.0, max_value=1.0)
inquiries_last_12m = st.number_input("Inquiries in Last 12 Months", value=1)
total_credit_utilized = st.number_input("Total Credit Utilized", value=4000.0)
total_credit_limit = st.number_input("Total Credit Limit", value=15000.0)
account_never_delinq_percent = st.number_input("Account Never Delinquent (%)", value=90.0, min_value=0.0, max_value=100.0)
loan_amount = st.number_input("Loan Amount", value=8000.0)
interest_rate = st.number_input("Interest Rate (%)", value=13.0)
 
if st.button("Predict"):
    # Construct input DataFrame
    input_data = pd.DataFrame([{
        'annual_income': annual_income,
        'debt_to_income': debt_to_income,
        'inquiries_last_12m': inquiries_last_12m,
        'total_credit_utilized': total_credit_utilized,
        'total_credit_limit': total_credit_limit,
        'account_never_delinq_percent': account_never_delinq_percent,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate
    }])

    # Business Rule-Based Decision
    rule_based = (
        (debt_to_income < 0.4) and
        (annual_income > 50000) and
        (interest_rate < 20) and
        (account_never_delinq_percent > 80)
    )
    st.markdown(f"📋 **Rule-Based Decision:** {'✅ Approved' if rule_based else '❌ Rejected'}")

    # Model Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown(f"🧠 **Model Prediction:** {'✅ Approved' if prediction == 1 else '❌ Rejected'}")
    st.markdown(f"🔍 **Probability of Approval:** {probability:.2%}")
