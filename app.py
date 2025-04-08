import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('final_xgb_weighted_model.pkl')

# Streamlit page setup
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")

st.markdown("Fill in the applicant details to check approval prediction:")

# Input form
annual_income = st.number_input("Annual Income", min_value=0.0, value=65000.0)
debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.25)
inquiries_last_12m = st.number_input("Inquiries in Last 12 Months", min_value=0, value=1)
total_credit_utilized = st.number_input("Total Credit Utilized", min_value=0.0, value=4000.0)
total_credit_limit = st.number_input("Total Credit Limit", min_value=0.0, value=18000.0)
account_never_delinq_percent = st.number_input("Account Never Delinquent (%)", min_value=0.0, max_value=100.0, value=92.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=9000.0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)

if st.button("🔍 Predict"):
    # Prepare input data
    input_data = pd.DataFrame([{
        'total_credit_utilized': total_credit_utilized,
        'total_credit_limit': total_credit_limit,
        'debt_to_income': debt_to_income,
        'annual_income': annual_income,
        'interest_rate': interest_rate,
        'loan_amount': loan_amount,
        'account_never_delinq_percent': account_never_delinq_percent,
        'inquiries_last_12m': inquiries_last_12m
    }])

    # Business rule logic
    meets_business_rule = (
        (debt_to_income < 0.4) and
        (annual_income > 50000) and
        (interest_rate < 20) and
        (account_never_delinq_percent > 80)
    )

    # Model prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show business rule result
    if meets_business_rule:
        st.info("✅ Business Rule: Meets Loan Approval Criteria")
    else:
        st.warning("⚠️ Business Rule: Does NOT Meet Loan Approval Criteria")

    # Show model result
    st.markdown("---")
    st.subheader("📊 Model Output")
    st.success(f"Prediction: {'Approved (1)' if prediction == 1 else 'Rejected (0)'}")
    st.info(f"Probability of Approval: {probability:.2%}")
