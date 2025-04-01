import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load('final_logistic_model.pkl')

# Set up the Streamlit page
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("📊 Credit Risk Predictor")
st.markdown("Enter applicant information below to predict loan approval risk.")

# Input form for user data
with st.form("input_form"):
    total_credit_utilized = st.number_input("Total Credit Utilized", min_value=0.0, step=100.0)
    total_credit_limit = st.number_input("Total Credit Limit", min_value=0.0, step=100.0)
    debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, step=0.01)
    annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    account_never_delinq_percent = st.number_input("Account Never Delinquent (%)", min_value=0.0, max_value=100.0, step=0.1)
    inquiries_last_12m = st.number_input("Inquiries in Last 12 Months", min_value=0, step=1)

    submitted = st.form_submit_button("Predict")

# Run prediction if form is submitted
if submitted:
    # Dynamically get expected feature names from the pipeline
    expected_features = model.named_steps['logr'].feature_names_in_

    # Map form input to dictionary
    input_values = {
        'total_credit_utilized': total_credit_utilized,
        'total_credit_limit': total_credit_limit,
        'debt_to_income': debt_to_income,
        'annual_income': annual_income,
        'interest_rate': interest_rate,
        'loan_amount': loan_amount,
        'account_never_delinq_percent': account_never_delinq_percent,
        'inquiries_last_12m': inquiries_last_12m
    }

    # Ensure correct order of features
    ordered_input = [[input_values[feature] for feature in expected_features]]
    input_data = pd.DataFrame(ordered_input, columns=expected_features)

    # Predict and show result
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.success(f"🔮 Prediction: {'High Risk (1)' if prediction == 1 else 'Low Risk (0)'}")
    st.info(f"📈 Probability of High Risk: {probability:.2%}")
