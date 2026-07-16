"""
app.py
Streamlit app for Car Purchase Price Prediction.
Run with: streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Car Purchase Price Predictor",
    page_icon="🚗",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def main():
    st.title("🚗 Car Purchase Price Predictor")
    st.write(
        "Estimate how much a customer is likely to spend on a car "
        "based on their demographic and financial profile."
    )

    model, scaler = load_artifacts()

    st.sidebar.header("Customer Details")
    gender = st.sidebar.selectbox("Gender", options=["Female", "Male"])
    gender_value = 1 if gender == "Male" else 0
    age = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
    annual_salary = st.sidebar.number_input("Annual Salary (USD)", min_value=0.0, value=60000.0, step=1000.0)
    credit_card_debt = st.sidebar.number_input("Credit Card Debt (USD)", min_value=0.0, value=5000.0, step=500.0)
    net_worth = st.sidebar.number_input("Net Worth (USD)", min_value=0.0, value=500000.0, step=10000.0)

    if st.sidebar.button("Predict Purchase Amount", type="primary"):
        input_df = pd.DataFrame(
            [[gender_value, age, annual_salary, credit_card_debt, net_worth]],
            columns=["gender", "age", "annual_salary", "credit_card_debt", "net_worth"],
        )
        input_scaled = scaler.transform(input_df)
        prediction = max(model.predict(input_scaled)[0], 0)
        st.success(f"### Estimated Car Purchase Amount: **${prediction:,.2f}**")
        with st.expander("See input summary"):
            st.dataframe(input_df)

    st.divider()
    st.caption("Model trained on customer demographic and financial data using scikit-learn.")


if __name__ == "__main__":
    main()
