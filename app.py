import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

BRANDS = [
    "Acura", "Alfa Romeo", "Aston Martin", "Audi", "BMW", "Bentley",
    "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "FIAT",
    "Ferrari", "Ford", "GMC", "Genesis", "Honda", "Hyundai", "INFINITI",
    "Jaguar", "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln", "MINI",
    "Maserati", "Mazda", "Mercedes-Benz", "Mitsubishi", "Nissan",
    "Polestar", "Porsche", "RAM", "Rivian", "Rolls-Royce", "Subaru",
    "Tesla", "Toyota", "Volkswagen", "Volvo"
]

FUEL_TYPES = ["Gasoline", "Hybrid", "Diesel", "E85 Flex Fuel", "Plug-In Hybrid", "Other"]
TRANSMISSIONS = ["Automatic", "Manual", "Other"]

@st.cache_resource
def load_artifacts():
    model        = joblib.load("best_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    log_target   = joblib.load("log_target.pkl")
    return model, scaler, feature_cols, log_target

def build_input_row(brand, model_year, mileage, horsepower,
                    fuel_type, transmission, accident, clean_title,
                    feature_cols):
    row = {col: 0 for col in feature_cols}
    row["model_year"]       = model_year
    row["mileage"]          = mileage
    row["horsepower"]       = horsepower
    row["accident_flag"]    = 1 if accident == "Yes (accident reported)" else 0
    row["clean_title_flag"] = 1 if clean_title == "Yes" else 0

    brand_col = f"brand_{brand}"
    if brand_col in row:
        row[brand_col] = 1

    fuel_col = f"fuel_type_clean_{fuel_type}"
    if fuel_col in row:
        row[fuel_col] = 1

    trans_col = f"transmission_clean_{transmission}"
    if trans_col in row:
        row[trans_col] = 1

    return pd.DataFrame([row])

def main():
    st.title("🚗 Used Car Price Predictor")
    st.write(
        "Get an estimated market price for a used car based on real listings "
        "from **cars.com**. Adjust the inputs in the sidebar and click **Predict**."
    )

    model, scaler, feature_cols, log_target = load_artifacts()

    st.sidebar.header("Car Details")
    brand        = st.sidebar.selectbox("Brand", BRANDS)
    model_year   = st.sidebar.slider("Model Year", min_value=2000, max_value=2024, value=2018)
    mileage      = st.sidebar.number_input("Mileage", min_value=0, max_value=300000, value=45000, step=1000)
    horsepower   = st.sidebar.number_input("Horsepower (HP)", min_value=50, max_value=1000, value=250, step=10)
    fuel_type    = st.sidebar.selectbox("Fuel Type", FUEL_TYPES)
    transmission = st.sidebar.selectbox("Transmission", TRANSMISSIONS)
    accident     = st.sidebar.selectbox("Accident History", ["None reported", "Yes (accident reported)"])
    clean_title  = st.sidebar.selectbox("Clean Title?", ["Yes", "No"])

    if st.sidebar.button("Predict Price", type="primary"):
        input_df     = build_input_row(brand, model_year, mileage, horsepower,
                                       fuel_type, transmission, accident, clean_title,
                                       feature_cols)
        input_scaled = scaler.transform(input_df)
        pred_log     = model.predict(input_scaled)[0]
        prediction   = np.expm1(pred_log) if log_target else pred_log
        prediction   = max(prediction, 0)

        st.success(f"### Estimated Market Price: **${prediction:,.0f}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Brand", brand)
        col2.metric("Year", model_year)
        col3.metric("Mileage", f"{mileage:,} mi")

        with st.expander("Full input details"):
            st.write({
                "Brand": brand, "Model Year": model_year,
                "Mileage": f"{mileage:,} mi", "Horsepower": f"{horsepower} HP",
                "Fuel Type": fuel_type, "Transmission": transmission,
                "Accident History": accident, "Clean Title": clean_title,
            })

    st.divider()
    st.caption(
        "Model trained on 4,009 real used car listings scraped from cars.com. "
        "Gradient Boosting Regressor — R² = 0.87, MAE ≈ $8,500."
    )

if __name__ == "__main__":
    main()