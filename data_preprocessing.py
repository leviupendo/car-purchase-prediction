import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "price"

def clean_price(val):
    return float(str(val).replace("$", "").replace(",", "").strip())

def clean_mileage(val):
    return float(str(val).replace(" mi.", "").replace(",", "").strip())

def extract_hp(engine_str):
    match = re.search(r"([\d.]+)\s*HP", str(engine_str), re.IGNORECASE)
    return float(match.group(1)) if match else np.nan

def simplify_transmission(val):
    val = str(val).upper()
    if "MANUAL" in val or "M/T" in val:
        return "Manual"
    elif "AUTO" in val or "A/T" in val or "CVT" in val:
        return "Automatic"
    return "Other"

def clean_fuel(val):
    val = str(val).strip()
    if val in ["-", "not supported", "nan", ""]:
        return "Other"
    return val

def group_rare(series, min_count=5, other_label="Other"):
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.replace(rare, other_label)

def load_and_clean(path="used_cars.csv"):
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].apply(clean_price)
    df["mileage"] = df["milage"].apply(clean_mileage)
    df = df.drop(columns=["milage"])
    df["horsepower"] = df["engine"].apply(extract_hp)
    df = df.drop(columns=["engine"])
    df["accident"] = df["accident"].fillna("None reported")
    df["accident_flag"] = (df["accident"] == "At least 1 accident or damage reported").astype(int)
    df = df.drop(columns=["accident"])
    df["clean_title"] = df["clean_title"].fillna("No")
    df["clean_title_flag"] = (df["clean_title"] == "Yes").astype(int)
    df = df.drop(columns=["clean_title"])
    df["transmission_clean"] = df["transmission"].apply(simplify_transmission)
    df = df.drop(columns=["transmission"])
    df["fuel_type"] = df["fuel_type"].fillna("Other")
    df["fuel_type_clean"] = df["fuel_type"].apply(clean_fuel)
    df = df.drop(columns=["fuel_type"])
    df = df.drop(columns=["model", "ext_col", "int_col"])
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df["model_grouped"] = group_rare(df["model"] if "model" in df.columns else pd.Series(["Other"]*len(df)))
    price_cap = df[TARGET].quantile(0.995)
    df = df[df[TARGET] <= price_cap]
    df = pd.get_dummies(df, columns=["brand", "model_grouped", "fuel_type_clean", "transmission_clean"], drop_first=False)
    df = df.dropna()
    return df

def split_and_scale(df, scaler_path="scaler.pkl"):
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, "feature_cols.pkl")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def preprocess(path="used_cars.csv", scaler_path="scaler.pkl"):
    df = load_and_clean(path)
    print(f"Cleaned dataset shape: {df.shape}")
    return split_and_scale(df, scaler_path=scaler_path)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = preprocess()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")