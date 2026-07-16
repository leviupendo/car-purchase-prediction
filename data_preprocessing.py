"""
data_preprocessing.py
Loads, cleans, splits, and scales the car purchasing dataset.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["gender", "age", "annual_salary", "credit_card_debt", "net_worth"]
TARGET_COLUMN = "car_purchase_amount"


def load_data(path: str = "car_purchasing.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in ["customer_name", "customer_email", "country"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                    scaler_path: str = "scaler.pkl"):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preprocess(path: str = "car_purchasing.csv", scaler_path: str = "scaler.pkl"):
    df = load_data(path)
    df = clean_data(df)
    return split_and_scale(df, scaler_path=scaler_path)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features used: {FEATURE_COLUMNS}")
