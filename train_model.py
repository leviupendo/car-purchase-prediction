"""
train_model.py
Trains, tunes, and selects the best regression model.
"""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data_preprocessing import preprocess


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\n--- {name} ---")
    print(f"R^2:  {r2:.4f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    return {"name": name, "model": model, "r2": r2, "mae": mae, "rmse": rmse}


def train_and_select_best(scaler_path: str = "scaler.pkl", model_path: str = "best_model.pkl"):
    X_train, X_test, y_train, y_test, scaler = preprocess(scaler_path=scaler_path)
    results = []

    # 1. Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate(lr, X_test, y_test, "Linear Regression"))

    # 2. Random Forest
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
        cv=5, scoring="r2", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"\nBest RF params: {rf_grid.best_params_}")
    results.append(evaluate(rf_grid.best_estimator_, X_test, y_test, "Random Forest"))

    # 3. Gradient Boosting
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        cv=5, scoring="r2", n_jobs=-1,
    )
    gb_grid.fit(X_train, y_train)
    print(f"\nBest GB params: {gb_grid.best_params_}")
    results.append(evaluate(gb_grid.best_estimator_, X_test, y_test, "Gradient Boosting"))

    best = max(results, key=lambda r: r["r2"])
    print(f"\n=== Best model: {best['name']} (R^2 = {best['r2']:.4f}) ===")
    joblib.dump(best["model"], model_path)
    print(f"Saved to {model_path}")
    return best


if __name__ == "__main__":
    train_and_select_best()
