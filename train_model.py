import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data_preprocessing import preprocess

def evaluate(model, X_test, y_test_log, name):
    preds_log = model.predict(X_test)
    preds     = np.expm1(preds_log)
    actuals   = np.expm1(y_test_log)
    r2   = r2_score(y_test_log, preds_log)
    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"\n--- {name} ---")
    print(f"R^2 (log scale): {r2:.4f}")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    return {"name": name, "model": model, "r2": r2, "mae": mae, "rmse": rmse}

def train_and_select_best(model_path="best_model.pkl"):
    X_train, X_test, y_train, y_test, feature_cols = preprocess()

    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    joblib.dump(True, "log_target.pkl")

    results = []

    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    results.append(evaluate(lr, X_test, y_test_log, "Linear Regression"))

    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 5]},
        cv=3, scoring="r2", n_jobs=-1, verbose=0,
    )
    rf_grid.fit(X_train, y_train_log)
    print(f"Best RF params: {rf_grid.best_params_}")
    results.append(evaluate(rf_grid.best_estimator_, X_test, y_test_log, "Random Forest"))

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        cv=3, scoring="r2", n_jobs=-1, verbose=0,
    )
    gb_grid.fit(X_train, y_train_log)
    print(f"Best GB params: {gb_grid.best_params_}")
    results.append(evaluate(gb_grid.best_estimator_, X_test, y_test_log, "Gradient Boosting"))

    best = max(results, key=lambda r: r["r2"])
    print(f"\n=== Best model: {best['name']} (R^2={best['r2']:.4f}, MAE=${best['mae']:,.0f}) ===")
    joblib.dump(best["model"], model_path)
    print(f"Saved to {model_path}")
    return best

if __name__ == "__main__":
    train_and_select_best()