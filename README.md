# Used Car Price Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![R2](https://img.shields.io/badge/R²-0.86-brightgreen)

End-to-end machine learning system that predicts the **market price of a used car** based on real listings scraped from [cars.com](https://www.cars.com). Deployed as an interactive **Streamlit** web app.

## Dataset

**Source:** [Used Car Price Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)

- 4,009 real vehicle listings from cars.com
- Features: brand, model year, mileage, horsepower, fuel type, transmission, accident history, clean title

## Project Structure
car-purchase-prediction/
├── used_cars.csv             # Real dataset from cars.com (via Kaggle)
├── data_preprocessing.py     # Cleaning, feature engineering, scaling
├── train_model.py            # Model training + GridSearchCV tuning
├── app.py                    # Streamlit deployment app
├── best_model.pkl            # Trained Gradient Boosting model
├── scaler.pkl                # Fitted StandardScaler
├── feature_cols.pkl          # Feature column names for inference
├── log_target.pkl            # Flag: target was log-transformed
├── requirements.txt
└── README.md

## Model Performance

| Model               | R² (log scale) | MAE      | RMSE     |
|---------------------|----------------|----------|----------|
| Linear Regression   | 0.81           | $11,313  | $24,356  |
| Random Forest       | 0.83           | $10,188  | $20,797  |
| **Gradient Boosting** | **0.86**     | **$8,854** | **$16,729** |

> Price was log-transformed before training to handle the natural right skew of car prices.

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/leviupendo/car-purchase-prediction.git
cd car-purchase-prediction
```

### 2. Set up environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` — use the sidebar to enter car details and get a price estimate.

## How It Works

1. **Preprocessing** — cleans price/mileage strings, extracts horsepower from engine description, encodes accident history and clean title as binary flags, one-hot encodes brand, fuel type, and transmission
2. **Training** — compares Linear Regression, Random Forest, and Gradient Boosting with GridSearchCV (3-fold CV), selects best model by R²
3. **Deployment** — Streamlit app loads saved model and scaler, takes car details as input, returns predicted market price

## License

MIT License — see `LICENSE` for details.