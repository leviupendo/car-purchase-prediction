# Car Purchase Price Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

End-to-end machine learning system that predicts how much a customer is likely to spend on a car, based on their demographic and financial profile (gender, age, annual salary, credit card debt, net worth). Deployed as an interactive **Streamlit** web app.

## Project Structure

```
car-purchase-prediction/
├── car_purchasing.csv        # Dataset (500 customers)
├── generate_data.py          # Script that generated the dataset
├── data_preprocessing.py     # Cleaning, train/test split, scaling
├── train_model.py            # Model training + GridSearchCV tuning
├── app.py                    # Streamlit deployment app
├── best_model.pkl            # Saved best regression model
├── scaler.pkl                # Saved StandardScaler
├── requirements.txt
└── README.md
```

## Features Used

| Column              | Description                        |
|---------------------|------------------------------------|
| gender              | 0 = Female, 1 = Male               |
| age                 | Customer age (years)               |
| annual_salary       | Annual salary (USD)                |
| credit_card_debt    | Outstanding credit card debt (USD) |
| net_worth           | Customer net worth (USD)           |
| car_purchase_amount | **Target** — purchase amount (USD) |

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

### 3. Generate dataset
```bash
python generate_data.py
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Model Performance

Three models are compared (Linear Regression, Random Forest, Gradient Boosting). The best by R² is auto-selected and saved. Baseline Linear Regression achieved **R² = 0.72**.

## License

MIT License — see `LICENSE` for details.
