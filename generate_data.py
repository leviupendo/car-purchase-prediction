"""
generate_data.py
Generates a synthetic car_purchasing.csv dataset.
"""
import numpy as np
import pandas as pd
from faker import Faker

np.random.seed(42)
fake = Faker()
Faker.seed(42)

N = 500
countries = ["Kenya", "United States", "United Kingdom", "Germany", "Canada",
             "Nigeria", "South Africa", "India", "Australia", "France"]

genders = np.random.randint(0, 2, size=N)
ages = np.random.normal(loc=40, scale=12, size=N).clip(18, 70).round(0)
annual_salary = np.random.normal(loc=60000, scale=20000, size=N).clip(15000, 150000).round(2)
credit_card_debt = np.random.normal(loc=8000, scale=4000, size=N).clip(0, 30000).round(2)
net_worth = np.random.normal(loc=500000, scale=250000, size=N).clip(10000, 2000000).round(2)

base = (
    0.15 * annual_salary
    + 0.02 * net_worth
    - 0.10 * credit_card_debt
    + 200 * (35 - np.abs(ages - 35))
)
noise = np.random.normal(loc=0, scale=3000, size=N)
car_purchase_amount = (base + noise).clip(5000, 80000).round(2)

df = pd.DataFrame({
    "customer_name": [fake.name() for _ in range(N)],
    "customer_email": [fake.email() for _ in range(N)],
    "country": np.random.choice(countries, size=N),
    "gender": genders,
    "age": ages,
    "annual_salary": annual_salary,
    "credit_card_debt": credit_card_debt,
    "net_worth": net_worth,
    "car_purchase_amount": car_purchase_amount,
})

df.to_csv("car_purchasing.csv", index=False)
print(f"Generated car_purchasing.csv with {len(df)} rows")
print(df.head())
