"""
Generates a synthetic dataset matching the structure of the
IBM Telco Customer Churn dataset (commonly used on Kaggle).

NOTE: This is for local development/testing only.
For your actual portfolio project, download the real dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
and place it at data/raw/telco_churn.csv with the same column names.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7000

genders = np.random.choice(["Male", "Female"], N)
senior = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = np.random.randint(0, 73, N)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.random.choice(
    ["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.10])
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

online_security = np.random.choice(
    ["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
online_backup = np.random.choice(
    ["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22])
device_protection = np.random.choice(
    ["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22])
tech_support = np.random.choice(
    ["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
streaming_tv = np.random.choice(
    ["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22])
streaming_movies = np.random.choice(
    ["Yes", "No", "No internet service"], N, p=[0.39, 0.39, 0.22])

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

monthly_charges = np.round(np.random.uniform(18, 120, N), 2)
total_charges = np.round(monthly_charges * tenure +
                         np.random.normal(0, 50, N), 2)
total_charges = np.clip(total_charges, 0, None)

# Build churn probability based on realistic signals
churn_prob = (
    0.05
    + 0.25 * (contract == "Month-to-month")
    + 0.10 * (internet_service == "Fiber optic")
    + 0.08 * (payment_method == "Electronic check")
    - 0.002 * tenure
    + 0.0015 * monthly_charges
    - 0.05 * (partner == "Yes")
    - 0.05 * (dependents == "Yes")
)
churn_prob = np.clip(churn_prob, 0.02, 0.85)
churn = np.random.binomial(1, churn_prob)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"{i:04d}-{np.random.choice(list('ABCDEFGHIJ'))}{np.random.choice(list('KLMNOPQRST'))}{np.random.choice(list('UVWXYZ'))}{np.random.choice(list('123456789'))}" for i in range(N)],
    "gender": genders,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_label,
})

df.to_csv("data/raw/telco_churn.csv", index=False)
print(f"Generated {len(df)} rows -> data/raw/telco_churn.csv")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
