"""
Preprocessing for the Telco Customer Churn dataset.

Steps:
- Drop customerID (identifier, not predictive)
- Convert TotalCharges to numeric (it's a known dirty column - has blanks)
- Encode target variable (Churn: Yes/No -> 1/0)
- One-hot encode categorical features
- Split into train/test sets
- Save processed data + the feature column list (needed at inference time)
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_DIR = "data/processed"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges sometimes arrives as a string with blanks (real-dataset quirk)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df = df.drop(columns=["customerID"])

    return df


def encode(df: pd.DataFrame):
    target = (df["Churn"] == "Yes").astype(int)
    features = df.drop(columns=["Churn"])

    features_encoded = pd.get_dummies(features, drop_first=True)

    return features_encoded, target


def main():
    df = load_and_clean(RAW_PATH)
    X, y = encode(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    # Save the exact column order/names - inference must match this exactly
    with open(f"{PROCESSED_DIR}/feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")
    print(f"Feature count: {X.shape[1]}")


if __name__ == "__main__":
    main()
