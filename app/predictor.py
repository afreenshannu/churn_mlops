"""
Handles model loading and inference.
Separated from main.py so it can be tested independently.
"""

import json
import os
import pickle

import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "data/processed/feature_columns.json")


class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None

    def load(self):
        """Load model and feature schema from disk."""
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(FEATURE_COLUMNS_PATH, "r") as f:
            self.feature_columns = json.load(f)

        print(f"Model loaded from {MODEL_PATH}")
        print(f"Expecting {len(self.feature_columns)} features")

    def preprocess(self, raw_input: dict) -> pd.DataFrame:
        """
        Apply the same encoding used during training.
        raw_input is a flat dict of original column values (before one-hot encoding).
        """
        df = pd.DataFrame([raw_input])

        # One-hot encode exactly as preprocess.py did
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Align to the exact columns the model was trained on
        # Missing columns (unseen categories) become 0
        df_aligned = df_encoded.reindex(columns=self.feature_columns, fill_value=0)

        return df_aligned

    def predict(self, raw_input: dict) -> dict:
        """
        Returns churn prediction and probability for a single customer.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = self.preprocess(raw_input)
        prediction = int(self.model.predict(X)[0])
        probability = round(float(self.model.predict_proba(X)[0][1]), 4)

        return {
            "churn_prediction": prediction,         # 0 or 1
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": probability,        # 0.0 - 1.0
        }


# Singleton — loaded once at startup, reused for every request
predictor = ChurnPredictor()
