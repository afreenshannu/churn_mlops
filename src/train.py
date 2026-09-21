# # """
# # Trains a churn classification model on the preprocessed data.

# # Phase 1: plain script, no MLflow yet. We'll add experiment tracking in Phase 2.
# # Saves:
# # - model artifact (pickle)
# # - metrics (json) - used later by evaluate.py as a CI gate
# # """

# # import json
# # import pickle

# # import pandas as pd
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import (
# #     accuracy_score,
# #     f1_score,
# #     precision_score,
# #     recall_score,
# #     roc_auc_score,
# # )

# # PROCESSED_DIR = "data/processed"
# # MODEL_PATH = "models/model.pkl"
# # METRICS_PATH = "models/metrics.json"


# # def load_data():
# #     X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
# #     X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
# #     y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
# #     y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
# #     return X_train, X_test, y_train, y_test


# # def train(X_train, y_train, **params):
# #     model = RandomForestClassifier(random_state=42, **params)
# #     model.fit(X_train, y_train)
# #     return model


# # def evaluate(model, X_test, y_test):
# #     preds = model.predict(X_test)
# #     probs = model.predict_proba(X_test)[:, 1]

# #     return {
# #         "accuracy": round(accuracy_score(y_test, preds), 4),
# #         "precision": round(precision_score(y_test, preds), 4),
# #         "recall": round(recall_score(y_test, preds), 4),
# #         "f1": round(f1_score(y_test, preds), 4),
# #         "roc_auc": round(roc_auc_score(y_test, probs), 4),
# #     }


# # def main():
# #     import os
# #     os.makedirs("models", exist_ok=True)

# #     X_train, X_test, y_train, y_test = load_data()

# #     # Baseline hyperparameters - we'll experiment with these via MLflow in Phase 2
# #     params = {"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"}
# #     model = train(X_train, y_train, **params)

# #     metrics = evaluate(model, X_test, y_test)
# #     print("Metrics:", json.dumps(metrics, indent=2))

# #     with open(MODEL_PATH, "wb") as f:
# #         pickle.dump(model, f)

# #     with open(METRICS_PATH, "w") as f:
# #         json.dump({"params": params, "metrics": metrics}, f, indent=2)

# #     print(f"\nSaved model -> {MODEL_PATH}")
# #     print(f"Saved metrics -> {METRICS_PATH}")


# # if __name__ == "__main__":
# #     main()
# """
# Trains a churn classification model on the preprocessed data,
# logging every run to MLflow for experiment comparison.

# Usage:
#     python src/train.py
#     python src/train.py --n_estimators 200 --max_depth 12
#     python src/train.py --n_estimators 50 --max_depth 4 --class_weight None

# Each run logs:
# - params (hyperparameters)
# - metrics (accuracy, precision, recall, f1, roc_auc)
# - the trained model artifact

# Still saves models/model.pkl and models/metrics.json so the
# serving app and CI evaluation gate (Phase 4/5) don't depend on MLflow.
# """

# import argparse
# import json
# import os
# import pickle

# import mlflow
# import mlflow.sklearn
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )

# PROCESSED_DIR = "data/processed"
# MODEL_PATH = "models/model.pkl"
# METRICS_PATH = "models/metrics.json"
# EXPERIMENT_NAME = "churn-prediction"


# def load_data():
#     X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
#     X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
#     y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
#     y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
#     return X_train, X_test, y_train, y_test


# def train(X_train, y_train, **params):
#     model = RandomForestClassifier(random_state=42, **params)
#     model.fit(X_train, y_train)
#     return model


# def evaluate(model, X_test, y_test):
#     preds = model.predict(X_test)
#     probs = model.predict_proba(X_test)[:, 1]

#     return {
#         "accuracy": round(accuracy_score(y_test, preds), 4),
#         "precision": round(precision_score(y_test, preds), 4),
#         "recall": round(recall_score(y_test, preds), 4),
#         "f1": round(f1_score(y_test, preds), 4),
#         "roc_auc": round(roc_auc_score(y_test, probs), 4),
#     }


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_estimators", type=int, default=100)
#     parser.add_argument("--max_depth", type=int, default=8)
#     parser.add_argument(
#         "--class_weight",
#         type=str,
#         default="balanced",
#         help="'balanced' or 'None' (string 'None' disables it)",
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     os.makedirs("models", exist_ok=True)

#     class_weight = None if args.class_weight == "None" else args.class_weight
#     params = {
#         "n_estimators": args.n_estimators,
#         "max_depth": args.max_depth,
#         "class_weight": class_weight,
#     }

#     X_train, X_test, y_train, y_test = load_data()

#     mlflow.set_experiment(EXPERIMENT_NAME)

#     with mlflow.start_run():
#         model = train(X_train, y_train, **params)
#         metrics = evaluate(model, X_test, y_test)

#         mlflow.log_params(params)
#         mlflow.log_metrics(metrics)


#       # mlflow.sklearn.log_model(model, artifact_path="model")
#         # Skip model artifact logging in CI (skops security check blocks it)
# # For full artifact logging, run locally with mlflow ui
# try:
#     mlflow.sklearn.log_model(model, artifact_path="model")
# except Exception as e:
#     print(f"Warning: Could not log model artifact: {e}")

#     print("Params:", json.dumps(params, indent=2))
#     print("Metrics:", json.dumps(metrics, indent=2))

#     # Keep saving these for the serving app (Phase 4) and CI gate (Phase 5)
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(model, f)

#     with open(METRICS_PATH, "w") as f:
#         json.dump({"params": params, "metrics": metrics}, f, indent=2)

#     print(f"\nSaved model -> {MODEL_PATH}")
#     print(f"Saved metrics -> {METRICS_PATH}")
#     print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


# if __name__ == "__main__":
#     main()

import argparse
import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"
EXPERIMENT_NAME = "churn-prediction"


def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, **params):
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--class_weight", type=str, default="balanced")
    return parser.parse_args()


def log_to_mlflow(params, metrics):
    """Log to MLflow if available — skipped gracefully in CI."""
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        print(f"MLflow run logged (experiment: {EXPERIMENT_NAME})")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


def main():
    os.makedirs("models", exist_ok=True)

    args = parse_args()
    class_weight = None if args.class_weight == "None" else args.class_weight
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "class_weight": class_weight,
    }

    X_train, X_test, y_train, y_test = load_data()
    model = train(X_train, y_train, **params)
    metrics = evaluate(model, X_test, y_test)

    log_to_mlflow(params, metrics)

    print("Params:", json.dumps(params, indent=2))
    print("Metrics:", json.dumps(metrics, indent=2))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(METRICS_PATH, "w") as f:
        json.dump({"params": params, "metrics": metrics}, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
