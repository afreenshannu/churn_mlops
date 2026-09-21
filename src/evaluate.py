"""
Standalone model evaluation gate.
Used by CI to fail the pipeline if model quality drops below thresholds.

Exit code 0 = passed, Exit code 1 = failed (blocks deployment).
"""

import json
import sys


THRESHOLDS = {
    "f1": 0.55,
    "roc_auc": 0.75,
}

METRICS_PATH = "models/metrics.json"


def main():
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        metrics = data["metrics"]
    except FileNotFoundError:
        print(f"ERROR: {METRICS_PATH} not found. Run train.py first.")
        sys.exit(1)

    print("\n── Model Evaluation Gate ──────────────────")
    all_passed = True

    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric)
        passed = value is not None and value >= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {metric}: {value} (threshold: {threshold})")
        if not passed:
            all_passed = False

    print("───────────────────────────────────────────")

    if all_passed:
        print("All checks passed. Model is ready for deployment.\n")
        sys.exit(0)
    else:
        print("One or more checks FAILED. Deployment blocked.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
