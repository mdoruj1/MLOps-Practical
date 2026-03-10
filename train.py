"""
Phase 2: MLflow Experiment Tracking
------------------------------------
Run: python train.py --n_estimators 100 --max_depth 7
Then: mlflow ui  →  open http://127.0.0.1:5000
"""

import argparse
import pickle
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train(n_estimators: int, max_depth: int, learning_rate: float):
    # ── Load Data ──────────────────────────────────────────────────────────
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # ── MLflow: auto-log everything ────────────────────────────────────────
    mlflow.set_experiment("iris-classification")
    mlflow.sklearn.autolog()           # captures params, metrics, model artifact

    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):

        # Manual params (autolog also picks up sklearn params automatically)
        mlflow.log_param("learning_rate", learning_rate)

        # ── Train ──────────────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ── Log extra metrics ──────────────────────────────────────────────
        metrics = {
            "accuracy":  accuracy_score(y_test, preds),
            "f1_score":  f1_score(y_test, preds, average="weighted"),
            "precision": precision_score(y_test, preds, average="weighted"),
            "recall":    recall_score(y_test, preds, average="weighted"),
        }
        mlflow.log_metrics(metrics)

        # ── Save model locally too ────────────────────────────────────────
        with open("models/rf_model.pkl", "wb") as f:
            pickle.dump(model, f)

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Run ID: {run_id}")
        for k, v in metrics.items():
            print(f"   {k:12s}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int,   default=100)
    parser.add_argument("--max_depth",    type=int,   default=7)
    parser.add_argument("--learning_rate",type=float, default=0.1)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth, args.learning_rate)
