"""
Phase 4: Pytest Test Suite for CI/CD
---------------------------------------
Run: pytest tests/ -v
Used by GitHub Actions on every git push.
"""

import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    with open("models/rf_model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def iris_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names


# ── Phase 1: Model Artifact Tests ─────────────────────────────────────────────
class TestModelArtifact:
    def test_model_file_exists(self):
        """DVC-tracked model file must be present."""
        import os
        assert os.path.exists("models/rf_model.pkl"), "Model file missing!"

    def test_model_loads_successfully(self, model):
        """Model must unpickle without errors."""
        assert model is not None

    def test_model_has_predict_method(self, model):
        assert hasattr(model, "predict") and callable(model.predict)

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba")


# ── Phase 2: Prediction Sanity Tests ─────────────────────────────────────────
class TestPredictions:
    def test_output_shape(self, model, iris_data):
        X, _, _ = iris_data
        preds = model.predict(X)
        assert preds.shape == (len(X),), "Wrong output shape"

    def test_output_classes(self, model, iris_data):
        X, _, _ = iris_data
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2}), "Unexpected class labels"

    def test_proba_sums_to_one(self, model, iris_data):
        X, _, _ = iris_data
        probas = model.predict_proba(X)
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1"

    def test_proba_in_range(self, model, iris_data):
        X, _, _ = iris_data
        probas = model.predict_proba(X)
        assert (probas >= 0).all() and (probas <= 1).all()


# ── Phase 3: Performance Gate (CI breaks if accuracy drops) ──────────────────
class TestPerformanceGate:
    ACCURACY_THRESHOLD = 0.90   # ← raise this for stricter gate

    def test_accuracy_above_threshold(self, model, iris_data):
        """
        CRITICAL: CI pipeline fails if model accuracy drops below threshold.
        This detects model decay caused by data drift or code bugs.
        """
        X, y, _ = iris_data
        preds    = model.predict(X)
        accuracy = accuracy_score(y, preds)
        print(f"\n  Model accuracy: {accuracy:.4f} (threshold: {self.ACCURACY_THRESHOLD})")
        assert accuracy >= self.ACCURACY_THRESHOLD, (
            f"Model accuracy {accuracy:.4f} is below threshold {self.ACCURACY_THRESHOLD}. "
            f"Possible data drift or model corruption! Retrain required."
        )

    def test_no_class_completely_missed(self, model, iris_data):
        """Model must predict all three classes — not collapse to 1 or 2."""
        X, _, _ = iris_data
        preds = model.predict(X)
        assert len(set(preds)) == 3, "Model is not predicting all 3 classes"

    def test_single_sample_prediction(self, model):
        """Smoke test: single known setosa sample."""
        setosa = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred   = model.predict(setosa)[0]
        assert pred == 0, f"Expected class 0 (setosa), got {pred}"


# ── Phase 4: Data Integrity Tests ────────────────────────────────────────────
class TestDataIntegrity:
    def test_training_data_exists(self):
        import os
        assert os.path.exists("data/iris.csv")
        assert os.path.exists("data/train_reference.csv")

    def test_no_missing_values(self):
        df = pd.read_csv("data/iris.csv")
        assert df.isnull().sum().sum() == 0, "Missing values in dataset!"

    def test_expected_row_count(self):
        df = pd.read_csv("data/iris.csv")
        assert len(df) == 150, f"Expected 150 rows, got {len(df)}"

    def test_feature_value_ranges(self):
        df = pd.read_csv("data/iris.csv")
        assert df["sepal length (cm)"].between(0, 20).all()
        assert df["petal length (cm)"].between(0, 10).all()

    def test_dvc_pointer_exists(self):
        import os
        assert os.path.exists("data/iris.csv.dvc"), (
            "DVC pointer file missing. Run: dvc add data/iris.csv"
        )
