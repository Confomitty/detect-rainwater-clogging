"""
tests/test_train.py
===================
Tests for src/train.py and src/evaluate.py using synthetic data.

These tests confirm the pipeline runs without crashing.
They do NOT test model accuracy — that would require real labelled data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.preprocess import (
    apply_preprocessor,
    build_preprocessor,
    extract_target,
    split_by_group,
)
from src.train import load_model, save_model, train_models
from src.evaluate import evaluate_model, compare_models


# ------------------------------------------------------------------
# Helper — build preprocessed arrays from synthetic_df
# ------------------------------------------------------------------

@pytest.fixture
def preprocessed_data(synthetic_df):
    df_train, df_test = split_by_group(synthetic_df, test_size=0.2)
    preprocessor = build_preprocessor(df_train)
    X_train, feature_names = apply_preprocessor(preprocessor, df_train, fit=True)
    X_test, _ = apply_preprocessor(preprocessor, df_test, fit=False)
    y_train = extract_target(df_train)
    y_test = extract_target(df_test)
    return X_train, X_test, y_train, y_test, preprocessor, feature_names


# ------------------------------------------------------------------
# Training tests
# ------------------------------------------------------------------

class TestTrainModels:
    def test_returns_both_models(self, preprocessed_data):
        X_train, _, y_train, _, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        assert "logistic_regression" in models
        assert "random_forest" in models

    def test_models_can_predict(self, preprocessed_data):
        X_train, X_test, y_train, _, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        for name, model in models.items():
            preds = model.predict(X_test)
            assert len(preds) == len(X_test), f"{name}: wrong prediction length"
            assert set(preds).issubset({0, 1}), f"{name}: predictions must be 0 or 1"

    def test_models_have_predict_proba(self, preprocessed_data):
        X_train, X_test, y_train, _, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        for name, model in models.items():
            assert hasattr(model, "predict_proba"), (
                f"{name} should support predict_proba for risk scoring"
            )
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2)
            # Probabilities must sum to ~1 per row
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ------------------------------------------------------------------
# Save / Load tests
# ------------------------------------------------------------------

class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, preprocessed_data):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = (
            preprocessed_data
        )
        models = train_models(X_train, y_train)
        model = models["random_forest"]

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_model.pkl"
            save_model(model, preprocessor, feature_names, pkl_path)

            assert pkl_path.exists(), "Saved .pkl file should exist"

            loaded_model, loaded_preprocessor, loaded_names = load_model(pkl_path)

        # Loaded model should produce the same predictions
        preds_original = model.predict(X_test)
        preds_loaded = loaded_model.predict(X_test)
        assert np.array_equal(preds_original, preds_loaded), (
            "Loaded model must produce identical predictions"
        )
        assert loaded_names == feature_names


# ------------------------------------------------------------------
# Evaluation tests
# ------------------------------------------------------------------

class TestEvaluateModel:
    def test_metrics_keys_present(self, preprocessed_data):
        X_train, X_test, y_train, y_test, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        metrics = evaluate_model(
            models["logistic_regression"], X_test, y_test,
            model_name="Logistic Regression"
        )
        required_keys = [
            "model_name", "accuracy", "precision", "recall",
            "f1", "confusion_matrix", "classification_report",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_confusion_matrix_shape(self, preprocessed_data):
        X_train, X_test, y_train, y_test, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        metrics = evaluate_model(models["random_forest"], X_test, y_test)
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_metrics_in_valid_range(self, preprocessed_data):
        X_train, X_test, y_train, y_test, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        for name, model in models.items():
            m = evaluate_model(model, X_test, y_test, model_name=name)
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0

    def test_compare_models_runs_without_error(self, preprocessed_data):
        X_train, X_test, y_train, y_test, _, _ = preprocessed_data
        models = train_models(X_train, y_train)
        all_metrics = [
            evaluate_model(m, X_test, y_test, model_name=n)
            for n, m in models.items()
        ]
        # Should print without raising
        compare_models(all_metrics)

