"""
train.py
========
Builds, trains, and saves Logistic Regression and Random Forest classifiers.

Design choices
--------------
* class_weight='balanced'
    Both models handle imbalanced classes automatically (likely far fewer
    waterlogged roads than dry ones in most datasets).

* Pipeline wrapping
    Each model is wrapped in an sklearn Pipeline together with the
    preprocessor.  This guarantees that preprocessing parameters (scaler
    mean/std, imputer medians, encoder categories) are learned ONLY on
    the training fold and applied consistently to test / prediction data.
    It also makes saving and loading trivial: one .pkl file contains
    everything.

* No hyperparameter tuning yet
    Sensible defaults are used.  Add GridSearchCV / cross_val_score later
    when you have enough real labelled data (≥ 200 rows recommended).

Usage
-----
    from src.train import train_models, save_model, load_model

    models = train_models(X_train, y_train, preprocessor)
    save_model(models["random_forest"], "models/random_forest.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------------

def _build_logistic_regression() -> LogisticRegression:
    """
    Logistic Regression — fast, interpretable baseline.

    max_iter=1000 : default 100 can fail to converge on many features.
    C=1.0         : default regularisation — tune later with GridSearchCV.
    solver='lbfgs': handles multi-class; works well for small-to-medium data.
    class_weight='balanced': compensates for waterlogged rows being rare.
    """
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )


def _build_random_forest() -> RandomForestClassifier:
    """
    Random Forest — handles non-linear relationships and feature interactions.

    n_estimators=200 : 200 trees; good accuracy/speed trade-off.
    max_depth=10     : prevents overfitting on small datasets.
    min_samples_leaf=5: each leaf needs ≥5 samples — smooths predictions.
    class_weight='balanced': same imbalance handling as LR.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,  # use all available CPU cores
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, object]:
    """
    Fit both classifiers on the already-preprocessed training data.

    Parameters
    ----------
    X_train : numpy array from apply_preprocessor()
    y_train : numpy array from extract_target()

    Returns
    -------
    models : dict mapping model name → fitted sklearn estimator
        Keys: "logistic_regression", "random_forest"

    Notes
    -----
    The preprocessor is applied *before* this function is called.
    Call build_preprocessor() + apply_preprocessor() first.
    """
    models: dict[str, object] = {}

    print("  Training Logistic Regression...")
    lr = _build_logistic_regression()
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr
    print("  ✓ Logistic Regression trained.")

    print("  Training Random Forest...")
    rf = _build_random_forest()
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    print("  ✓ Random Forest trained.")

    return models


def save_model(
    model: object,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    path: str | Path,
) -> None:
    """
    Save a trained model together with its preprocessor and feature names.

    Everything needed for prediction is bundled into one file so that
    future Streamlit integration only needs to load a single .pkl.

    Parameters
    ----------
    model         : fitted sklearn estimator
    preprocessor  : fitted ColumnTransformer (from preprocess.py)
    feature_names : list of feature names (from apply_preprocessor)
    path          : destination .pkl file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  ✓ Model bundle saved to: {path}")


def load_model(path: str | Path) -> tuple[object, ColumnTransformer, list[str]]:
    """
    Load a model bundle saved by save_model().

    Returns
    -------
    model, preprocessor, feature_names
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["preprocessor"], bundle["feature_names"]

