"""
predict.py
==========
Load a saved model bundle and make predictions on new, unlabelled data.

This module is the entry point your future Streamlit app will call.

Usage
-----
    from src.predict import predict_from_csv, predict_from_dataframe

    # From a CSV file:
    results = predict_from_csv(
        model_path="models/random_forest.pkl",
        data_path="data/raw/new_roads.csv",
    )
    print(results)

    # From a DataFrame already in memory:
    results = predict_from_dataframe(
        model_path="models/random_forest.pkl",
        df=my_dataframe,
    )
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from src.train import load_model
from src.preprocess import apply_preprocessor
from src.validate import validate_dataframe


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def predict_from_csv(
    model_path: str | Path,
    data_path: str | Path,
) -> pd.DataFrame:
    """
    Load a saved model and predict waterlogging risk for roads in a CSV.

    Parameters
    ----------
    model_path : path to the .pkl bundle saved by train.save_model()
    data_path  : path to the new data CSV (no 'waterlogged' column needed)

    Returns
    -------
    results : DataFrame with original columns plus:
        predicted_waterlogged : 0 or 1
        prob_waterlogged      : probability of waterlogging (0.0 – 1.0)
                                (only if model supports predict_proba)
    """
    df = pd.read_csv(data_path)
    return predict_from_dataframe(model_path, df)


def predict_from_dataframe(
    model_path: str | Path,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load a saved model and predict on an in-memory DataFrame.

    Parameters
    ----------
    model_path : path to the .pkl bundle
    df         : input DataFrame (does NOT need 'waterlogged' column)

    Returns
    -------
    results : original DataFrame plus prediction columns
    """
    # Validate — do not require target column for prediction
    report = validate_dataframe(df, require_target=False)
    if not report["valid"]:
        raise ValueError(
            "Input data failed validation:\n" +
            "\n".join(f"  • {e}" for e in report["errors"])
        )
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  ⚠️  {w}")

    # Load model bundle
    model, preprocessor, feature_names = load_model(model_path)

    # Preprocess — fit=False means we use the scaler/encoder from training
    X, _ = apply_preprocessor(preprocessor, df, fit=False)

    # Predict
    y_pred = model.predict(X)
    results = df.copy()
    results["predicted_waterlogged"] = y_pred

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        results["prob_waterlogged"] = np.round(y_proba, 4)

    return results


def predict_single_road(
    model_path: str | Path,
    road_data: dict,
) -> dict:
    """
    Predict waterlogging risk for a single road segment described as a dict.

    Useful for Streamlit forms where a user fills in one road at a time.

    Parameters
    ----------
    model_path : path to the .pkl bundle
    road_data  : dict of feature values, e.g.:
                 {
                     "road_id": "SB_001",
                     "event_id": "EVT_TEST",
                     "latitude": 12.917,
                     "longitude": 77.622,
                     "elevation_m": 912.0,
                     "slope_pct": 1.2,
                     "rainfall_mm_hr": 35.0,
                     "rainfall_mm_24h": 80.0,
                     "drain_density": 0.45,
                     "imperv_pct": 78.0,
                     "land_cover": "commercial",
                     "road_type": "arterial",
                 }

    Returns
    -------
    result : dict with keys:
        "predicted_waterlogged": 0 or 1
        "prob_waterlogged"     : float (if available)
        "risk_label"           : "HIGH RISK" / "LOW RISK"
    """
    df = pd.DataFrame([road_data])
    results = predict_from_dataframe(model_path, df)

    output: dict = {
        "predicted_waterlogged": int(results["predicted_waterlogged"].iloc[0]),
    }
    if "prob_waterlogged" in results.columns:
        prob = float(results["prob_waterlogged"].iloc[0])
        output["prob_waterlogged"] = prob
        output["risk_label"] = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    else:
        output["risk_label"] = (
            "HIGH RISK" if output["predicted_waterlogged"] == 1 else "LOW RISK"
        )

    return output

