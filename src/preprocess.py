"""
preprocess.py
=============
Cleans and transforms the raw DataFrame into model-ready feature arrays.

Steps performed
---------------
1. Drop ID columns (road_id, event_id) — not model features.
2. Separate features (X) from target (y).
3. Add any optional numeric columns that are present in the data.
4. Impute missing values:
     numeric    → median  (robust to outliers)
     categorical → most-frequent (mode)
5. Encode categorical columns with one-hot encoding.
6. Scale numeric columns with StandardScaler.

The fitted preprocessor (sklearn Pipeline) is returned so it can be
applied identically to future prediction data (no data leakage).

Usage
-----
    from src.preprocess import build_preprocessor, apply_preprocessor

    # During training:
    preprocessor = build_preprocessor(df_train)
    X_train, feature_names = apply_preprocessor(preprocessor, df_train, fit=True)
    X_test,  _             = apply_preprocessor(preprocessor, df_test, fit=False)

    # During prediction on new data:
    X_new, _ = apply_preprocessor(preprocessor, df_new, fit=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    OPTIONAL_NUMERIC,
    TARGET_COLUMN,
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numeric_cols, categorical_cols) actually present in *df*,
    including any optional columns that exist.
    """
    numeric_cols = list(NUMERIC_FEATURES)  # copy

    # Include optional numeric columns only if they are in the DataFrame
    for col in OPTIONAL_NUMERIC:
        if col in df.columns:
            numeric_cols.append(col)

    categorical_cols = list(CATEGORICAL_FEATURES)

    # Filter to only columns that actually exist in df
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    return numeric_cols, categorical_cols


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build (but do NOT yet fit) a sklearn ColumnTransformer for *df*.

    Parameters
    ----------
    df : training DataFrame (used only to determine which columns exist)

    Returns
    -------
    preprocessor : unfitted ColumnTransformer
    """
    numeric_cols, categorical_cols = get_feature_columns(df)

    # Numeric pipeline: impute with median → scale to zero mean / unit variance
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute with most frequent → one-hot encode
    # handle_unknown='ignore' ensures unseen categories during prediction
    # don't crash the pipeline.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",  # silently drop ID columns and any others
    )

    return preprocessor


def apply_preprocessor(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    fit: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Apply the preprocessor to *df* and return (X_array, feature_names).

    Parameters
    ----------
    preprocessor : ColumnTransformer returned by build_preprocessor()
    df           : DataFrame to transform
    fit          : if True, fit the preprocessor first (training set only!)
                   if False, only transform (test / prediction data)

    Returns
    -------
    X            : numpy array of shape (n_rows, n_features)
    feature_names: list of feature names matching X columns
    """
    # Drop target and ID columns before transforming
    cols_to_drop = [c for c in [TARGET_COLUMN] + ID_COLUMNS if c in df.columns]
    df_features = df.drop(columns=cols_to_drop)

    if fit:
        X = preprocessor.fit_transform(df_features)
    else:
        X = preprocessor.transform(df_features)

    feature_names = _get_feature_names(preprocessor)
    return X, feature_names


def extract_target(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the target column as a numpy array.

    Raises
    ------
    KeyError : if TARGET_COLUMN is not in df (call validate first!)
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in DataFrame. "
            "This column must contain 0/1 waterlogging labels from real "
            "field observations or official flood event records."
        )
    return df[TARGET_COLUMN].values.astype(int)


def split_by_group(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train and test sets such that ALL rows belonging to
    the same event_id stay together in one split.

    Why this matters
    ----------------
    A road segment can appear in multiple rainfall events.  If we split
    randomly at the row level, data from the SAME EVENT could appear in
    both train and test — the model would effectively see test data during
    training (data leakage), leading to over-optimistic accuracy on paper
    but poor real-world performance.

    By splitting on event_id instead of on rows, we ensure the model is
    evaluated on truly unseen rainfall events.

    Parameters
    ----------
    df          : full DataFrame (must contain 'event_id' column)
    test_size   : fraction of events to hold out for testing (default 20%)
    random_state: random seed for reproducibility

    Returns
    -------
    df_train, df_test : DataFrames
    """
    from src.schema import GROUP_COLUMN

    if GROUP_COLUMN not in df.columns:
        # Fallback: simple random row split with a warning
        import warnings
        warnings.warn(
            f"'{GROUP_COLUMN}' column not found. "
            "Falling back to random row-level split — this may cause mild "
            "data leakage if the same road appears in multiple events.",
            UserWarning,
            stacklevel=2,
        )
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=random_state)

    rng = np.random.default_rng(random_state)
    unique_groups = np.array(df[GROUP_COLUMN].unique())  # ensure plain numpy array
    rng.shuffle(unique_groups)

    n_test = max(1, int(len(unique_groups) * test_size))
    test_groups = set(unique_groups[:n_test])

    df_test = df[df[GROUP_COLUMN].isin(test_groups)].reset_index(drop=True)
    df_train = df[~df[GROUP_COLUMN].isin(test_groups)].reset_index(drop=True)

    return df_train, df_test


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    Works with sklearn >= 1.0.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for very old sklearn (shouldn't occur with 1.9)
        names: list[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                try:
                    ohe = transformer.named_steps["encoder"]
                    for i, col in enumerate(cols):
                        for cat in ohe.categories_[i]:
                            names.append(f"{col}_{cat}")
                except Exception:
                    names.extend(cols)
        return names
