"""
tests/test_preprocess.py
========================
Tests for src/preprocess.py using synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    apply_preprocessor,
    build_preprocessor,
    extract_target,
    get_feature_columns,
    split_by_group,
)
from src.schema import TARGET_COLUMN


class TestGetFeatureColumns:
    def test_returns_expected_numeric_columns(self, synthetic_df):
        numeric, categorical = get_feature_columns(synthetic_df)
        assert "elevation_m" in numeric
        assert "rainfall_mm_hr" in numeric
        assert "imperv_pct" in numeric

    def test_returns_expected_categorical_columns(self, synthetic_df):
        _, categorical = get_feature_columns(synthetic_df)
        assert "land_cover" in categorical
        assert "road_type" in categorical

    def test_optional_columns_included_when_present(self, synthetic_df):
        df = synthetic_df.copy()
        df["road_width_m"] = 10.0  # add optional column
        numeric, _ = get_feature_columns(df)
        assert "road_width_m" in numeric

    def test_optional_columns_excluded_when_absent(self, synthetic_df):
        numeric, _ = get_feature_columns(synthetic_df)
        assert "road_width_m" not in numeric


class TestBuildPreprocessor:
    def test_preprocessor_builds_without_error(self, synthetic_df):
        preprocessor = build_preprocessor(synthetic_df)
        assert preprocessor is not None

    def test_fit_produces_correct_output_shape(self, synthetic_df):
        preprocessor = build_preprocessor(synthetic_df)
        X, feature_names = apply_preprocessor(preprocessor, synthetic_df, fit=True)
        assert X.ndim == 2
        assert X.shape[0] == len(synthetic_df)
        assert X.shape[1] == len(feature_names)
        assert len(feature_names) > 0

    def test_output_has_no_nan(self, synthetic_df):
        preprocessor = build_preprocessor(synthetic_df)
        X, _ = apply_preprocessor(preprocessor, synthetic_df, fit=True)
        assert not np.isnan(X).any(), "Preprocessed array must not contain NaN"

    def test_transform_only_doesnt_refit(self, synthetic_df):
        """Applying fit=False uses training scaler parameters."""
        df_train = synthetic_df.iloc[:60].reset_index(drop=True)
        df_test = synthetic_df.iloc[60:].reset_index(drop=True)
        preprocessor = build_preprocessor(df_train)
        X_train, _ = apply_preprocessor(preprocessor, df_train, fit=True)
        X_test, _ = apply_preprocessor(preprocessor, df_test, fit=False)
        # Both should have same number of feature columns
        assert X_train.shape[1] == X_test.shape[1]

    def test_missing_values_handled(self, synthetic_df):
        df = synthetic_df.copy()
        df.loc[:10, "elevation_m"] = np.nan  # introduce NaN
        preprocessor = build_preprocessor(df)
        X, _ = apply_preprocessor(preprocessor, df, fit=True)
        assert not np.isnan(X).any()


class TestExtractTarget:
    def test_returns_numpy_array(self, synthetic_df):
        y = extract_target(synthetic_df)
        assert isinstance(y, np.ndarray)
        assert len(y) == len(synthetic_df)

    def test_values_are_0_or_1(self, synthetic_df):
        y = extract_target(synthetic_df)
        assert set(y).issubset({0, 1})

    def test_raises_when_target_missing(self, synthetic_df_no_target):
        with pytest.raises(KeyError, match="waterlogged"):
            extract_target(synthetic_df_no_target)


class TestSplitByGroup:
    def test_no_overlap_between_splits(self, synthetic_df):
        df_train, df_test = split_by_group(synthetic_df, test_size=0.2)
        train_events = set(df_train["event_id"].unique())
        test_events = set(df_test["event_id"].unique())
        assert train_events.isdisjoint(test_events), (
            "Same event_id appeared in both train and test — data leakage!"
        )

    def test_all_rows_accounted_for(self, synthetic_df):
        df_train, df_test = split_by_group(synthetic_df, test_size=0.2)
        assert len(df_train) + len(df_test) == len(synthetic_df)

    def test_test_size_roughly_respected(self, synthetic_df):
        df_train, df_test = split_by_group(synthetic_df, test_size=0.2)
        # Allow ±2 events of tolerance due to integer rounding
        n_total_events = synthetic_df["event_id"].nunique()
        n_test_events = df_test["event_id"].nunique()
        expected = int(n_total_events * 0.2)
        assert abs(n_test_events - expected) <= 2

    def test_fallback_when_no_event_id(self, synthetic_df):
        df = synthetic_df.drop(columns=["event_id"])
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df_train, df_test = split_by_group(df, test_size=0.2)
            assert len(w) == 1
            assert "leakage" in str(w[0].message).lower()
        assert len(df_train) + len(df_test) == len(df)

