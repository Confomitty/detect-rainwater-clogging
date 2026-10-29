"""
tests/test_validate.py
======================
Tests for src/validate.py using synthetic data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.validate import validate_dataframe


class TestValidateDataframe:
    # ── Happy path ──────────────────────────────────────────────

    def test_valid_data_passes(self, synthetic_df):
        report = validate_dataframe(synthetic_df, require_target=True)
        assert report["valid"] is True
        assert report["errors"] == []

    def test_stats_populated(self, synthetic_df):
        report = validate_dataframe(synthetic_df, require_target=True)
        assert report["stats"]["rows"] == len(synthetic_df)
        assert "waterlogged_rate_pct" in report["stats"]

    # ── Missing columns ─────────────────────────────────────────

    def test_missing_required_column_is_error(self, synthetic_df):
        df = synthetic_df.drop(columns=["elevation_m"])
        report = validate_dataframe(df, require_target=True)
        assert report["valid"] is False
        assert any("elevation_m" in e for e in report["errors"])

    def test_missing_target_when_required_is_error(self, synthetic_df_no_target):
        report = validate_dataframe(synthetic_df_no_target, require_target=True)
        assert report["valid"] is False
        assert any("waterlogged" in e for e in report["errors"])

    def test_missing_target_when_not_required_is_warning(
        self, synthetic_df_no_target
    ):
        report = validate_dataframe(synthetic_df_no_target, require_target=False)
        # Should still be valid (no blocking errors from missing target)
        assert report["valid"] is True
        assert any("waterlogged" in w for w in report["warnings"])

    # ── Empty DataFrame ─────────────────────────────────────────

    def test_empty_dataframe_is_invalid(self):
        report = validate_dataframe(pd.DataFrame(), require_target=True)
        assert report["valid"] is False
        assert "empty" in report["errors"][0].lower()

    # ── Categorical validation ───────────────────────────────────

    def test_invalid_land_cover_is_error(self, synthetic_df):
        df = synthetic_df.copy()
        df.loc[0, "land_cover"] = "unknown_category"
        report = validate_dataframe(df, require_target=True)
        assert report["valid"] is False
        assert any("land_cover" in e for e in report["errors"])

    def test_invalid_road_type_is_error(self, synthetic_df):
        df = synthetic_df.copy()
        df.loc[0, "road_type"] = "motorway"  # not in allowed list
        report = validate_dataframe(df, require_target=True)
        assert report["valid"] is False
        assert any("road_type" in e for e in report["errors"])

    # ── Numeric validation ───────────────────────────────────────

    def test_out_of_range_imperv_pct_is_warning(self, synthetic_df):
        df = synthetic_df.copy()
        df.loc[0, "imperv_pct"] = 150.0  # above 100%
        report = validate_dataframe(df, require_target=True)
        assert any("imperv_pct" in w for w in report["warnings"])

    def test_negative_rainfall_is_warning(self, synthetic_df):
        df = synthetic_df.copy()
        df.loc[0, "rainfall_mm_hr"] = -5.0
        report = validate_dataframe(df, require_target=True)
        assert any("rainfall_mm_hr" in w for w in report["warnings"])

    # ── Target value validation ──────────────────────────────────

    def test_invalid_target_values_are_error(self, synthetic_df):
        df = synthetic_df.copy()
        df.loc[0, "waterlogged"] = 2  # must be 0 or 1
        report = validate_dataframe(df, require_target=True)
        assert report["valid"] is False
        assert any("waterlogged" in e for e in report["errors"])

    # ── Missing values ───────────────────────────────────────────

    def test_missing_values_trigger_warning(self, synthetic_df):
        df = synthetic_df.copy()
        # Introduce >20% missing in one column
        df.loc[:20, "elevation_m"] = None
        report = validate_dataframe(df, require_target=True)
        assert any("elevation_m" in w for w in report["warnings"])

