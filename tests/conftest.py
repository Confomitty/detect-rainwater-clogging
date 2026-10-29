"""
tests/conftest.py
=================
Shared synthetic dataset fixture used across all test files.

IMPORTANT: This data is entirely fake — randomly generated numbers
labelled arbitrarily.  It is used ONLY to verify that the pipeline
code runs without errors.  It tells us NOTHING about real waterlogging
patterns at Silk Board.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# Fixed seed so every test run produces identical synthetic data
RNG = np.random.default_rng(seed=0)

N_EVENTS = 10    # number of unique rainfall events
N_ROADS = 8      # number of unique road segments
N_ROWS = N_EVENTS * N_ROADS  # 80 rows total


def _make_synthetic_df(include_target: bool = True) -> pd.DataFrame:
    """
    Build a minimal synthetic DataFrame that satisfies the schema.

    Parameters
    ----------
    include_target : if True, include the 'waterlogged' column
    """
    event_ids = [f"EVT_{i:03d}" for i in range(N_EVENTS)]
    road_ids = [f"RD_{j:03d}" for j in range(N_ROADS)]

    # Every road appears once per event → N_EVENTS × N_ROADS rows
    rows = []
    for evt in event_ids:
        for rd in road_ids:
            rows.append({"road_id": rd, "event_id": evt})

    df = pd.DataFrame(rows)
    n = len(df)

    # Numeric features — plausible but fake values
    df["latitude"] = RNG.uniform(12.80, 12.98, n)
    df["longitude"] = RNG.uniform(77.55, 77.75, n)
    df["elevation_m"] = RNG.uniform(880, 940, n)
    df["slope_pct"] = RNG.uniform(0, 5, n)
    df["rainfall_mm_hr"] = RNG.uniform(0, 60, n)
    df["rainfall_mm_24h"] = RNG.uniform(0, 150, n)
    df["drain_density"] = RNG.uniform(0.1, 1.5, n)
    df["imperv_pct"] = RNG.uniform(20, 95, n)

    # Categorical features
    df["land_cover"] = RNG.choice(
        ["residential", "commercial", "road", "green", "mixed"], n
    )
    df["road_type"] = RNG.choice(
        ["arterial", "collector", "local", "highway"], n
    )

    # Target — arbitrarily assigned (NOT a real model!)
    if include_target:
        df["waterlogged"] = RNG.integers(0, 2, n)

    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Full synthetic DataFrame with target column."""
    return _make_synthetic_df(include_target=True)


@pytest.fixture
def synthetic_df_no_target() -> pd.DataFrame:
    """Synthetic DataFrame WITHOUT target column (for prediction tests)."""
    return _make_synthetic_df(include_target=False)

