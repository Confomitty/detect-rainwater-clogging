"""
schema.py
=========
Single source of truth for the expected dataset columns and their types.

Your teammate must deliver a CSV (or multiple CSVs) whose columns match
REQUIRED_COLUMNS exactly.  Optional columns are used when present.

Column descriptions
-------------------
road_id         : Unique identifier for each road segment (string/int).
                  Example: "SB_001" or 101
event_id        : Identifier grouping readings from the same rainfall event.
                  Example: "EVT_2023_06_01"  ← used for group-aware splitting
latitude        : WGS-84 latitude of the road segment centroid (float, degrees)
longitude       : WGS-84 longitude of the road segment centroid (float, degrees)
elevation_m     : Mean elevation of the segment in metres above sea level (float)
slope_pct       : Road slope in percent (float, ≥ 0)
rainfall_mm_hr  : Rainfall intensity during the event in mm/hr (float, ≥ 0)
rainfall_mm_24h : Total rainfall in the 24 h before the event in mm (float, ≥ 0)
drain_density   : Length of drainage network within 500 m radius / area (float, ≥ 0)
imperv_pct      : Fraction of impervious surface within 100 m (float, 0–100)
land_cover      : Land-cover category of the segment (string category)
                  Allowed values: 'residential', 'commercial', 'road',
                  'green', 'water', 'industrial', 'mixed'
road_type       : Type of road (string category)
                  Allowed values: 'arterial', 'collector', 'local', 'highway'
waterlogged     : TARGET — 1 if the segment waterlogged during the event, 0 if not.
                  REQUIRED for training.  If missing the pipeline will stop and
                  explain what is needed.

Optional columns (used when present, ignored when absent)
---------------------------------------------------------
road_width_m    : Width of the road in metres (float, > 0)
speed_limit_kmh : Posted speed limit (float, > 0)
last_cleaned_days : Days since last drain cleaning (int, ≥ 0)
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Column specifications
# ------------------------------------------------------------------

# Columns that MUST be present for any pipeline step to run
REQUIRED_COLUMNS: list[str] = [
    "road_id",
    "event_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_pct",
    "rainfall_mm_hr",
    "rainfall_mm_24h",
    "drain_density",
    "imperv_pct",
    "land_cover",
    "road_type",
]

# Target column — must be present for training, not for prediction
TARGET_COLUMN: str = "waterlogged"

# Columns used as grouping key for train/test splitting
# (all rows sharing the same event_id stay in the same split)
GROUP_COLUMN: str = "event_id"

# Columns that identify a row but are not model features
ID_COLUMNS: list[str] = ["road_id", "event_id"]

# Numeric feature columns
NUMERIC_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_pct",
    "rainfall_mm_hr",
    "rainfall_mm_24h",
    "drain_density",
    "imperv_pct",
]

# Categorical feature columns
CATEGORICAL_FEATURES: list[str] = [
    "land_cover",
    "road_type",
]

# Optional numeric columns (included in model if present in CSV)
OPTIONAL_NUMERIC: list[str] = [
    "road_width_m",
    "speed_limit_kmh",
    "last_cleaned_days",
]

# Allowed values for categorical columns
ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "land_cover": [
        "residential",
        "commercial",
        "road",
        "green",
        "water",
        "industrial",
        "mixed",
    ],
    "road_type": ["arterial", "collector", "local", "highway"],
}

# Numeric bounds — (min, max); None means no bound on that side
NUMERIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "latitude": (10.0, 14.0),       # roughly Bangalore region
    "longitude": (76.0, 78.5),      # roughly Bangalore region
    "elevation_m": (700.0, 1000.0), # Silk Board area is ~910 m
    "slope_pct": (0.0, None),
    "rainfall_mm_hr": (0.0, None),
    "rainfall_mm_24h": (0.0, None),
    "drain_density": (0.0, None),
    "imperv_pct": (0.0, 100.0),
    "road_width_m": (1.0, None),
    "speed_limit_kmh": (5.0, 120.0),
    "last_cleaned_days": (0.0, None),
}

# Expected dtype categories for validation (used as a rough check)
DTYPE_EXPECTATIONS: dict[str, str] = {
    "elevation_m": "numeric",
    "slope_pct": "numeric",
    "rainfall_mm_hr": "numeric",
    "rainfall_mm_24h": "numeric",
    "drain_density": "numeric",
    "imperv_pct": "numeric",
    "latitude": "numeric",
    "longitude": "numeric",
    "land_cover": "categorical",
    "road_type": "categorical",
    "waterlogged": "numeric",  # 0 or 1
}

