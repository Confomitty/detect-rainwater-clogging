import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

from rasterstats import zonal_stats
from pathlib import Path


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

ROAD_FILE = "data/raw/roads/road_segments.geojson"

ELEVATION_FILE = "data/raw/terrain/elevation.tif"
SLOPE_FILE = "data/raw/terrain/slope.tif"
LANDCOVER_FILE = "data/raw/landcover/landcover.tif"

OUTPUT_DIR = Path("data/processed/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD ROAD NETWORK
# --------------------------------------------------

print("Loading road network...")

roads = gpd.read_file(ROAD_FILE)

if roads.empty:
    raise ValueError("Road network is empty.")

if "road_id" not in roads.columns:
    raise ValueError("road_id column is missing.")

print("Road segments:", len(roads))


# --------------------------------------------------
# 3. REPROJECT ROADS
# --------------------------------------------------

with rasterio.open(ELEVATION_FILE) as src:
    target_crs = src.crs

roads = roads.to_crs(target_crs)

print("Reprojected roads to:", target_crs)


# --------------------------------------------------
# 4. CREATE ROAD BUFFERS
# --------------------------------------------------

# A 10-metre buffer on either side of each road.
# This provides an area from which raster pixels
# can be sampled instead of sampling a zero-width line.

road_buffers = roads.geometry.buffer(10)

print("Created road buffers.")


# --------------------------------------------------
# 5. EXTRACT ELEVATION FEATURES
# --------------------------------------------------

print("Extracting elevation features...")

elevation_stats = zonal_stats(
    road_buffers,
    ELEVATION_FILE,
    stats=["mean", "min", "max"],
    all_touched=True
)

elevation_df = pd.DataFrame(elevation_stats)

elevation_df = elevation_df.rename(columns={
    "mean": "elevation_mean_m",
    "min": "elevation_min_m",
    "max": "elevation_max_m"
})


# --------------------------------------------------
# 6. EXTRACT SLOPE FEATURES
# --------------------------------------------------

print("Extracting slope features...")

slope_stats = zonal_stats(
    road_buffers,
    SLOPE_FILE,
    stats=["mean", "min", "max"],
    all_touched=True
)

slope_df = pd.DataFrame(slope_stats)

slope_df = slope_df.rename(columns={
    "mean": "slope_mean_deg",
    "min": "slope_min_deg",
    "max": "slope_max_deg"
})


# --------------------------------------------------
# 7. EXTRACT LAND COVER FEATURES
# --------------------------------------------------

print("Extracting land-cover features...")

with rasterio.open(LANDCOVER_FILE) as src:
    landcover_nodata = src.nodata

if landcover_nodata is None:
    landcover_nodata = 0

landcover_stats = zonal_stats(
    road_buffers,
    LANDCOVER_FILE,
    categorical=True,
    all_touched=True,
    nodata=landcover_nodata
)

# ESA WorldCover class codes
class_names = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "builtup",
    60: "bare",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangroves",
    100: "moss_lichen"
}

landcover_rows = []

for stats in landcover_stats:

    total_pixels = sum(stats.values())

    row = {}

    for class_code, class_name in class_names.items():

        count = stats.get(class_code, 0)

        percentage = (
            (count / total_pixels) * 100
            if total_pixels > 0 else np.nan
        )

        row[f"{class_name}_percentage"] = percentage

    # Aggregated land-cover features
    row["vegetation_percentage"] = sum(
        row[f"{name}_percentage"]
        for name in [
            "tree", "shrub", "grass",
            "wetland", "mangroves", "moss_lichen"
        ]
    )

    row["total_valid_landcover_pixels"] = total_pixels

    landcover_rows.append(row)

landcover_df = pd.DataFrame(landcover_rows)


# --------------------------------------------------
# 8. COMBINE ALL FEATURES
# --------------------------------------------------

print("Combining features...")

features = pd.concat(
    [
        roads.reset_index(drop=True).drop(columns="geometry"),
        elevation_df,
        slope_df,
        landcover_df
    ],
    axis=1
)

# Keep road length and OSM attributes where available.
# This is a road-level dataset, not yet the final
# road-segment x rainfall-event training dataset.


# --------------------------------------------------
# 9. SAVE CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "road_features.csv"

features.to_csv(csv_path, index=False)


# --------------------------------------------------
# 10. SAVE SPATIAL VERSION
# --------------------------------------------------

spatial_roads = roads.copy()

for column in [
    "elevation_mean_m",
    "elevation_min_m",
    "elevation_max_m",
    "slope_mean_deg",
    "slope_min_deg",
    "slope_max_deg",
    "builtup_percentage",
    "vegetation_percentage",
    "water_percentage"
]:
    if column in features.columns:
        spatial_roads[column] = features[column].values

gpkg_path = OUTPUT_DIR / "road_features.gpkg"
spatial_roads.to_file(gpkg_path, driver="GPKG")


# --------------------------------------------------
# 11. SUMMARY
# --------------------------------------------------

print("\n--------------------------------")
print("ROAD FEATURE EXTRACTION COMPLETE")
print("--------------------------------")

print("Road segments:", len(features))
print("Total columns:", len(features.columns))

print("\nMissing values in important features:")

important_columns = [
    "elevation_mean_m",
    "slope_mean_deg",
    "builtup_percentage",
    "vegetation_percentage",
    "water_percentage"
]

print(features[important_columns].isna().sum())

print("\nCSV saved:", csv_path)
print("GeoPackage saved:", gpkg_path)

print("\nFirst five rows:")
print(features.head())
