import geopandas as gpd
import pandas as pd

from pathlib import Path

# -----------------------------------------
# FILES
# -----------------------------------------

BBMP_FILE = "data/processed/observations/bbmp_all_locations.gpkg"
ROAD_FILE = "data/raw/roads/road_segments.geojson"
FEATURE_FILE = "data/processed/features/road_features.csv"

OUTPUT_DIR = Path("data/processed/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "road_features_with_bbmp.csv"
OUTPUT_GPKG = OUTPUT_DIR / "road_features_with_bbmp.gpkg"

TARGET_CRS = "EPSG:32643"


# -----------------------------------------
# LOAD DATA
# -----------------------------------------

print("Loading datasets...")

bbmp = gpd.read_file(BBMP_FILE)
roads = gpd.read_file(ROAD_FILE)

features = pd.read_csv(FEATURE_FILE)

roads = roads.to_crs(TARGET_CRS)
bbmp = bbmp.to_crs(TARGET_CRS)

if "road_id" not in roads.columns:
    raise ValueError("road_id missing from road network.")

if "road_id" not in features.columns:
    raise ValueError("road_id missing from road features.")

if "source_dataset" not in bbmp.columns:
    raise ValueError("source_dataset missing from BBMP data.")

if roads["road_id"].duplicated().any():
    raise ValueError(
        "Duplicate road IDs found. Resolve road identity "
        "before merging features."
    )

if features["road_id"].duplicated().any():
    raise ValueError(
        "Duplicate road IDs found in road feature CSV."
    )

print("Roads:", len(roads))
print("BBMP locations:", len(bbmp))


# -----------------------------------------
# CREATE STABLE ROW IDENTIFIERS
# -----------------------------------------

roads = roads.reset_index(drop=True).copy()
roads["road_row_id"] = range(len(roads))

bbmp = bbmp.reset_index(drop=True).copy()

if "bbmp_location_id" not in bbmp.columns:
    bbmp["bbmp_location_id"] = [
        f"BBMP_{i+1:03d}" for i in range(len(bbmp))
    ]


# -----------------------------------------
# FUNCTION: NEAREST BBMP LOCATION PER ROAD
# -----------------------------------------

def calculate_nearest(road_gdf, bbmp_gdf, prefix):

    joined = gpd.sjoin_nearest(
        road_gdf[["road_row_id", "geometry"]],
        bbmp_gdf[
            ["bbmp_location_id", "geometry"]
        ],
        how="left",
        distance_col="distance_m"
    )

    # In case of equal-distance ties, retain the
    # minimum distance and a deterministic location ID.
    result = (
        joined.groupby("road_row_id")
        .agg(
            distance_m=("distance_m", "min"),
            nearest_location_id=("bbmp_location_id", "min")
        )
        .reset_index()
    )

    result = result.rename(columns={
        "distance_m": f"{prefix}_distance_m",
        "nearest_location_id": f"{prefix}_nearest_location_id"
    })

    return result


# -----------------------------------------
# ALL BBMP LOCATIONS
# -----------------------------------------

print("\nCalculating distance to all BBMP locations...")

all_nearest = calculate_nearest(
    roads,
    bbmp,
    "bbmp"
)

# -----------------------------------------
# EACH BBMP SOURCE SEPARATELY
# -----------------------------------------

source_groups = {
    "vulnerable_flood_locations":
        "bbmp_vulnerable",

    "flood_prone_locations":
        "bbmp_flood_prone",

    "bbmp_low_lying_areas":
        "bbmp_low_lying"
}

results = all_nearest

for source_name, prefix in source_groups.items():

    subset = bbmp[
        bbmp["source_dataset"] == source_name
    ].copy()

    print(
        f"\nProcessing {source_name}:",
        len(subset),
        "locations"
    )

    if subset.empty:
        print("No records found. Skipping.")
        continue

    nearest = calculate_nearest(
        roads,
        subset,
        prefix
    )

    results = results.merge(
        nearest,
        on="road_row_id",
        how="left",
        validate="one_to_one"
    )


# -----------------------------------------
# PROXIMITY INDICATORS
# -----------------------------------------

distance_columns = [
    column for column in results.columns
    if column.endswith("_distance_m")
]

for column in distance_columns:

    prefix = column.removesuffix("_distance_m")

    results[f"{prefix}_within_50m"] = (
        results[column] <= 50
    ).astype(int)

    results[f"{prefix}_within_100m"] = (
        results[column] <= 100
    ).astype(int)


# -----------------------------------------
# MERGE WITH EXISTING ROAD FEATURES
# -----------------------------------------

print("\nCombining with road-level features...")

road_lookup = roads[
    ["road_row_id", "road_id"]
]

results = results.merge(
    road_lookup,
    on="road_row_id",
    how="left",
    validate="one_to_one"
)

final = features.merge(
    results.drop(columns="road_row_id"),
    on="road_id",
    how="left",
    validate="one_to_one"
)


# -----------------------------------------
# SAVE CSV
# -----------------------------------------

final.to_csv(OUTPUT_CSV, index=False)


# -----------------------------------------
# SAVE SPATIAL OUTPUT
# -----------------------------------------

spatial = roads.merge(
    results.drop(columns="road_row_id"),
    on="road_id",
    how="left",
    validate="one_to_one"
)

spatial.to_file(
    OUTPUT_GPKG,
    driver="GPKG",
    engine="pyogrio"
)


# -----------------------------------------
# SUMMARY
# -----------------------------------------

print("\n================================")
print("BBMP ROAD FEATURES COMPLETE")
print("================================")

print("Road records:", len(final))
print("Columns:", len(final.columns))

print("\nProximity feature summary:")

for column in distance_columns:

    print(
        column,
        "| within 50m:",
        (results[column] <= 50).sum(),
        "| within 100m:",
        (results[column] <= 100).sum()
    )

print("\nCSV:", OUTPUT_CSV)
print("GeoPackage:", OUTPUT_GPKG)

print("\nFirst five records:")
print(final.head())
