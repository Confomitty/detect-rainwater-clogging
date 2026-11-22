import geopandas as gpd
import pandas as pd
import fiona
import json
import re

# Enable KML / LIBKML reading in Fiona
fiona.drvsupport.supported_drivers["LIBKML"] = "r"
fiona.drvsupport.supported_drivers["KML"] = "r"

print("Enabled KML and LIBKML drivers.")

from pathlib import Path
from shapely.geometry import Point


# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_DIR = Path("data/raw/observations/spatial_reference")

OUTPUT_DIR = Path("data/processed/observations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KML_FILES = {
    "vulnerable_flood_locations": "vulnerable_flood_locations.kml",
    "flood_prone_locations": "flood_prone_locations.kml",
    "bbmp_low_lying_areas": "bbmp_low_lying_areas.kml"
}

# Approximate Silk Board study area
CENTER_LAT = 12.917
CENTER_LON = 77.623
RADIUS_METERS = 2000

TARGET_CRS = "EPSG:32643"


# ==================================================
# 2. CREATE STUDY AREA
# ==================================================

center = gpd.GeoSeries(
    [Point(CENTER_LON, CENTER_LAT)],
    crs="EPSG:4326"
).to_crs(TARGET_CRS).iloc[0]

study_area = center.buffer(RADIUS_METERS)

print("\nSilk Board study area created.")
print("Radius:", RADIUS_METERS, "metres")


# ==================================================
# 3. PROCESS EACH KML FILE
# ==================================================

all_records = []
summary_records = []

# Each BBMP source will have its own GeoPackage.
# This avoids Fiona append-layer issues.

for dataset_name, filename in KML_FILES.items():

    filepath = INPUT_DIR / filename

    print("\n-----------------------------------")
    print("Processing:", filename)
    print("-----------------------------------")

    if not filepath.exists():
        print("FILE NOT FOUND:", filepath)
        continue

    # Find available layers inside KML
    layers = fiona.listlayers(str(filepath))

    print("Layers found:", layers)

    dataset_total = 0
    dataset_relevant = 0

    for layer_index, layer_name in enumerate(layers):

        print("\nReading layer:", layer_name)

        gdf = gpd.read_file(
            filepath,
            layer=layer_name,
            engine="fiona"
        )

        if gdf.empty:
            print("Empty layer. Skipping.")
            continue

        original_count = len(gdf)

        # KML coordinates normally use WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        # Reproject to metres
        gdf = gdf.to_crs(TARGET_CRS)

        # Remove records without geometry
        gdf = gdf[
            gdf.geometry.notna() &
            ~gdf.geometry.is_empty
        ].copy()

        # Keep only locations intersecting our study area
        relevant = gdf[
            gdf.geometry.intersects(study_area)
        ].copy()

        dataset_total += original_count
        dataset_relevant += len(relevant)

        print("Original features:", original_count)
        print("Features in study area:", len(relevant))

        if relevant.empty:
            continue

        # Preserve source information
        relevant["source_dataset"] = dataset_name
        relevant["source_file"] = filename
        relevant["source_layer"] = str(layer_name)

        # Generate geographic metadata
        centroids = relevant.geometry.centroid

        centroid_gdf = gpd.GeoSeries(
            centroids,
            crs=TARGET_CRS
        ).to_crs("EPSG:4326")

        relevant["centroid_longitude"] = centroid_gdf.x.values
        relevant["centroid_latitude"] = centroid_gdf.y.values

        relevant["geometry_type"] = (
            relevant.geometry.geom_type.values
        )

        # Prepare CSV records
        table = relevant.drop(columns="geometry").copy()

        table["geometry_wkt"] = relevant.geometry.to_wkt().values

        # Convert complex values to CSV-friendly strings
        for column in table.columns:
            table[column] = table[column].apply(
                lambda value: json.dumps(value)
                if isinstance(value, (dict, list))
                else value
            )

        all_records.append(table)

# Save each source separately using Pyogrio

gpkg_path = OUTPUT_DIR / f"{dataset_name}.gpkg"

# Remove previous output when rerunning
if gpkg_path.exists():
    gpkg_path.unlink()

relevant.to_file(
    gpkg_path,
    driver="GPKG",
    engine="pyogrio"
)

print("Spatial output saved:", gpkg_path)

summary_records.append({
    "dataset": dataset_name,
    "source_file": filename,
    "total_features": dataset_total,
    "features_in_study_area": dataset_relevant
})


# ==================================================
# 4. SAVE COMBINED CSV
# ==================================================

if all_records:

    combined = pd.concat(
        all_records,
        ignore_index=True,
        sort=False
    )

    csv_path = OUTPUT_DIR / "bbmp_flood_locations.csv"

    combined.to_csv(csv_path, index=False)

    print("\nCombined CSV saved:", csv_path)

else:
    print("\nNo relevant features found in the supplied KML files.")


# ==================================================
# 5. SAVE DATASET SUMMARY
# ==================================================

summary = pd.DataFrame(summary_records)

summary_path = OUTPUT_DIR / "bbmp_dataset_summary.csv"

summary.to_csv(summary_path, index=False)

print("\nDataset summary:")
print(summary.to_string(index=False))

print("\nSummary saved:", summary_path)


# ==================================================
# 6. FINAL REPORT
# ==================================================

print("\n===================================")
print("BBMP KML PROCESSING COMPLETE")
print("===================================")

if all_records:
    print("Total relevant records:", len(combined))

print("GeoPackages saved separately for each BBMP source.")
print("Summary:", summary_path)
