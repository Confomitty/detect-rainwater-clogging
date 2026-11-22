import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

from rasterstats import zonal_stats
from pathlib import Path


def extract_road_features(raw_dir, processed_dir, area_name):
    """
    Extract elevation, slope and land-cover features
    for road segments in the selected study area.

    Parameters
    ----------
    raw_dir : Path or str
        Area-specific raw data directory.
    processed_dir : Path or str
        Area-specific processed data directory.
    area_name : str
        Selected study area name.
    """

    # --------------------------------------------------
    # 1. FILE PATHS
    # --------------------------------------------------

    raw_dir = Path(raw_dir)
    OUTPUT_DIR = Path(processed_dir) / "features"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ROAD_FILE = raw_dir / "roads" / "road_segments.geojson"

    ELEVATION_FILE = raw_dir / "terrain" / "elevation.tif"
    SLOPE_FILE = raw_dir / "terrain" / "slope.tif"
    LANDCOVER_FILE = raw_dir / "landcover" / "landcover.tif"

    required_files = [
        ROAD_FILE,
        ELEVATION_FILE,
        SLOPE_FILE,
        LANDCOVER_FILE
    ]

    for filepath in required_files:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required input file not found: {filepath}"
            )

    print(f"\nExtracting road features for {area_name}...")

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

    if target_crs is None:
        raise ValueError("Elevation raster has no CRS.")

    roads = roads.to_crs(target_crs)

    print("Reprojected roads to:", target_crs)

    # --------------------------------------------------
    # 4. CREATE ROAD BUFFERS
    # --------------------------------------------------

    # A 10-metre buffer on either side of each road.
    road_buffers = roads.geometry.buffer(10)

    print("Created road buffers.")

    # --------------------------------------------------
    # 5. EXTRACT ELEVATION FEATURES
    # --------------------------------------------------

    print("Extracting elevation features...")

    elevation_stats = zonal_stats(
        road_buffers,
        str(ELEVATION_FILE),
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
        str(SLOPE_FILE),
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
        str(LANDCOVER_FILE),
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

    spatial_roads.to_file(
        gpkg_path,
        driver="GPKG"
    )

    # --------------------------------------------------
    # 11. SUMMARY
    # --------------------------------------------------

    print("\n--------------------------------")
    print(f"ROAD FEATURE EXTRACTION: {area_name}")
    print("--------------------------------")

    print("Road segments:", len(features))
    print("Total columns:", len(features.columns))

    important_columns = [
        "elevation_mean_m",
        "slope_mean_deg",
        "builtup_percentage",
        "vegetation_percentage",
        "water_percentage"
    ]

    print("\nMissing values in important features:")
    print(features[important_columns].isna().sum())

    print("\nCSV saved:", csv_path)
    print("GeoPackage saved:", gpkg_path)

    print("\nFirst five rows:")
    print(features.head())

    return features, spatial_roads


