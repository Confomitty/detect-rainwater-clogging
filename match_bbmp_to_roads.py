import geopandas as gpd
import pandas as pd

from pathlib import Path


def match_bbmp_to_roads(raw_dir, processed_dir, area_name):
    """
    Match BBMP vulnerability locations to their nearest
    road segments in the selected study area.

    Parameters
    ----------
    raw_dir : Path or str
        Area-specific raw data directory.
    processed_dir : Path or str
        Area-specific processed data directory.
    area_name : str
        Selected study area name.
    """

    # -----------------------------------------
    # CONFIGURATION
    # -----------------------------------------

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    BASE = processed_dir / "observations"

    BBMP_FILE = BASE / "bbmp_all_locations.gpkg"
    ROAD_FILE = raw_dir / "roads" / "road_segments.geojson"

    OUTPUT_DIR = processed_dir / "features"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TARGET_CRS = "EPSG:32643"
    MAX_DISTANCE_METERS = 500

    # -----------------------------------------
    # CHECK INPUT FILES
    # -----------------------------------------

    for filepath in [BBMP_FILE, ROAD_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required input file not found: {filepath}"
            )

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------

    print(f"Loading BBMP locations and roads for {area_name}...")

    bbmp = gpd.read_file(BBMP_FILE)
    roads = gpd.read_file(ROAD_FILE)

    print("BBMP locations:", len(bbmp))
    print("Road segments:", len(roads))

    if bbmp.empty:
        raise ValueError("BBMP dataset is empty.")

    if roads.empty:
        raise ValueError("Road network is empty.")

    # -----------------------------------------
    # REPROJECT BOTH DATASETS
    # -----------------------------------------

    bbmp = bbmp.to_crs(TARGET_CRS)
    roads = roads.to_crs(TARGET_CRS)

    if "road_id" not in roads.columns:
        raise ValueError("road_id column missing from road network.")

    if "source_dataset" not in bbmp.columns:
        raise ValueError("source_dataset column missing from BBMP data.")

    # -----------------------------------------
    # CREATE UNIQUE BBMP IDENTIFIERS
    # -----------------------------------------

    bbmp = bbmp.reset_index(drop=True)

    bbmp["bbmp_location_id"] = [
        f"BBMP_{i+1:03d}" for i in range(len(bbmp))
    ]

    # -----------------------------------------
    # FIND NEAREST ROAD FOR EACH BBMP POINT
    # -----------------------------------------

    print("Finding nearest road segments...")

    matches = gpd.sjoin_nearest(
        bbmp,
        roads[["road_id", "highway", "length", "geometry"]],
        how="left",
        max_distance=MAX_DISTANCE_METERS,
        distance_col="distance_to_road_m"
    )

    # Keep relevant fields
    result = pd.DataFrame(
        matches.drop(columns="geometry")
    )

    result = result.rename(columns={
        "road_id": "nearest_road_id",
        "highway": "nearest_road_type",
        "length": "nearest_road_length_m"
    })

    # Identify points without a nearby road
    result["road_match_status"] = result[
        "nearest_road_id"
    ].notna().map({
        True: "matched",
        False: f"no_road_within_{MAX_DISTANCE_METERS}m"
    })

    # -----------------------------------------
    # SAVE RESULTS
    # -----------------------------------------

    csv_path = OUTPUT_DIR / "bbmp_road_proximity.csv"

    result.to_csv(csv_path, index=False)

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\nBBMP-ROAD MATCHING COMPLETE")
    print("--------------------------------")

    print("Area:", area_name)
    print("BBMP locations:", len(bbmp))

    print(
        "Matched locations:",
        (result["road_match_status"] == "matched").sum()
    )

    print(
        "Unmatched locations:",
        (result["road_match_status"] != "matched").sum()
    )

    print("\nResults:")

    print(
        result[
            [
                "bbmp_location_id",
                "source_dataset",
                "nearest_road_id",
                "distance_to_road_m",
                "road_match_status"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", csv_path)

    return result


