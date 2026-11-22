import geopandas as gpd
import pandas as pd
import fiona
import json

from pathlib import Path
from shapely.geometry import Point


def process_bbmp_kml(
    latitude,
    longitude,
    radius,
    common_dir,
    processed_dir,
    area_name
):
    """
    Process Bengaluru-wide BBMP KML datasets and filter
    locations within the selected study area's radius.

    Parameters
    ----------
    latitude : float
        Study area centre latitude.
    longitude : float
        Study area centre longitude.
    radius : int
        Study area radius in metres.
    common_dir : Path or str
        Shared data directory containing the BBMP KML files.
    processed_dir : Path or str
        Area-specific processed data directory.
    area_name : str
        Selected study area name.
    """

    # ==================================================
    # 1. CONFIGURATION
    # ==================================================

    # Enable KML / LIBKML reading in Fiona
    fiona.drvsupport.supported_drivers["LIBKML"] = "r"
    fiona.drvsupport.supported_drivers["KML"] = "r"

    print("Enabled KML and LIBKML drivers.")

    INPUT_DIR = Path(common_dir) / "bbmp"

    OUTPUT_DIR = Path(processed_dir) / "observations"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    KML_FILES = {
        "vulnerable_flood_locations": "vulnerable_flood_locations.kml",
        "flood_prone_locations": "flood_prone_locations.kml",
        "bbmp_low_lying_areas": "bbmp_low_lying_areas.kml"
    }

    TARGET_CRS = "EPSG:32643"

    # ==================================================
    # 2. CREATE STUDY AREA
    # ==================================================

    center = gpd.GeoSeries(
        [Point(longitude, latitude)],
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS).iloc[0]

    study_area = center.buffer(radius)

    print(f"\nBBMP processing for: {area_name}")
    print("Study area created.")
    print("Coordinates:", latitude, longitude)
    print("Radius:", radius, "metres")

    # ==================================================
    # 3. PROCESS EACH KML FILE
    # ==================================================

    all_records = []
    summary_records = []

    for dataset_name, filename in KML_FILES.items():

        filepath = INPUT_DIR / filename

        print("\n-----------------------------------")
        print("Processing:", filename)
        print("-----------------------------------")

        if not filepath.exists():
            print("FILE NOT FOUND:", filepath)

            summary_records.append({
                "dataset": dataset_name,
                "source_file": filename,
                "total_features": 0,
                "features_in_study_area": 0
            })

            continue

        layers = fiona.listlayers(str(filepath))

        print("Layers found:", layers)

        dataset_total = 0
        dataset_relevant = 0

        relevant_layers = []

        for layer_name in layers:

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

            # Filter to selected study area
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

            relevant_layers.append(relevant)

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

        # ------------------------------------------
        # SAVE EACH BBMP SOURCE AS A GEOPACKAGE
        # ------------------------------------------

        if relevant_layers:

            source_gdf = gpd.GeoDataFrame(
                pd.concat(
                    relevant_layers,
                    ignore_index=True
                ),
                geometry="geometry",
                crs=TARGET_CRS
            )

            gpkg_path = OUTPUT_DIR / f"{dataset_name}.gpkg"

            if gpkg_path.exists():
                gpkg_path.unlink()

            source_gdf.to_file(
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
        combined = pd.DataFrame()
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
    print(f"BBMP KML PROCESSING COMPLETE: {area_name}")
    print("===================================")

    print("Total relevant records:", len(combined))
    print("Output directory:", OUTPUT_DIR)

    return combined, summary


