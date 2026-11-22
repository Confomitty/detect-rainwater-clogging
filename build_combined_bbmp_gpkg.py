import pandas as pd
import geopandas as gpd

from pathlib import Path


def build_combined_bbmp_gpkg(processed_dir, area_name):
    """
    Build a combined spatial GeoPackage from the processed
    BBMP CSV for the selected study area.

    Parameters
    ----------
    processed_dir : Path or str
        Area-specific processed data directory.
    area_name : str
        Selected study area name.
    """

    # -----------------------------------------
    # CONFIGURATION
    # -----------------------------------------

    BASE = Path(processed_dir) / "observations"

    csv_path = BASE / "bbmp_flood_locations.csv"
    gpkg_path = BASE / "bbmp_all_locations.gpkg"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"BBMP CSV not found: {csv_path}"
        )

    print(f"Building combined BBMP GeoPackage for {area_name}...")

    # -----------------------------------------
    # READ COMBINED CSV
    # -----------------------------------------

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"No BBMP records available for {area_name}."
        )

    if "geometry_wkt" not in df.columns:
        raise ValueError(
            "The BBMP CSV does not contain geometry_wkt."
        )

    # -----------------------------------------
    # RECONSTRUCT GEOMETRY
    # -----------------------------------------

    geometry = gpd.GeoSeries.from_wkt(
        df["geometry_wkt"],
        crs="EPSG:32643"
    )

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs="EPSG:32643"
    )

    # Remove invalid or missing geometries
    gdf = gdf[
        gdf.geometry.notna() &
        ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError(
            "No valid BBMP geometries found."
        )

    # -----------------------------------------
    # SAVE GEOPACKAGE
    # -----------------------------------------

    if gpkg_path.exists():
        gpkg_path.unlink()

    gdf.to_file(
        gpkg_path,
        driver="GPKG",
        engine="pyogrio"
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\nCombined BBMP GeoPackage created!")
    print("Area:", area_name)
    print("Features:", len(gdf))
    print("CRS:", gdf.crs)

    print("\nSources:")
    print(gdf["source_dataset"].value_counts())

    print("\nSaved:", gpkg_path)

    return gdf


