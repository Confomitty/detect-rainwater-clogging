import pandas as pd
import geopandas as gpd

from pathlib import Path

BASE = Path("data/processed/observations")

csv_path = BASE / "bbmp_flood_locations.csv"
gpkg_path = BASE / "bbmp_all_locations.gpkg"

# Read the combined CSV
df = pd.read_csv(csv_path)

# Reconstruct geometry from saved WKT
geometry = gpd.GeoSeries.from_wkt(df["geometry_wkt"])

gdf = gpd.GeoDataFrame(
    df.drop(columns=["geometry_wkt"]),
    geometry=geometry,
    crs="EPSG:32643"
)

# Save all 11 features into one spatial file
gdf.to_file(
    gpkg_path,
    driver="GPKG",
    engine="pyogrio"
)

print("Combined BBMP GeoPackage created!")
print("Features:", len(gdf))
print("CRS:", gdf.crs)
print("Sources:")
print(gdf["source_dataset"].value_counts())

print("Saved:", gpkg_path)
