import pandas as pd
import geopandas as gpd
import pyogrio
from pathlib import Path

BASE = Path("data/processed/observations")

# 1. Inspect combined CSV
csv_file = BASE / "bbmp_flood_locations.csv"
df = pd.read_csv(csv_file)

print("\n========== BBMP CSV AUDIT ==========")

print("Total records:", len(df))
print("\nRecords by source:")
print(df["source_dataset"].value_counts())

print("\nGeometry types:")
print(df["geometry_type"].value_counts())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

# 2. Inspect GeoPackages
print("\n========== GEOPACKAGE AUDIT ==========")

for path in BASE.glob("*.gpkg"):

    print(f"\nFile: {path.name}")

    layers = pyogrio.list_layers(path)

    print("Layers:")
    print(layers)

    for layer in layers[:, 0]:

        gdf = gpd.read_file(
            path,
            layer=layer,
            engine="pyogrio"
        )

        print("Layer:", layer)
        print("Features:", len(gdf))
        print("CRS:", gdf.crs)
        print("Geometry types:",
              gdf.geometry.geom_type.value_counts().to_dict())

print("\nBBMP audit completed.")
