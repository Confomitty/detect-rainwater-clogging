import rasterio
import geopandas as gpd
import numpy as np

# Check land cover raster
with rasterio.open("data/raw/landcover/landcover.tif") as src:
    landcover = src.read(1)

    print("\nLAND COVER")
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Dimensions:", src.width, "x", src.height)
    print("Classes found:", np.unique(landcover))

# Check road network
roads = gpd.read_file("data/raw/roads/road_segments.geojson")

print("\nROAD NETWORK")
print("Number of road segments:", len(roads))
print("CRS:", roads.crs)
print("Columns:", roads.columns.tolist())

print("\nFirst five road records:")
print(roads[["road_id", "highway", "length"]].head())
