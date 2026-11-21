import osmnx as ox
from pathlib import Path

# Study area
CENTER = (12.917, 77.623)
DISTANCE_METERS = 2000

output_dir = Path("data/raw/roads")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading road network around Silk Board...")

# Download drivable roads
G = ox.graph_from_point(
    CENTER,
    dist=DISTANCE_METERS,
    network_type="drive",
    simplify=True,
    retain_all=True
)

# Convert network edges into individual road segments
nodes, edges = ox.graph_to_gdfs(
    G,
    nodes=True,
    edges=True
)

# Give each road segment a stable ID within this download
edges = edges.reset_index()
edges["road_id"] = (
    edges["u"].astype(str) + "_" +
    edges["v"].astype(str) + "_" +
    edges["key"].astype(str)
)

# Save spatial road network
geojson_path = output_dir / "road_segments.geojson"
edges.to_file(geojson_path, driver="GeoJSON")

# Save a non-spatial road inventory CSV
csv_path = output_dir / "road_inventory.csv"

tabular = edges.drop(columns="geometry").copy()

# Convert list-valued attributes into CSV-friendly strings
for column in tabular.columns:
    tabular[column] = tabular[column].apply(
        lambda value: ";".join(map(str, value))
        if isinstance(value, list) else value
    )

tabular.to_csv(csv_path, index=False)

print("\nRoad network downloaded successfully!")
print("Number of road segments:", len(edges))
print("GeoJSON:", geojson_path)
print("CSV:", csv_path)
print("CRS:", edges.crs)
