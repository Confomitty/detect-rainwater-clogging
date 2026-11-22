import osmnx as ox
import pandas as pd

from pathlib import Path


def download_roads(latitude, longitude, radius, output_dir):
    """
    Download the drivable road network for a selected study area.

    Parameters
    ----------
    latitude : float
        Study area centre latitude.
    longitude : float
        Study area centre longitude.
    radius : int
        Study area radius in metres.
    output_dir : Path or str
        Area-specific output directory.
    """

    CENTER = (latitude, longitude)

    output_dir = Path(output_dir) / "roads"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading road network around {CENTER}...")
    print(f"Study radius: {radius} metres")

    # Download drivable roads
    G = ox.graph_from_point(
        CENTER,
        dist=radius,
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

    # Give each road segment an ID within this download
    edges = edges.reset_index()

    edges["road_id"] = (
        edges["u"].astype(str) + "_" +
        edges["v"].astype(str) + "_" +
        edges["key"].astype(str)
    )

    # Save spatial road network
    geojson_path = output_dir / "road_segments.geojson"
    edges.to_file(geojson_path, driver="GeoJSON")

    # Save non-spatial road inventory CSV
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

    return edges
