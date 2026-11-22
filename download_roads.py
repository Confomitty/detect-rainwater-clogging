
import osmnx as ox
import pandas as pd

from pathlib import Path
import time


def download_roads(latitude, longitude, radius, output_dir):
    """
    Download the drivable road network for a selected study area.

    Uses sequential retries and alternative Overpass endpoints
    to handle temporary server connection failures.
    """

    CENTER = (latitude, longitude)

    output_dir = Path(output_dir) / "roads"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading road network around {CENTER}...")
    print(f"Study radius: {radius} metres")

    # Configure OSMnx request settings
    ox.settings.requests_timeout = 300
    ox.settings.overpass_rate_limit = True

    # Try public Overpass endpoints sequentially
    endpoints = [
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api"
    ]

    G = None
    last_error = None

    for endpoint in endpoints:
        ox.settings.overpass_url = endpoint

        print(f"\nTrying Overpass endpoint: {endpoint}")

        for attempt in range(1, 3):
            try:
                print(f"Attempt {attempt}/2...")

                G = ox.graph_from_point(
                    CENTER,
                    dist=radius,
                    network_type="drive",
                    simplify=True,
                    retain_all=True
                )

                print("Road network downloaded successfully!")
                break

            except Exception as error:
                last_error = error

                print(f"Attempt {attempt} failed: {error}")

                if attempt < 2:
                    print("Waiting 20 seconds before retry...")
                    time.sleep(20)

        if G is not None:
            break

        print("This endpoint could not be reached.")

    if G is None:
        raise RuntimeError(
            "Unable to download the road network from the available "
            "Overpass endpoints. Please try again later."
        ) from last_error

    # Convert network edges into individual road segments
    nodes, edges = ox.graph_to_gdfs(
        G,
        nodes=True,
        edges=True
    )

    edges = edges.reset_index()

    # Create road IDs
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

    for column in tabular.columns:
        tabular[column] = tabular[column].apply(
            lambda value: ";".join(map(str, value))
            if isinstance(value, list) else value
        )

    tabular.to_csv(csv_path, index=False)

    print("\n--------------------------------")
    print("ROAD NETWORK DOWNLOAD COMPLETE")
    print("--------------------------------")
    print("Number of road segments:", len(edges))
    print("GeoJSON:", geojson_path)
    print("CSV:", csv_path)
    print("CRS:", edges.crs)

    return edges
