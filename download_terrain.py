
import ee
import requests
from pathlib import Path


def download_terrain(latitude, longitude, radius, output_dir):
    """
    Download elevation and slope GeoTIFFs from Google Earth Engine
    directly into the selected study area's raw/terrain directory.
    """

    PROJECT_ID = "waterloggingbengaluru"

    # Initialize Earth Engine
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

    # Create output directory
    terrain_dir = Path(output_dir) / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    # Define study area
    center = ee.Geometry.Point([longitude, latitude])
    region = center.buffer(radius)

    print("Preparing terrain downloads...")
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"Radius: {radius} metres")
    print(f"Output directory: {terrain_dir}")

    # Load Copernicus DEM
    dem_collection = ee.ImageCollection(
        "COPERNICUS/DEM/GLO30_2024_1"
    )

    dem = (
        dem_collection
        .select("DEM")
        .mosaic()
        .clip(region)
    )

    # Calculate slope
    slope = ee.Terrain.slope(dem)

    # Prepare export parameters
    export_params = {
        "region": region,
        "scale": 30,
        "crs": "EPSG:32643",
        "format": "GEO_TIFF"
    }

    # Download elevation
    print("\nDownloading elevation raster...")

    elevation_url = dem.getDownloadURL(export_params)
    elevation_path = terrain_dir / "elevation.tif"

    response = requests.get(elevation_url, timeout=180)
    response.raise_for_status()

    elevation_path.write_bytes(response.content)

    print("Elevation saved:", elevation_path)

    # Download slope
    print("\nDownloading slope raster...")

    slope_url = slope.getDownloadURL(export_params)
    slope_path = terrain_dir / "slope.tif"

    response = requests.get(slope_url, timeout=180)
    response.raise_for_status()

    slope_path.write_bytes(response.content)

    print("Slope saved:", slope_path)

    print("\nTerrain downloads completed successfully!")

    return elevation_path, slope_path
