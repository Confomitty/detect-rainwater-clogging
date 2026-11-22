import ee
import requests

from pathlib import Path


def download_landcover(latitude, longitude, radius, output_dir):
    """
    Download ESA WorldCover data for the selected study area.

    Parameters
    ----------
    latitude : float
        Study area centre latitude.
    longitude : float
        Study area centre longitude.
    radius : int
        Study area radius in metres.
    output_dir : Path or str
        Area-specific raw data directory.
    """

    PROJECT_ID = "waterloggingbengaluru"

    # Initialize Earth Engine
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

    # Define selected study area
    center = ee.Geometry.Point([longitude, latitude])
    region = center.buffer(radius)

    # Create area-specific output directory
    output_dir = Path(output_dir) / "landcover"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ESA WorldCover 2021, 10 m resolution
    landcover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .clip(region)
    )

    print(f"Preparing WorldCover download for {latitude}, {longitude}...")
    print(f"Study radius: {radius} metres")

    # Generate download URL
    url = landcover.getDownloadURL({
        "scale": 10,
        "crs": "EPSG:32643",
        "region": region,
        "format": "GEO_TIFF"
    })

    # Download raster
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    filepath = output_dir / "landcover.tif"
    filepath.write_bytes(response.content)

    print("\nLand-cover download completed!")
    print("Saved:", filepath)
    print("Download size:", len(response.content), "bytes")

    return filepath
