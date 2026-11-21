import ee
import requests
from pathlib import Path

# Use the same Google Cloud Project ID as before
PROJECT_ID = "waterloggingbengaluru"

ee.Initialize(project=PROJECT_ID)

# Create output folder
output_dir = Path("data/raw/terrain")
output_dir.mkdir(parents=True, exist_ok=True)

# Same approximate Silk Board study area
silk_board = ee.Geometry.Point([77.623, 12.917])
region = silk_board.buffer(2000)

# Recreate the exported images
dem = (
    ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
    .select("DEM")
    .mosaic()
    .clip(region)
)

slope = ee.Terrain.slope(dem)

# Download both images directly as GeoTIFF
images = {
    "elevation.tif": dem,
    "slope.tif": slope
}

for filename, image in images.items():

    print(f"Preparing {filename}...")

    url = image.getDownloadURL({
        "scale": 30,
        "crs": "EPSG:32643",
        "region": region,
        "format": "GEO_TIFF"
    })

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    filepath = output_dir / filename
    filepath.write_bytes(response.content)

    print(f"Downloaded: {filepath}")

print("Both raster files downloaded successfully!")