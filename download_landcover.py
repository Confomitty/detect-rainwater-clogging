import ee
import requests
from pathlib import Path

PROJECT_ID = "waterloggingbengaluru"

ee.Initialize(project=PROJECT_ID)

# Study area: approximately 2 km around Silk Board
center = ee.Geometry.Point([77.623, 12.917])
region = center.buffer(2000)

output_dir = Path("data/raw/landcover")
output_dir.mkdir(parents=True, exist_ok=True)

# ESA WorldCover 2021, 10 m resolution
landcover = (
    ee.ImageCollection("ESA/WorldCover/v200")
    .first()
    .select("Map")
    .clip(region)
)

print("Preparing WorldCover download...")

url = landcover.getDownloadURL({
    "scale": 10,
    "crs": "EPSG:32643",
    "region": region,
    "format": "GEO_TIFF"
})

response = requests.get(url, timeout=120)
response.raise_for_status()

filepath = output_dir / "landcover.tif"
filepath.write_bytes(response.content)

print("Saved:", filepath)
print("Download size:", len(response.content), "bytes")
