import ee

PROJECT_ID = "waterloggingbengaluru"

# Connect to Google Earth Engine
ee.Initialize(project=PROJECT_ID)

print("Earth Engine connected!")

# Test Copernicus DEM (This is dataset1)
dem = ee.ImageCollection(
    "COPERNICUS/DEM/GLO30_2024_1"
)

print("New DEM images:", dem.size().getInfo())

# Test ESA WorldCover (this is dataset2)
worldcover = ee.ImageCollection(
    "ESA/WorldCover/v200"
)

print("WorldCover images:", worldcover.size().getInfo())

print("Both datasets are accessible!")
