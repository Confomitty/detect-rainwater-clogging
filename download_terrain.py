import ee

# 1. Initialize Earth Engine
PROJECT_ID = "waterloggingbengaluru"
ee.Initialize(project=PROJECT_ID)

# 2. Define Silk Board location
silk_board = ee.Geometry.Point([77.623, 12.917])

# Approximately 4 km across
region = silk_board.buffer(2000)

# 3. Load Copernicus DEM
dem_collection = ee.ImageCollection(
    "COPERNICUS/DEM/GLO30_2024_1"
)

dem = dem_collection \
    .select("DEM") \
    .mosaic() \
    .clip(region)

# 4. Calculate terrain slope
slope = ee.Terrain.slope(dem)

# 5. Export elevation to Google Drive
dem_task = ee.batch.Export.image.toDrive(
    image=dem,
    description="SilkBoard_Elevation",
    folder="Waterlogging_Data",
    fileNamePrefix="silk_board_elevation",
    region=region,
    scale=30,
    crs="EPSG:32643",
    maxPixels=1e13
)

dem_task.start()

# 6. Export slope to Google Drive
slope_task = ee.batch.Export.image.toDrive(
    image=slope,
    description="SilkBoard_Slope",
    folder="Waterlogging_Data",
    fileNamePrefix="silk_board_slope",
    region=region,
    scale=30,
    crs="EPSG:32643",
    maxPixels=1e13
)

slope_task.start()

print("Elevation export started:", dem_task.id)
print("Slope export started:", slope_task.id)
