import ee


def download_terrain(latitude, longitude, radius, area_name):
    """
    Export elevation and slope rasters for the selected study area
    from Google Earth Engine to Google Drive.
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

    print(f"Preparing terrain exports for {area_name}...")
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"Radius: {radius} metres")

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

    # Calculate terrain slope
    slope = ee.Terrain.slope(dem)

    # Create area-specific export names
    area_prefix = area_name.lower().replace(" ", "_")

    # Export elevation
    dem_task = ee.batch.Export.image.toDrive(
        image=dem,
        description=f"{area_prefix}_Elevation",
        folder="Waterlogging_Data",
        fileNamePrefix=f"{area_prefix}_elevation",
        region=region,
        scale=30,
        crs="EPSG:32643",
        maxPixels=1e13
    )

    dem_task.start()

    # Export slope
    slope_task = ee.batch.Export.image.toDrive(
        image=slope,
        description=f"{area_prefix}_Slope",
        folder="Waterlogging_Data",
        fileNamePrefix=f"{area_prefix}_slope",
        region=region,
        scale=30,
        crs="EPSG:32643",
        maxPixels=1e13
    )

    slope_task.start()

    print("\nTerrain exports started successfully!")
    print("Elevation task ID:", dem_task.id)
    print("Slope task ID:", slope_task.id)

    return dem_task, slope_task
