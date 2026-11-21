import rasterio
import matplotlib.pyplot as plt
import numpy as np

files = {
    "Elevation": "data/raw/terrain/elevation.tif",
    "Slope": "data/raw/terrain/slope.tif"
}

for name, filepath in files.items():

    with rasterio.open(filepath) as src:

        data = src.read(1).astype(float)

        # Ignore NoData pixels
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        print(f"\n{name} information")
        print("-------------------------")
        print("Width:", src.width)
        print("Height:", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Minimum:", np.nanmin(data))
        print("Maximum:", np.nanmax(data))
        print("Mean:", np.nanmean(data))

        plt.figure(figsize=(9, 7))
        plt.imshow(data, cmap="terrain")
        plt.colorbar(label="Elevation (metres)" if name == "Elevation"
                     else "Slope (degrees)")
        plt.title(f"Silk Board - {name}")
        plt.xlabel("Raster column")
        plt.ylabel("Raster row")
        plt.show()