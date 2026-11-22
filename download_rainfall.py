import requests
import pandas as pd
import json

from pathlib import Path
from datetime import datetime, timezone


def download_rainfall(latitude, longitude, output_dir, area_name):
    """
    Download historical rainfall and soil moisture data
    for a selected study area using Open-Meteo ERA5.

    Parameters
    ----------
    latitude : float
        Study area centre latitude.
    longitude : float
        Study area centre longitude.
    output_dir : Path or str
        Area-specific raw data directory.
    area_name : str
        Name of the selected study area.
    """

    # -----------------------------------------
    # CONFIGURATION
    # -----------------------------------------

    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"

    URL = "https://archive-api.open-meteo.com/v1/archive"

    OUTPUT_DIR = Path(output_dir) / "rainfall"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # REQUEST HISTORICAL WEATHER
    # -----------------------------------------

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "models": "era5",

        "hourly": ",".join([
            "precipitation",
            "rain",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm"
        ]),

        "timezone": "Asia/Kolkata",
        "precipitation_unit": "mm"
    }

    print(f"Requesting historical weather data for {area_name}...")
    print("Coordinates:", latitude, longitude)
    print("Period:", START_DATE, "to", END_DATE)

    response = requests.get(
        URL,
        params=params,
        timeout=180
    )

    response.raise_for_status()

    weather = response.json()

    if "hourly" not in weather:
        raise ValueError(f"Unexpected API response: {weather}")

    # -----------------------------------------
    # CONVERT RESPONSE INTO DATAFRAME
    # -----------------------------------------

    df = pd.DataFrame(weather["hourly"])

    df.rename(columns={
        "time": "timestamp",
        "precipitation": "precipitation_mm",
        "rain": "rain_mm"
    }, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # Rolling rainfall accumulations
    df["rainfall_3h_mm"] = (
        df["precipitation_mm"]
        .rolling(window=3, min_periods=3)
        .sum()
    )

    df["rainfall_6h_mm"] = (
        df["precipitation_mm"]
        .rolling(window=6, min_periods=6)
        .sum()
    )

    df["rainfall_24h_mm"] = (
        df["precipitation_mm"]
        .rolling(window=24, min_periods=24)
        .sum()
    )

    # -----------------------------------------
    # SAVE CSV
    # -----------------------------------------

    csv_path = OUTPUT_DIR / "rainfall_hourly.csv"

    df.to_csv(csv_path, index=False)

    # -----------------------------------------
    # SAVE SOURCE METADATA
    # -----------------------------------------

    metadata = {
        "area": area_name,
        "source": "Open-Meteo Historical Weather API",
        "endpoint": URL,
        "model": "era5",
        "latitude_requested": latitude,
        "longitude_requested": longitude,
        "latitude_returned": weather.get("latitude"),
        "longitude_returned": weather.get("longitude"),
        "timezone": weather.get("timezone"),
        "period_start": START_DATE,
        "period_end": END_DATE,
        "hourly_variables": list(weather["hourly"].keys()),
        "units": weather.get("hourly_units", {}),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Historical gridded weather estimates. "
            "Returned coordinates identify the weather grid cell, "
            "not a street-level measurement."
        )
    }

    metadata_path = OUTPUT_DIR / "rainfall_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\n-------------------------------")
    print(f"RAINFALL COLLECTION COMPLETE: {area_name}")
    print("-------------------------------")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    print(
        "Returned weather grid coordinates:",
        weather.get("latitude"),
        weather.get("longitude")
    )

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nTotal precipitation (mm):",
          df["precipitation_mm"].sum())

    print("\nMaximum hourly precipitation (mm):",
          df["precipitation_mm"].max())

    print("\nCSV saved:", csv_path)
    print("Metadata saved:", metadata_path)

    print("\nFirst five records:")
    print(df.head())

    return df, metadata


