import pandas as pd
import json

from pathlib import Path


def extract_rainfall_events(raw_dir, processed_dir, area_name):
    """
    Extract rainfall events for a selected study area.

    Parameters
    ----------
    raw_dir : Path or str
        Area-specific raw data directory.
    processed_dir : Path or str
        Area-specific processed data directory.
    area_name : str
        Name of the selected study area.
    """

    # -----------------------------------------
    # CONFIGURATION
    # -----------------------------------------

    INPUT_FILE = Path(raw_dir) / "rainfall" / "rainfall_hourly.csv"

    OUTPUT_DIR = Path(processed_dir) / "rainfall"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RAIN_THRESHOLD_MM = 0.1
    DRY_GAP_HOURS = 6

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Rainfall input file not found: {INPUT_FILE}"
        )

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------

    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["precipitation_mm"].isna().any():
        raise ValueError("Missing hourly precipitation values detected.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if not df["timestamp"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Hourly timestamps are not continuous.")

    print(f"Processing rainfall events for {area_name}...")
    print("Loaded hourly records:", len(df))

    # -----------------------------------------
    # IDENTIFY WET HOURS
    # -----------------------------------------

    wet = df["precipitation_mm"] >= RAIN_THRESHOLD_MM

    wet_indices = df.index[wet].tolist()

    events = []
    event_ids = pd.Series(index=df.index, dtype="object")

    if wet_indices:

        groups = []
        current_group = [wet_indices[0]]

        for index in wet_indices[1:]:

            previous = current_group[-1]

            gap = (
                df.loc[index, "timestamp"]
                - df.loc[previous, "timestamp"]
            ).total_seconds() / 3600

            if gap > DRY_GAP_HOURS:
                groups.append(current_group)
                current_group = [index]
            else:
                current_group.append(index)

        groups.append(current_group)

        # -------------------------------------
        # CALCULATE EVENT FEATURES
        # -------------------------------------

        for number, group in enumerate(groups, start=1):

            first_idx = group[0]
            last_idx = group[-1]

            start_time = df.loc[first_idx, "timestamp"]
            end_time = df.loc[last_idx, "timestamp"]

            event_rows = df.loc[first_idx:last_idx]

            # Rainfall during the 24 hours
            # immediately preceding the event.
            antecedent_start = start_time - pd.Timedelta(hours=24)

            antecedent_rows = df[
                (df["timestamp"] >= antecedent_start) &
                (df["timestamp"] < start_time)
            ]

            event_id = f"EVT_{number:04d}"

            event_ids.loc[first_idx:last_idx] = event_id

            events.append({

                "event_id": event_id,

                "event_start": start_time,
                "event_end": end_time,

                "duration_hours": len(event_rows),

                "wet_hours": len(group),

                "total_precipitation_mm":
                    event_rows["precipitation_mm"].sum(),

                "peak_hourly_rainfall_mm":
                    event_rows["precipitation_mm"].max(),

                "max_3h_rainfall_mm":
                    event_rows["rainfall_3h_mm"].max(),

                "max_6h_rainfall_mm":
                    event_rows["rainfall_6h_mm"].max(),

                "antecedent_24h_rainfall_mm":
                    antecedent_rows["precipitation_mm"].sum()

            })

    # -----------------------------------------
    # SAVE EVENT DATASET
    # -----------------------------------------

    events_df = pd.DataFrame(events)

    events_path = OUTPUT_DIR / "rainfall_events.csv"

    events_df.to_csv(events_path, index=False)

    # -----------------------------------------
    # SAVE HOURLY EVENT MAPPING
    # -----------------------------------------

    df["event_id"] = event_ids

    hourly_path = OUTPUT_DIR / "rainfall_hourly_with_events.csv"

    df.to_csv(hourly_path, index=False)

    # -----------------------------------------
    # SAVE EVENT DEFINITION
    # -----------------------------------------

    metadata = {
        "area": area_name,
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "dry_gap_hours": DRY_GAP_HOURS,
        "event_definition": (
            "Wet hours separated by more than the configured "
            "dry gap are treated as separate events."
        ),
        "number_of_events": len(events_df),
        "source_file": str(INPUT_FILE)
    }

    metadata_path = OUTPUT_DIR / "event_definition.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\n-------------------------------")
    print(f"RAINFALL EVENT EXTRACTION: {area_name}")
    print("-------------------------------")

    print("Total hourly records:", len(df))
    print("Rainfall events detected:", len(events_df))

    if not events_df.empty:

        print(
            "Maximum event rainfall:",
            events_df["total_precipitation_mm"].max(),
            "mm"
        )

        print(
            "Maximum event duration:",
            events_df["duration_hours"].max(),
            "hours"
        )

        print("\nFirst five events:")
        print(events_df.head())

    print("\nSaved:")
    print(events_path)
    print(hourly_path)
    print(metadata_path)

    return events_df, df


