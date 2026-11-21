import pandas as pd

EVENT_FILE = "data/processed/rainfall/rainfall_events.csv"

events = pd.read_csv(EVENT_FILE, parse_dates=[
    "event_start",
    "event_end"
])

# Recalculate elapsed event span
events["span_hours"] = (
    (events["event_end"] - events["event_start"])
    .dt.total_seconds() / 3600
) + 1

print("\n========== RAINFALL EVENT AUDIT ==========")

print("\nTotal events:", len(events))

print("\nRainfall total statistics:")
print(events["total_precipitation_mm"].describe())

print("\nEvent duration statistics:")
print(events["span_hours"].describe())

print("\nWet hours statistics:")
print(events["wet_hours"].describe())

print("\nLongest 10 events:")
print(
    events.sort_values("span_hours", ascending=False)[
        [
            "event_id",
            "event_start",
            "event_end",
            "span_hours",
            "wet_hours",
            "total_precipitation_mm",
            "peak_hourly_rainfall_mm"
        ]
    ].head(10).to_string(index=False)
)

print("\n10 events with highest total rainfall:")
print(
    events.sort_values(
        "total_precipitation_mm",
        ascending=False
    )[
        [
            "event_id",
            "event_start",
            "event_end",
            "span_hours",
            "wet_hours",
            "total_precipitation_mm",
            "peak_hourly_rainfall_mm"
        ]
    ].head(10).to_string(index=False)
)

# Exploratory categories, not official flood thresholds
events["rainfall_category"] = pd.cut(
    events["total_precipitation_mm"],
    bins=[-0.001, 2, 10, 25, 50, float("inf")],
    labels=[
        "Very light",
        "Light",
        "Moderate",
        "Heavy",
        "Very heavy"
    ]
)

print("\nExploratory rainfall categories:")
print(events["rainfall_category"].value_counts().sort_index())

print("\nAudit completed.")
