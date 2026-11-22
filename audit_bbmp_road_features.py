import pandas as pd

FILE = "data/processed/features/road_features_with_bbmp.csv"

df = pd.read_csv(FILE)

distance_columns = [
    "bbmp_distance_m",
    "bbmp_vulnerable_distance_m",
    "bbmp_flood_prone_distance_m",
    "bbmp_low_lying_distance_m"
]

flag_columns = [
    "bbmp_within_50m",
    "bbmp_within_100m",
    "bbmp_vulnerable_within_50m",
    "bbmp_vulnerable_within_100m",
    "bbmp_flood_prone_within_50m",
    "bbmp_flood_prone_within_100m",
    "bbmp_low_lying_within_50m",
    "bbmp_low_lying_within_100m"
]

print("\n========== BBMP FEATURE AUDIT ==========")

print("Total road records:", len(df))

print("\nDistance statistics:")

for col in distance_columns:
    if col in df.columns:
        print(f"\n{col}")
        print(df[col].describe())

print("\nProximity flag counts:")

for col in flag_columns:
    if col in df.columns:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False))

print("\nRoads close to multiple BBMP categories:")

category_flags = [
    "bbmp_vulnerable_within_100m",
    "bbmp_flood_prone_within_100m",
    "bbmp_low_lying_within_100m"
]

available = [c for c in category_flags if c in df.columns]

if available:
    df["number_of_nearby_categories"] = df[available].sum(axis=1)

    print(
        df["number_of_nearby_categories"]
        .value_counts()
        .sort_index()
    )

print("\nDuplicate road IDs:", df["road_id"].duplicated().sum())

print("\nAudit complete.")
