import os
import pandas as pd

# Ships with a bundled sample so the app runs out of the box.
# Swap this to your real NYC TLC trip record CSV when you have one
# (e.g. a yellow_tripdata_YYYY-MM.csv downloaded from
#  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page )
DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_trip_data.csv")

ZONE_LOOKUP_PATH = os.path.join(os.path.dirname(__file__), "taxi_zone_lookup.csv")

# Standard NYC TLC yellow-cab column names this app expects.
# Any column that isn't present in your file is simply skipped downstream.
PICKUP_DT_COL = "tpep_pickup_datetime"
DROPOFF_DT_COL = "tpep_dropoff_datetime"


def load_zone_lookup(path: str = ZONE_LOOKUP_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    # Parse datetimes if present
    for col in (PICKUP_DT_COL, DROPOFF_DT_COL):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Trip duration in minutes
    if PICKUP_DT_COL in df.columns and DROPOFF_DT_COL in df.columns:
        df["trip_duration_min"] = (
            df[DROPOFF_DT_COL] - df[PICKUP_DT_COL]
        ).dt.total_seconds() / 60.0

    # Merge in pickup / dropoff Borough, Zone, service_zone from the TLC lookup table
    zones = load_zone_lookup()

    if "PULocationID" in df.columns:
        pu = zones.rename(columns={
            "LocationID": "PULocationID",
            "Borough": "PUBorough",
            "Zone": "PUZone",
            "service_zone": "PUServiceZone",
        })
        df = df.merge(pu, on="PULocationID", how="left")

    if "DOLocationID" in df.columns:
        do = zones.rename(columns={
            "LocationID": "DOLocationID",
            "Borough": "DOBorough",
            "Zone": "DOZone",
            "service_zone": "DOServiceZone",
        })
        df = df.merge(do, on="DOLocationID", how="left")

    # Rush-hour / time-of-day features -- this is the "Ride Rush" part
    if PICKUP_DT_COL in df.columns:
        df["pickup_hour"] = df[PICKUP_DT_COL].dt.hour
        df["pickup_dayofweek"] = df[PICKUP_DT_COL].dt.day_name()
        df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    return df


def get_data_summary() -> dict:
    df = load_data()
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).astype(str).to_dict("records"),
    }
    return summary


if __name__ == "__main__":
    print(get_data_summary())
