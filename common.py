"""
Common settings and small helpers shared by every RideRush module.
"""

import base64
import io
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402

from load_data import load_data          # noqa: E402

# ============================================================
# CONFIG
# ============================================================

ARTIFACT_DIR = "artifacts"

# Binary classification target (mirrors PlacementStatus in the placement app)
TARGET_COLUMN = "is_rush_hour"

# Continuous regression target (mirrors Salary Package in the placement app)
REGRESSION_TARGET = "fare_amount"

# Columns that must never be used as classification input features:
#   - pickup_hour directly determines is_rush_hour -> leakage
#   - raw datetime columns can't go through a simple encoder
#   - PUZone/DOZone/*ServiceZone are very high-cardinality text (260+ zones)
#   - PULocationID/DOLocationID are arbitrary ID codes, like StudentID
LEAKAGE_COLUMNS = [
    "pickup_hour",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PUZone",
    "DOZone",
    "PUServiceZone",
    "DOServiceZone",
    "PULocationID",
    "DOLocationID",
]

SAMPLE_ROWS = None   # dataset is small (a few thousand rows); no sampling needed

RANDOM_STATE = 42

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_data():
    """Load the trip data with all engineered columns (rush hour, borough, etc.)."""
    return load_data()


def sample_data(data, n_rows=SAMPLE_ROWS):
    if n_rows is None or len(data) <= n_rows:
        return data
    return data.sample(n=n_rows, random_state=RANDOM_STATE)


def drop_leakage_columns(data):
    columns_to_remove = [c for c in LEAKAGE_COLUMNS if c in data.columns]
    return data.drop(columns=columns_to_remove)


# ============================================================
# PLOT HELPER
# ============================================================

def figure_to_base64(figure=None):
    """Convert the current matplotlib figure into a base64 PNG string."""
    if figure is None:
        figure = plt.gcf()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(figure)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")

    return "data:image/png;base64," + encoded


def dataframe_to_table(data, max_rows=10):
    subset = data.head(max_rows)

    return {
        "columns": [str(c) for c in subset.columns],
        "rows": subset.astype(object).where(subset.notna(), "").values.tolist(),
    }