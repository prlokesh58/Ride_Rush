import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for web application

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from load_data import load_data

# Set styling
sns.set_theme(style="whitegrid")

# Charts directory configuration
CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)

def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)

def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")

def run_eda() -> dict:
    print("\n========== EDA STARTED ==========")

    # 1. LOAD DATA
    data = load_data()
    print("=" * 80)
    print("1. Data loaded")
    print("=" * 80)
    print("shape:", data.shape)
    print("\nFirst 5 rows of data:\n", data.head())

    charts = []

    # 2. BASIC INFO / STRUCTURE
    print("\n" + "=" * 80)
    print("2. BASIC INFO")
    print("=" * 80)
    data.info()
    print("\nColumns dtypes:\n", data.dtypes)
    print("\nDescribe (numeric):\n", data.describe())
    try:
        print("\nDescribe (categorical):\n", data.describe(include="object"))
    except ValueError:
        print("\nDescribe (categorical): No categorical columns to describe.")

    # 3. MISSING VALUES
    print("\n" + "=" * 80)
    print("3. MISSING VALUES")
    print("=" * 80)
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_pct"] > 0].sort_values(by="missing_count", ascending=False)
    print(missing_df)

    if not missing_df.empty:
        plt.figure(figsize=(10, 5), dpi=100)
        sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Percentage of missing values")
        plt.title("Missing values by column")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # 4. DUPLICATES
    print("\n" + "=" * 80)
    print("4. Duplicate Rows")
    print("=" * 80)
    duplicate_count = int(data.duplicated().sum())
    print("Duplicate rows:", duplicate_count)

    # 5. TARGET VARIABLE - FARE AMOUNT
    print("\n" + "=" * 80)
    print("5. TARGET VARIABLE - FARE_AMOUNT")
    print("=" * 80)
    fare_statistics = {}
    if "fare_amount" in data.columns:
        fare_statistics = data["fare_amount"].describe().to_dict()
        print(data["fare_amount"].describe())

        plt.figure(dpi=125)
        sns.histplot(data["fare_amount"].dropna(), kde=True, bins=40)
        plt.axvline(x=np.mean(data["fare_amount"]), color="green", linestyle="--", label="Mean")
        plt.legend()
        plt.xlabel("Fare Amount ($)")
        plt.ylabel("Count")
        plt.title("Fare Amount Distribution")
        _save("target_distribution.png")
        charts.append("target_distribution.png")

    # 6. NUMERIC FEATURE DISTRIBUTION
    print("\n" + "=" * 80)
    print("6. NUMERIC FEATURE DISTRIBUTION")
    print("=" * 80)
    hist_cols = ["trip_distance", "fare_amount", "tip_amount", "tolls_amount",
                 "total_amount", "passenger_count", "trip_duration_min",
                 "extra", "congestion_surcharge"]
    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:
        data[hist_cols].hist(figsize=(14, 10), bins=20)
        _save("numeric_distribution.png")
        charts.append("numeric_distribution.png")

    # Mean line example for trip_distance
    if "trip_distance" in data.columns:
        plt.figure(dpi=125)
        sns.histplot(data["trip_distance"], kde=True)
        plt.axvline(x=np.mean(data["trip_distance"]), color="green", linestyle="--", label="Mean")
        plt.legend()
        plt.title("Trip Distance Distribution with mean")
        _save("trip_distance_distribution_mean.png")
        charts.append("trip_distance_distribution_mean.png")

    # 7. OUTLIER DETECTION (BOXPLOTS)
    print("\n" + "=" * 80)
    print("7. OUTLIER DETECTION (BOXPLOTS)")
    print("=" * 80)
    box_cols = ["trip_distance", "fare_amount", "tip_amount", "tolls_amount",
                "total_amount", "passenger_count", "trip_duration_min"]
    box_cols = [c for c in box_cols if c in data.columns]

    for col in box_cols:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=data[col], color="skyblue")
        plt.title(f"Boxplot of {col}", fontsize=18)
        col_safe = col.replace(" ", "_").lower()
        fname = f"boxplot_{col_safe}.png"
        _save(fname)
        charts.append(fname)

    # 8. CORRELATION ANALYSIS (Multivariate)
    print("\n" + "=" * 80)
    print("8. CORRELATION ANALYSIS")
    print("=" * 80)
    corr = data.select_dtypes(include=[np.number]).corr()
    print(np.round(corr, 2))

    plt.figure(figsize=(16, 12), dpi=100)
    sns.heatmap(np.round(corr, 2), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    _save("correlation_heatmap.png")
    charts.append("correlation_heatmap.png")

    # 9. SCATTER OR REGRESSION PLOTS (BI-VARIATE)
    print("\n" + "=" * 80)
    print("9. RELATIONSHIP PLOTS (BI-VARIATE)")
    print("=" * 80)

    if "trip_distance" in data.columns and "fare_amount" in data.columns:
        plt.figure(figsize=(12, 6), dpi=100)
        sns.regplot(x=data["trip_distance"], y=data["fare_amount"], data=data,
                     color="lightblue", scatter_kws={"alpha": 0.3})
        plt.title("Trip Distance vs Fare Amount")
        _save("distance_vs_fare.png")
        charts.append("distance_vs_fare.png")

    if "passenger_count" in data.columns and "fare_amount" in data.columns:
        plt.figure(figsize=(12, 6), dpi=100)
        sns.scatterplot(x="passenger_count", y="fare_amount", data=data,
                         color="lightblue", alpha=0.3)
        plt.title("Passenger Count vs Fare Amount")
        _save("passengers_vs_fare.png")
        charts.append("passengers_vs_fare.png")

    # 10. CATEGORICAL FEATURE COUNTS
    print("\n" + "=" * 80)
    print("10. Categorical feature counts")
    print("=" * 80)
    cat_cols = ["payment_type", "VendorID", "RatecodeID", "PUBorough", "DOBorough"]
    cat_cols = [c for c in cat_cols if c in data.columns]

    for col in cat_cols:
        print(f"\n-----{col}-----\n")
        plt.figure(figsize=(10, 5), dpi=125)
        order = data[col].value_counts().index
        sns.countplot(x=col, order=order, data=data)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        col_safe = col.replace(" ", "_").lower()
        fname = f"countplot_{col_safe}.png"
        _save(fname)
        charts.append(fname)

    # 11. PAYMENT TYPE VS FARE AMOUNT
    print("\n" + "=" * 80)
    print("11. PAYMENT TYPE VS FARE AMOUNT")
    print("=" * 80)
    if "payment_type" in data.columns and "fare_amount" in data.columns:
        plt.figure(dpi=125)
        sns.boxplot(x="payment_type", y="fare_amount", data=data)
        plt.title("Fare Amount by Payment Type")
        _save("payment_type_vs_fare.png")
        charts.append("payment_type_vs_fare.png")

    # 12. PICKUP BOROUGH VS FARE AMOUNT
    print("\n" + "=" * 80)
    print("12. PICKUP BOROUGH VS FARE AMOUNT")
    print("=" * 80)
    if "PUBorough" in data.columns and "fare_amount" in data.columns:
        plt.figure(figsize=(10, 6), dpi=125)
        sns.boxplot(x="PUBorough", y="fare_amount", data=data)
        plt.title("Fare Amount by Pickup Borough")
        _save("borough_vs_fare.png")
        charts.append("borough_vs_fare.png")

    # 13. RUSH HOUR TREND -- trip volume & average fare across the day
    print("\n" + "=" * 80)
    print("13. RUSH HOUR TREND (avg fare / trip volume by pickup hour)")
    print("=" * 80)
    hourly_avg_fare = {}
    if "pickup_hour" in data.columns:
        hourly_counts = data["pickup_hour"].value_counts().sort_index()
        print("Trips by hour:\n", hourly_counts)

        plt.figure(figsize=(10, 6))
        plt.plot(hourly_counts.index, hourly_counts.values, marker="o")
        plt.title("Trip Volume by Pickup Hour (Rush Hour Pattern)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Trips")
        plt.xticks(range(0, 24))
        _save("rush_hour_trip_volume.png")
        charts.append("rush_hour_trip_volume.png")

        if "fare_amount" in data.columns:
            hourly_avg_fare_series = data.groupby("pickup_hour")["fare_amount"].mean()
            hourly_avg_fare = hourly_avg_fare_series.to_dict()
            print("\nAverage fare by hour:\n", hourly_avg_fare_series)

            plt.figure(figsize=(10, 6))
            plt.plot(hourly_avg_fare_series.index, hourly_avg_fare_series.values,
                      marker="o", color="darkorange")
            plt.title("Average Fare by Pickup Hour")
            plt.xlabel("Hour of Day")
            plt.ylabel("Average Fare ($)")
            plt.xticks(range(0, 24))
            _save("avg_fare_by_hour.png")
            charts.append("avg_fare_by_hour.png")

        if "pickup_dayofweek" in data.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            heat_data = (
                data.groupby(["pickup_dayofweek", "pickup_hour"])
                .size()
                .unstack(fill_value=0)
                .reindex(day_order)
            )
            plt.figure(figsize=(14, 6), dpi=100)
            sns.heatmap(heat_data, cmap="YlOrRd")
            plt.title("Trip Volume Heatmap: Day of Week vs Hour")
            plt.xlabel("Hour of Day")
            plt.ylabel("Day of Week")
            _save("rush_hour_heatmap.png")
            charts.append("rush_hour_heatmap.png")

    # 14. FARE ANALYSIS (UNI-VARIATE & BI-VARIATE)
    print("\n" + "=" * 80)
    print("14. FARE AMOUNT ANALYSIS: RUSH VS NON-RUSH HOUR")
    print("=" * 80)
    rush_hour_comparison = {}
    rush_hour_counts = {}
    if "is_rush_hour" in data.columns:
        rush_hour_counts = data["is_rush_hour"].value_counts().to_dict()
    if "is_rush_hour" in data.columns and "fare_amount" in data.columns:
        rush_hour_comparison = (
            data.groupby("is_rush_hour")["fare_amount"].mean().to_dict()
        )
        print(data.groupby("is_rush_hour")["fare_amount"].describe())

        plt.figure(figsize=(10, 6), dpi=120)
        sns.boxplot(x="is_rush_hour", y="fare_amount", data=data)
        plt.xlabel("Rush Hour (0 = No, 1 = Yes)")
        plt.ylabel("Fare Amount ($)")
        plt.title("Fare Amount: Rush Hour vs Non-Rush Hour")
        _save("fare_rush_vs_nonrush.png")
        charts.append("fare_rush_vs_nonrush.png")

    # 15. PAIRPLOT (MULTI-VARIATE)
    print("\n" + "=" * 80)
    print("15. PAIRPLOT (MULTI-VARIATE)")
    print("=" * 80)
    pairplot_cols = ["trip_distance", "fare_amount", "tip_amount",
                      "trip_duration_min", "is_rush_hour"]
    pairplot_cols = [c for c in pairplot_cols if c in data.columns]

    if {"trip_distance", "fare_amount", "tip_amount", "trip_duration_min", "is_rush_hour"}.issubset(data.columns):
        pairplot_df = data[pairplot_cols].dropna()
        if len(pairplot_df) > 1000:
            pairplot_df = pairplot_df.sample(n=1000, random_state=42)

        sns.pairplot(
            pairplot_df,
            vars=["trip_distance", "fare_amount", "tip_amount", "trip_duration_min"],
            hue="is_rush_hour",
            corner=True,
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "s": 25},
        )
        plt.savefig(_chart_path("pairplot.png"), bbox_inches="tight")
        plt.close("all")
        charts.append("pairplot.png")

    # Numeric and Categorical columns classification
    numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(data.select_dtypes(include=["object", "category"]).columns)

    print("\n========== EDA COMPLETED ==========")
    print("Charts generated:", len(charts))

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": duplicate_count,
        "missing": {col: int(cnt) for col, cnt in missing.items() if cnt > 0},
        "fare_statistics": {str(k): float(v) for k, v in fare_statistics.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "hourly_avg_fare": {str(k): float(v) for k, v in hourly_avg_fare.items()},
        "rush_hour_comparison": {str(k): float(v) for k, v in rush_hour_comparison.items()},
        "rush_hour_counts": {str(k): int(v) for k, v in rush_hour_counts.items()},
        "charts": charts,
    }

if __name__ == "__main__":
    results = run_eda()
    print("\n================================================")
    print("EDA RESULTS PREVIEW")
    print("================================================")
    print("Rows:", results["n_rows"])
    print("Columns:", results["n_cols"])
    print("Duplicate rows:", results["duplicate_count"])
    print("Missing values count:", results["missing"])
    print("Fare statistics:", results["fare_statistics"])
    print("Charts generated:")
    for chart in results["charts"]:
        print(" -", chart)
