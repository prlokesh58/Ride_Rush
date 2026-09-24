"""
LINEAR REGRESSION - SKLEARN
Predicts "fare_amount" using sklearn LinearRegression and lets the
user type trip details into a form to get a fare estimate.
"""

import os
import pickle
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common import ARTIFACT_DIR, RANDOM_STATE, figure_to_base64, load_raw_data

FEATURES = [
    "trip_distance",
    "passenger_count",
    "trip_duration_min",
    "is_rush_hour",
]

TARGET = "fare_amount"

MODEL_FILE = os.path.join(ARTIFACT_DIR, "linear_regression_fare.pkl")


def run_linear_regression_sklearn() -> dict:

    data = load_raw_data()
    data = data[FEATURES + [TARGET]].dropna()
    data = data[data[TARGET] > 0]

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "mse": round(float(mse), 4),
        "rmse": round(float(sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.scatter(y_test, y_pred, s=8, alpha=0.4)
    axis.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="crimson",
    )
    axis.set_xlabel("Actual Fare Amount ($)")
    axis.set_ylabel("Predicted Fare Amount ($)")
    axis.set_title("Actual vs Predicted Fare")
    actual_vs_predicted_plot = figure_to_base64(figure)

    residuals = y_test - y_pred

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.hist(residuals, bins=30)
    axis.set_xlabel("Residual")
    axis.set_ylabel("Frequency")
    axis.set_title("Residual Histogram")
    residual_plot = figure_to_base64(figure)

    coefficients = [
        {"feature": feature, "coefficient": round(float(value), 4)}
        for feature, value in zip(FEATURES, model.coef_)
    ]

    return {
        "features": FEATURES,
        "rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "coefficients": coefficients,
        "intercept": round(float(model.intercept_), 4),
        "actual_vs_predicted_plot": actual_vs_predicted_plot,
        "residual_plot": residual_plot,
    }


def predict_fare(form_data: dict) -> dict:
    """Used by the form on the Linear Regression (Sklearn) page."""

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Fare model not found. Open the Linear Regression (Sklearn) "
            "page once so the model gets trained and saved."
        )

    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    features = bundle["features"]

    row = []
    for feature in features:
        value = form_data.get(feature, "")
        try:
            row.append(float(value))
        except (TypeError, ValueError):
            row.append(0.0)

    input_df = pd.DataFrame([row], columns=features)
    fare = float(model.predict(input_df)[0])

    return {
        "inputs": dict(zip(features, row)),
        "predicted_fare": round(fare, 2),
    }