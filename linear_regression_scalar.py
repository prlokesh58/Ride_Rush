"""
LINEAR REGRESSION - SCALAR
Trains Logistic Regression on is_rush_hour three times:
  1. Unscaled
  2. StandardScaler
  3. MinMaxScaler
and compares the accuracies, confusion matrix and ROC curve.
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import RANDOM_STATE, figure_to_base64, load_raw_data

FEATURES = [
    "trip_distance",
    "passenger_count",
    "trip_duration_min",
    "fare_amount",
]

TARGET = "is_rush_hour"


def _train_and_evaluate(X_train, X_test, y_train, y_test, name):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {
        "name": name,
        "train_accuracy": round(accuracy_score(y_train, train_pred), 4),
        "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
    }

    return model, result, test_pred


def run_linear_regression_scalar() -> dict:

    data = load_raw_data()
    data = data[FEATURES + [TARGET]].dropna()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    _, unscaled_result, _ = _train_and_evaluate(X_train, X_test, y_train, y_test, "Unscaled")
    results.append(unscaled_result)

    standard = StandardScaler()
    X_train_std = standard.fit_transform(X_train)
    X_test_std = standard.transform(X_test)

    model_standard, standard_result, standard_pred = _train_and_evaluate(
        X_train_std, X_test_std, y_train, y_test, "StandardScaler"
    )
    results.append(standard_result)

    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)

    _, minmax_result, _ = _train_and_evaluate(
        X_train_minmax, X_test_minmax, y_train, y_test, "MinMaxScaler"
    )
    results.append(minmax_result)

    cm = confusion_matrix(y_test, standard_pred)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(cm, cmap="Blues")
    axis.set_title("Confusion Matrix - StandardScaler")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, cm[i, j], ha="center", va="center")
    confusion_plot = figure_to_base64(figure)

    roc_plot = None
    auc_score = None

    try:
        probability = model_standard.predict_proba(X_test_std)[:, 1]
        auc_score = round(roc_auc_score(y_test, probability), 4)

        fpr, tpr, _ = roc_curve(y_test, probability)

        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(fpr, tpr, label="ROC Curve")
        axis.plot([0, 1], [0, 1], linestyle="--")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title("ROC Curve")
        axis.legend()
        roc_plot = figure_to_base64(figure)
    except ValueError:
        auc_score = None

    coefficients = [
        {"feature": feature, "coefficient": round(float(value), 4)}
        for feature, value in zip(FEATURES, model_standard.coef_[0])
    ]

    return {
        "features": FEATURES,
        "rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": results,
        "confusion_matrix": cm.tolist(),
        "confusion_plot": confusion_plot,
        "roc_plot": roc_plot,
        "auc_score": auc_score,
        "coefficients": coefficients,
        "report": classification_report(y_test, standard_pred, zero_division=0),
    }