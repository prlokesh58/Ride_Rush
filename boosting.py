"""
BOOSTING MODELS
Both AdaBoost and Gradient Boosting predict is_rush_hour with the
same preprocessing pipeline; only the classifier changes.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (
    RANDOM_STATE,
    TARGET_COLUMN,
    drop_leakage_columns,
    figure_to_base64,
    load_raw_data,
    sample_data,
)


def _build_preprocessor(X):
    numerical_features = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor, numerical_features, categorical_features


def _confusion_plot(cm, title):
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(cm, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(cm.shape[0]))
    axis.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, cm[i, j], ha="center", va="center")
    return figure_to_base64(figure)


def _importance_plot(names, values, title, top_n=12):
    pairs = sorted(zip(names, values), key=lambda p: p[1], reverse=True)[:top_n]
    pairs = pairs[::-1]

    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.barh([p[0] for p in pairs], [p[1] for p in pairs])
    axis.set_xlabel("Importance")
    axis.set_title(title)

    return figure_to_base64(figure), [
        {"feature": name, "importance": round(float(value), 4)}
        for name, value in pairs[::-1]
    ]


def _run_boosting(classifier, label) -> dict:

    data = load_raw_data()
    data = sample_data(data)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(data.columns)}")

    y = data[TARGET_COLUMN]
    X = drop_leakage_columns(data.drop(columns=[TARGET_COLUMN]))

    preprocessor, numerical_features, categorical_features = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_plot, importance_table = _importance_plot(
        [str(n).split("__", 1)[-1] for n in feature_names],
        importances,
        label + " - Top Features",
    )

    sample_predictions = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": y_pred[:10],
    })

    return {
        "model_name": label,
        "rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "target_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "accuracy": round(float(accuracy), 4),
        "accuracy_percent": round(float(accuracy) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "confusion_plot": _confusion_plot(cm, label + " - Confusion Matrix"),
        "importance_plot": importance_plot,
        "importance_table": importance_table,
        "report": classification_report(y_test, y_pred, zero_division=0),
        "sample_predictions": sample_predictions.values.tolist(),
    }


def run_adaboost() -> dict:
    classifier = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        n_estimators=100,
        learning_rate=1.0,
        random_state=RANDOM_STATE,
    )

    result = _run_boosting(classifier, "AdaBoost")
    result["parameters"] = {
        "base estimator": "DecisionTree (max_depth=1)",
        "n_estimators": 100,
        "learning_rate": 1.0,
    }
    return result


def run_gradient_boosting() -> dict:
    classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    result = _run_boosting(classifier, "Gradient Boosting")
    result["parameters"] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    }
    return result