"""
PREPROCESSING STEP
Builds the numeric + categorical pipeline for predicting is_rush_hour
and saves train/test arrays + the fitted preprocessor to artifacts/,
ready for Model Training and Prediction.
"""

import os
import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    ARTIFACT_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    drop_leakage_columns,
    load_raw_data,
)

TEST_SIZE = 0.2

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def run_preprocessing() -> dict:
    df = load_raw_data()
    rows_before = len(df)

    missing_before = df.isnull().sum()
    missing_before = missing_before[missing_before > 0].to_dict()
    missing_before = {str(k): int(v) for k, v in missing_before.items()}

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}"
        )

    y = df[TARGET_COLUMN]
    X = drop_leakage_columns(df.drop(columns=[TARGET_COLUMN]))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    with open(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(ARTIFACT_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)

    np.save(
        os.path.join(ARTIFACT_DIR, "X_train.npy"),
        X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed,
    )
    np.save(
        os.path.join(ARTIFACT_DIR, "X_test.npy"),
        X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed,
    )
    np.save(os.path.join(ARTIFACT_DIR, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(ARTIFACT_DIR, "y_test.npy"), y_test.to_numpy())

    return {
        "rows_before": rows_before,
        "rows_after": rows_before,
        "missing_before": missing_before,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }