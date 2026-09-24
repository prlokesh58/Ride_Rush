"""
MODEL TRAINING
Trains three classifiers on the preprocessed data (from preprocessing.py)
to predict is_rush_hour, compares them, and saves the best one.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

ARTIFACT_DIR = "artifacts"

CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def _load_splits():
    paths = {
        "X_train": os.path.join(ARTIFACT_DIR, "X_train.npy"),
        "X_test": os.path.join(ARTIFACT_DIR, "X_test.npy"),
        "y_train": os.path.join(ARTIFACT_DIR, "y_train.npy"),
        "y_test": os.path.join(ARTIFACT_DIR, "y_test.npy"),
    }
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing preprocessed artifacts: {missing}. Run the Preprocessing step first."
        )
    return {name: np.load(p) for name, p in paths.items()}


def run_model_training() -> dict:
    splits = _load_splits()
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    results = []
    best_name, best_model, best_f1 = None, None, -1.0

    for name, model in CANDIDATE_MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        results.append(metrics)

        if metrics["f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, metrics["f1"]

    with open(os.path.join(ARTIFACT_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump({"name": best_name, "model": best_model}, f)

    return {
        "results": results,
        "best_model": best_name,
        "best_f1": best_f1,
    }