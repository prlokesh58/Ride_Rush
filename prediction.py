"""
PREDICTION
Loads the saved preprocessor + best model and predicts whether a new
trip is a rush-hour trip.
"""

import os
import pickle

import pandas as pd

ARTIFACT_DIR = "artifacts"


def get_form_fields() -> list:
    path = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Feature column list not found. Run the Preprocessing step first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_artifacts():
    preproc_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")

    if not os.path.exists(preproc_path):
        raise FileNotFoundError("Preprocessor not found. Run the Preprocessing step first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Run the Model Training step first.")

    with open(preproc_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    return preprocessor, model_bundle["model"], model_bundle["name"]


def predict_rush_hour(form_data: dict) -> dict:
    preprocessor, model, model_name = _load_artifacts()
    fields = get_form_fields()

    row = {}
    for field in fields:
        raw = form_data.get(field, "")
        try:
            row[field] = float(raw)
        except (TypeError, ValueError):
            row[field] = raw
    input_df = pd.DataFrame([row], columns=fields)

    X = preprocessor.transform(input_df)
    pred = model.predict(X)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(X)[0][1]), 4)

    return {
        "model_used": model_name,
        "prediction": int(pred),
        "prediction_label": "Rush Hour Trip" if int(pred) == 1 else "Non Rush-Hour Trip",
        "probability": probability,
    }