"""
ENCODING STEP
Converts the categorical columns (PUBorough, DOBorough, pickup_dayofweek, etc.)
into numbers using Label Encoding or One-Hot Encoding.
"""

import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common import ARTIFACT_DIR, dataframe_to_table, drop_leakage_columns, load_raw_data

ENCODED_FILE = os.path.join(ARTIFACT_DIR, "encoded_trips.csv")


def run_encoding(method="onehot") -> dict:
    """
    method = "onehot"  -> pd.get_dummies (drop_first=True)
    method = "label"   -> LabelEncoder on every object/str column
    """

    data = load_raw_data()
    data = drop_leakage_columns(data)

    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    # str dtype columns show up as "object" in most pandas versions, but be safe:
    categorical_columns += [
        c for c in data.columns
        if data[c].dtype.name == "str" and c not in categorical_columns
    ]

    numerical_columns = [
        c for c in data.columns
        if c not in categorical_columns
    ]

    category_info = []
    for column in categorical_columns:
        values = data[column].dropna().unique().tolist()
        category_info.append({
            "column": column,
            "unique_count": len(values),
            "sample_values": [str(v) for v in values[:10]],
        })

    before_shape = data.shape

    mapping_info = []

    if method == "label":
        encoded = data.copy()

        for column in categorical_columns:
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))

            mapping_info.append({
                "column": column,
                "mapping": {
                    str(label): int(code)
                    for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_))
                },
            })

    else:
        method = "onehot"

        encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        bool_columns = encoded.select_dtypes(include=["bool"]).columns
        encoded[bool_columns] = encoded[bool_columns].astype(int)

        new_columns = [c for c in encoded.columns if c not in data.columns]
        mapping_info.append({
            "column": "One-Hot Columns Created",
            "mapping": {c: 1 for c in new_columns},
        })

    after_shape = encoded.shape

    encoded.to_csv(ENCODED_FILE, index=False)

    rush_hour_counts = {}
    if "is_rush_hour" in encoded.columns:
        rush_hour_counts = {
            str(k): int(v) for k, v in encoded["is_rush_hour"].value_counts().items()
        }

    return {
        "method": "One-Hot Encoding" if method == "onehot" else "Label Encoding",
        "rows": int(before_shape[0]),
        "columns_before": int(before_shape[1]),
        "columns_after": int(after_shape[1]),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "category_info": category_info,
        "mapping_info": mapping_info,
        "target_counts": rush_hour_counts,
        "saved_to": ENCODED_FILE,
        "preview": dataframe_to_table(encoded, max_rows=10),
    }