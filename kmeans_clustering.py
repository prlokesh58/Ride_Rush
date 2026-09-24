"""
K-MEANS CALCULATION
Clusters trips using numeric trip features (trip_distance, fare_amount,
passenger_count, trip_duration_min, tip_amount, ...).
K can be chosen manually, or found with the Elbow or Silhouette method.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE, figure_to_base64, load_raw_data

K_RANGE = list(range(2, 11))
SEARCH_SAMPLE = 2000

DEFAULT_X = "trip_distance"
DEFAULT_Y = "fare_amount"

CLUSTER_FEATURES = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "passenger_count",
    "trip_duration_min",
]


def get_feature_list():
    data = load_raw_data()
    available = [c for c in CLUSTER_FEATURES if c in data.columns]
    return available if available else data.select_dtypes(include=[np.number]).columns.tolist()


def _prepare():
    data = load_raw_data()
    features = get_feature_list()
    data = data[features].dropna()

    scaled = StandardScaler().fit_transform(data)
    return data, scaled


def _elbow(scaled):
    sample = scaled
    if len(sample) > SEARCH_SAMPLE:
        rng = np.random.RandomState(RANDOM_STATE)
        sample = sample[rng.choice(len(sample), SEARCH_SAMPLE, replace=False)]

    wcss = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(sample)
        wcss.append(float(kmeans.inertia_))

    x1, y1 = K_RANGE[0], wcss[0]
    x2, y2 = K_RANGE[-1], wcss[-1]

    distances = []
    for x0, y0 in zip(K_RANGE, wcss):
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    suggested_k = K_RANGE[int(np.argmax(distances))]

    figure, axis = plt.subplots(figsize=(5.5, 4))
    axis.plot(K_RANGE, wcss, marker="o")
    axis.axvline(suggested_k, linestyle="--", color="crimson")
    axis.set_xlabel("Number of Clusters (K)")
    axis.set_ylabel("WCSS")
    axis.set_title("Elbow Method")
    axis.set_xticks(K_RANGE)
    axis.grid(True, alpha=0.3)

    table = [{"k": k, "value": round(v, 2)} for k, v in zip(K_RANGE, wcss)]

    return suggested_k, table, figure_to_base64(figure)


def _silhouette(scaled):
    sample = scaled
    if len(sample) > SEARCH_SAMPLE:
        rng = np.random.RandomState(RANDOM_STATE)
        sample = sample[rng.choice(len(sample), SEARCH_SAMPLE, replace=False)]

    scores = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(sample)
        scores.append(float(silhouette_score(sample, labels)))

    best_k = K_RANGE[int(np.argmax(scores))]

    figure, axis = plt.subplots(figsize=(5.5, 4))
    axis.plot(K_RANGE, scores, marker="o")
    axis.axvline(best_k, linestyle="--", color="crimson")
    axis.set_xlabel("Number of Clusters (K)")
    axis.set_ylabel("Silhouette Score")
    axis.set_title("Silhouette Method")
    axis.set_xticks(K_RANGE)
    axis.grid(True, alpha=0.3)

    table = [{"k": k, "value": round(v, 4)} for k, v in zip(K_RANGE, scores)]

    return best_k, table, figure_to_base64(figure)


def run_kmeans(method="elbow", k=3, x_feature=DEFAULT_X, y_feature=DEFAULT_Y) -> dict:

    data, scaled = _prepare()

    method = (method or "elbow").lower()

    score_table = []
    method_plot = None
    score_label = ""

    if method == "manual":
        k = int(k)
        if k < 2:
            k = 2
        method_name = "Manual K Selection"

    elif method == "silhouette":
        k, score_table, method_plot = _silhouette(scaled)
        method_name = "Silhouette Method"
        score_label = "Silhouette Score"

    else:
        method = "elbow"
        k, score_table, method_plot = _elbow(scaled)
        method_name = "Elbow Method"
        score_label = "WCSS"

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(scaled)

    clustered = data.copy()
    clustered["Cluster"] = labels

    columns = clustered.columns.tolist()

    if x_feature not in columns:
        x_feature = DEFAULT_X if DEFAULT_X in columns else columns[0]
    if y_feature not in columns or y_feature == x_feature:
        y_feature = DEFAULT_Y if DEFAULT_Y in columns else columns[1]

    plot_sample = clustered
    if len(plot_sample) > 3000:
        plot_sample = plot_sample.sample(n=3000, random_state=RANDOM_STATE)

    figure, axis = plt.subplots(figsize=(6, 4.5))
    scatter = axis.scatter(
        plot_sample[x_feature],
        plot_sample[y_feature],
        c=plot_sample["Cluster"],
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    axis.set_xlabel(x_feature)
    axis.set_ylabel(y_feature)
    axis.set_title("K-Means Clustering (K = {})".format(k))
    figure.colorbar(scatter, ax=axis, label="Cluster")

    cluster_plot = figure_to_base64(figure)

    counts = clustered["Cluster"].value_counts().sort_index()

    cluster_summary = []
    for cluster_id in sorted(clustered["Cluster"].unique()):
        rows = clustered[clustered["Cluster"] == cluster_id]
        cluster_summary.append({
            "cluster": int(cluster_id),
            "count": int(len(rows)),
            "x_mean": round(float(rows[x_feature].mean()), 2),
            "y_mean": round(float(rows[y_feature].mean()), 2),
        })

    centre_means = clustered.groupby("Cluster").mean(numeric_only=True)

    return {
        "method": method,
        "method_name": method_name,
        "score_label": score_label,
        "score_table": score_table,
        "method_plot": method_plot,
        "k": int(k),
        "rows": int(len(clustered)),
        "features_used": int(data.shape[1]),
        "feature_list": columns[:-1],
        "x_feature": x_feature,
        "y_feature": y_feature,
        "cluster_plot": cluster_plot,
        "cluster_counts": {int(i): int(v) for i, v in counts.items()},
        "cluster_summary": cluster_summary,
        "centre_columns": centre_means.columns.tolist(),
        "centre_rows": [
            [int(index)] + [round(float(v), 2) for v in row]
            for index, row in zip(centre_means.index, centre_means.values)
        ],
    }