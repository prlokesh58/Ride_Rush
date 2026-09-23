"""
RIDERUSH PROJECT - Flask app
Each sidebar button is one route.
"""

from flask import Flask, render_template, request

from load_data import get_data_summary

from fare_eda import run_eda
from preprocessing import run_preprocessing
from encoding import run_encoding
from model_training import run_model_training
from linear_regression_scalar import run_linear_regression_scalar
from linear_regression_sklearn import predict_fare, run_linear_regression_sklearn
from boosting import run_adaboost, run_gradient_boosting
from kmeans_clustering import get_feature_list, run_kmeans
from prediction import get_form_fields, predict_rush_hour

app = Flask(__name__)


# ============================================================
# HOME / DATA LOADING
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template("index.html", active="data-loading", summary=summary, error=error)


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda():
    error = None
    eda_output = None
    try:
        eda_output = run_eda()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template("eda.html", active="eda", results=eda_output, error=error)


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():
    result = run_preprocessing()
    return render_template("preprocessing.html", active="preprocessing", result=result)


# ============================================================
# ENCODING
# ============================================================

@app.route("/encoding", methods=["GET", "POST"])
def encoding_page():
    method = request.form.get("method", "onehot")
    result = run_encoding(method=method)
    return render_template("encoding.html", active="encoding", method=method, result=result)


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route("/model-training")
def model_training_page():
    result = run_model_training()
    return render_template("model_training.html", active="model_training", result=result)


# ============================================================
# LINEAR REGRESSION - SCALAR (classification demo: is_rush_hour)
# ============================================================

@app.route("/linear-regression-scalar")
def linear_regression_scalar_page():
    result = run_linear_regression_scalar()
    return render_template("linear_scalar.html", active="linear_scalar", result=result)


# ============================================================
# LINEAR REGRESSION - SKLEARN (regression demo: fare_amount)
# ============================================================

@app.route("/linear-regression-sklearn", methods=["GET", "POST"])
def linear_regression_sklearn_page():
    result = run_linear_regression_sklearn()

    prediction = None
    error = None

    if request.method == "POST":
        try:
            prediction = predict_fare(request.form.to_dict())
        except Exception as exception:
            error = str(exception)

    return render_template(
        "linear_sklearn.html",
        active="linear_sklearn",
        result=result,
        prediction=prediction,
        error=error,
    )


# ============================================================
# ADABOOST
# ============================================================

@app.route("/adaboost")
def adaboost_page():
    result = run_adaboost()
    return render_template("boosting.html", active="adaboost", result=result)


# ============================================================
# GRADIENT BOOSTING
# ============================================================

@app.route("/gradient-boosting")
def gradient_boosting_page():
    result = run_gradient_boosting()
    return render_template("boosting.html", active="gradient_boosting", result=result)


# ============================================================
# K-MEANS
# ============================================================

@app.route("/kmeans", methods=["GET", "POST"])
def kmeans_page():
    method = request.form.get("method", "elbow")
    k = request.form.get("k", 3)
    x_feature = request.form.get("x_feature", "trip_distance")
    y_feature = request.form.get("y_feature", "fare_amount")

    result = run_kmeans(method=method, k=k, x_feature=x_feature, y_feature=y_feature)

    return render_template(
        "kmeans.html",
        active="kmeans",
        result=result,
        features=get_feature_list(),
        selected_method=method,
    )


# ============================================================
# PREDICTION (is_rush_hour, using the Model Training pipeline)
# ============================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    fields = get_form_fields()

    result = None
    error = None

    if request.method == "POST":
        try:
            result = predict_rush_hour(request.form.to_dict())
        except Exception as exception:
            error = str(exception)

    return render_template(
        "prediction.html",
        active="prediction",
        fields=fields,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)