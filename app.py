# hw4 flask api - layer 4: adding /predict/batch endpoint
# single-record and batch predictions with full validation

from flask import Flask, jsonify, request
import joblib
import pandas as pd
import json

app = Flask(__name__)

# load model and valid categories at startup
print("Loading model...")
model = joblib.load("model/model.pkl")
print("Model loaded.")

print("Loading valid categories...")
with open("valid_categories.json") as f:
    valid_categories = json.load(f)
print(f"Valid categories loaded: {list(valid_categories.keys())}")

# feature schema
numeric_features = [
    "delivery_days", "delivery_vs_estimated", "number_of_items",
    "total_order_value", "avg_product_price", "max_product_price",
    "min_product_price", "total_freight_value", "avg_freight_value",
    "max_freight_value", "min_product_photos_qty"
]
categorical_features = ["product_category", "seller_state", "payment_type"]
all_features = numeric_features + categorical_features

non_negative_fields = [
    "delivery_days", "number_of_items", "total_order_value",
    "avg_product_price", "max_product_price", "min_product_price",
    "total_freight_value", "avg_freight_value", "max_freight_value",
    "min_product_photos_qty"
]

MAX_BATCH_SIZE = 100


def validate_features(data):
    """check a single order dict against feature rules, return dict of errors"""
    errors = {}

    for field in all_features:
        if field not in data or data[field] is None:
            errors[field] = "required field missing"

    for field in numeric_features:
        if field not in errors and field in data:
            value = data[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors[field] = "must be a number"
            elif field in non_negative_fields and value < 0:
                errors[field] = "must be non-negative"

    for field in categorical_features:
        if field not in errors and field in data:
            value = data[field]
            if not isinstance(value, str):
                errors[field] = "must be a string"
            elif value not in valid_categories[field]:
                errors[field] = f"unknown value: '{value}'"

    return errors


def build_prediction_response(prediction, probability):
    """convert one prediction + probability into the response dict"""
    pred_int = int(prediction)
    return {
        "prediction": pred_int,
        "probability": round(float(probability), 4),
        "label": "positive" if pred_int == 1 else "negative"
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Invalid input",
            "details": {"body": "request body must be valid json with content-type application/json"}
        }), 400

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid input",
            "details": {"body": "request body must be a json object"}
        }), 400

    errors = validate_features(data)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    features_df = pd.DataFrame([{k: data[k] for k in all_features}])
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    return jsonify(build_prediction_response(prediction, probability))


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Invalid input",
            "details": {"body": "request body must be valid json with content-type application/json"}
        }), 400

    # must be a list of records
    if not isinstance(data, list):
        return jsonify({
            "error": "Invalid input",
            "details": {"body": "request body must be a json array of order objects"}
        }), 400

    # batch size limits
    if len(data) == 0:
        return jsonify({
            "error": "Invalid input",
            "details": {"body": "batch must contain at least one record"}
        }), 400
    if len(data) > MAX_BATCH_SIZE:
        return jsonify({
            "error": "Invalid input",
            "details": {"body": f"batch size {len(data)} exceeds limit of {MAX_BATCH_SIZE}"}
        }), 400

    # validate every record, collect errors by index
    all_errors = {}
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            all_errors[f"record_{i}"] = {"body": "must be a json object"}
            continue
        errors = validate_features(record)
        if errors:
            all_errors[f"record_{i}"] = errors

    if all_errors:
        return jsonify({"error": "Invalid input", "details": all_errors}), 400

    # all records valid - predict in one batched call
    features_df = pd.DataFrame([{k: r[k] for k in all_features} for r in data])
    predictions = model.predict(features_df)
    probabilities = model.predict_proba(features_df)[:, 1]

    results = [
        build_prediction_response(pred, prob)
        for pred, prob in zip(predictions, probabilities)
    ]

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)