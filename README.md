# HW4 MLOps: Olist Customer Satisfaction API

## Project Overview

This project is an MLOps pipeline that predicts customer satisfaction using information available in the Olist database. The deployed API takes order features (such as delivery days, total order value, freight costs, product category, seller state, and payment type) and returns a prediction of whether the customer will leave a positive review (4 or 5 stars) or a negative review (1 to 3 stars), as well as a probability score.

The model serving being used is a Gradient Boosting Classifier trained on 14 features from the Olist e-commerce dataset. This model achieved an 81% accuracy and 0.73 AUC on the test set when being built. Unlike a foundation model that reads review text after the fact, this model predicts satisfaction before a review is written, using only order level information.

This matters because flagging potentially negative reviews before they happen allows for early intervention. Customer service can reach out to at-risk orders, offer a discount, track the shipment more carefully, or address any issue before it becomes a bad review. This helps with customer retention, brand image, and overall customer experience.

## Live API URL

**https://hw4-mlops-hae2.onrender.com**

The API is deployed on Render's free tier, which spins down after 15 minutes of inactivity. The first request after the server has been idle may take 30 to 60 seconds while the container wakes up. All requests after that should respond in under a second.

## API Documentation

The API has three endpoints.

### GET /health

Returns a simple JSON response confirming the API is running and the model is loaded. Used for monitoring and liveness checks.

Example request:
```
curl https://hw4-mlops-hae2.onrender.com/health
```

Example response:
```json
{"status": "healthy", "model": "loaded"}
```

### POST /predict

Accepts a single order as a JSON object with all 14 features. Returns a prediction, a probability score, and a human-readable label.

Example request:
```
curl -X POST https://hw4-mlops-hae2.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "delivery_days": 10.0,
    "delivery_vs_estimated": -5.0,
    "number_of_items": 1.0,
    "total_order_value": 89.90,
    "avg_product_price": 89.90,
    "max_product_price": 89.90,
    "min_product_price": 89.90,
    "total_freight_value": 15.50,
    "avg_freight_value": 15.50,
    "max_freight_value": 15.50,
    "min_product_photos_qty": 3.0,
    "product_category": "housewares",
    "seller_state": "SP",
    "payment_type": "credit_card"
  }'
```

Example response:
```json
{"prediction": 1, "probability": 0.736, "label": "positive"}
```

### POST /predict/batch

Accepts up to 100 orders as a JSON array. Returns an array of predictions in the same order as the input.

Example request:
```
curl -X POST https://hw4-mlops-hae2.onrender.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...order1...}, {...order2...}, {...order3...}]'
```

Example response:
```json
{
  "predictions": [
    {"prediction": 1, "probability": 0.736, "label": "positive"},
    {"prediction": 0, "probability": 0.101, "label": "negative"},
    {"prediction": 1, "probability": 0.785, "label": "positive"}
  ]
}
```

### Input Validation and Errors

All endpoints validate incoming requests and return HTTP 400 with a descriptive error message when input is invalid. Reasons for rejection include missing required fields, wrong data types (for example, a string where a number is expected), negative values on fields that should be non-negative, and unrecognized values in categorical fields (product_category, seller_state, payment_type).

Example error response:
```json
{
  "error": "Invalid input",
  "details": {
    "total_order_value": "must be a number",
    "seller_state": "unknown value: 'ZZ'"
  }
}
```

## Input Schema

All 14 features are required for every prediction request.

| Feature | Type | Valid Range / Values |
|---|---|---|
| delivery_days | number | >= 0. Days between purchase and delivery. |
| delivery_vs_estimated | number | Any number. Negative means delivered early, positive means late. |
| number_of_items | number | >= 0. Number of items in the order. |
| total_order_value | number | >= 0. Total price in Brazilian reais. |
| avg_product_price | number | >= 0. Average price per item. |
| max_product_price | number | >= 0. Max item price in the order. |
| min_product_price | number | >= 0. Min item price in the order. |
| total_freight_value | number | >= 0. Total shipping cost. |
| avg_freight_value | number | >= 0. Average shipping cost per item. |
| max_freight_value | number | >= 0. Max shipping cost in the order. |
| min_product_photos_qty | number | >= 0. Minimum number of photos on a product listing. |
| product_category | string | One of 74 known categories (e.g., housewares, electronics, health_beauty, stationery, auto). See valid_categories.json in the repo for the full list. |
| seller_state | string | 2-letter Brazilian state code. One of 24 known values (e.g., SP, RJ, MG, PR). |
| payment_type | string | One of 5 values: boleto, credit_card, debit_card, voucher, not_defined. |

## Local Setup

### Prerequisites

- Python 3.11
- Docker Desktop (optional, for containerized setup)
- Git

### Clone the repo

```
git clone https://github.com/valeriebien-aime-cmyk/hw4-mlops.git
cd hw4-mlops
```

### Option 1: Run locally with Python

```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The API will be available at `http://127.0.0.1:5001`. This uses Flask's development server on port 5001 (not 5000, which is often taken by macOS AirPlay).

In a separate terminal, run the test suite:

```
source venv/bin/activate
python test_api.py --url http://127.0.0.1:5001
```

All 5 tests should pass.

### Option 2: Run with Docker

```
docker build -t hw4-api .
docker run -p 5002:5000 hw4-api
```

The API will be available at `http://127.0.0.1:5002` (the container listens on port 5000 internally, mapped to 5002 on your host to avoid AirPlay conflicts).

Run the test suite against the container:

```
python test_api.py --url http://127.0.0.1:5002
```

## Model Information

**Deployed Model:** Gradient Boosting Classifier (scikit-learn implementation)

**Training data:** Olist Brazilian e-commerce dataset, 98,673 orders with reviews. Split 80/20 train/test with `random_state=42`, stratified on the target.

**Features:** 14 order-level features (11 numeric, 3 categorical). See the Input Schema section above.

**Target:** Binary positive review (4-5 stars = 1) vs negative review (1-3 stars = 0).

**Hyperparameters:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `random_state=42`.

**Performance on held-out test set (19,735 orders):**

| Metric | Value |
|---|---|
| Accuracy | 0.8136 |
| ROC AUC | 0.7290 |
| Positive class recall | 0.98 |
| Negative class recall | 0.25 |
| F1 (positive class) | 0.89 |

### Known Limitations

The model is really good at catching happy customers (98% recall on positives) but misses most unhappy ones (only 25% recall on negatives). This is because the training data is 77% positive, so the model leans toward predicting positive when it is uncertain. Since the whole point of using this API is to flag negative reviews before they happen, and since a negative review costs a business way more than a false alarm on a positive one, users should lower the decision threshold below 0.5 to catch more negatives. The tradeoff is more false positives, but that is usually worth it.

The model was trained on historical Olist data from 2016 to 2018. It has not been tested on recent data, so performance in production today may be different. The Part 5 monitoring notebook simulates 6 months of drift and shows that a 6 percentage point drop in accuracy is possible if the model is left alone without retraining.

The 14 features used are all order level information available around the time of purchase. The model does not read review text (that is what the foundation model in Part 1 does). Once a customer writes a review, this model has nothing else to add.