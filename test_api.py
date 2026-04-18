# test_api.py - automated tests for the hw4 flask api
# usage:
#   python test_api.py                              # default: http://127.0.0.1:5001 (local flask)
#   python test_api.py --url http://127.0.0.1:5002  # local docker container
#   python test_api.py --url https://your-app.onrender.com  # deployed on render

import argparse
import requests
import sys

# a complete valid order, used as the baseline for tests
VALID_ORDER = {
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
}


# ---- test infrastructure ----

tests_run = 0
tests_passed = 0

def run_test(name, test_fn, base_url):
    global tests_run, tests_passed
    tests_run += 1
    print(f"\n--- Test {tests_run}: {name} ---")
    try:
        test_fn(base_url)
        tests_passed += 1
        print(f"PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


# ---- the five tests ----

def test_health_check(base_url):
    """GET /health returns expected fields"""
    response = requests.get(f"{base_url}/health", timeout=30)
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    assert response.status_code == 200, f"expected 200, got {response.status_code}"
    body = response.json()
    assert body.get("status") == "healthy", "expected status=healthy"
    assert body.get("model") == "loaded", "expected model=loaded"


def test_valid_single_prediction(base_url):
    """POST /predict with valid data returns prediction, probability, label"""
    response = requests.post(f"{base_url}/predict", json=VALID_ORDER, timeout=30)
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    assert response.status_code == 200, f"expected 200, got {response.status_code}"
    body = response.json()
    assert "prediction" in body, "response missing 'prediction' field"
    assert "probability" in body, "response missing 'probability' field"
    assert "label" in body, "response missing 'label' field"
    assert body["prediction"] in (0, 1), f"prediction must be 0 or 1, got {body['prediction']}"
    assert 0.0 <= body["probability"] <= 1.0, f"probability out of range: {body['probability']}"
    assert body["label"] in ("positive", "negative"), f"unexpected label: {body['label']}"


def test_valid_batch_prediction(base_url):
    """POST /predict/batch with 5 records returns 5 predictions"""
    batch = [VALID_ORDER.copy() for _ in range(5)]
    for i, record in enumerate(batch):
        record["delivery_days"] = 5.0 + i * 3.0

    response = requests.post(f"{base_url}/predict/batch", json=batch, timeout=30)
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    assert response.status_code == 200, f"expected 200, got {response.status_code}"
    body = response.json()
    assert "predictions" in body, "response missing 'predictions' field"
    assert len(body["predictions"]) == 5, f"expected 5 predictions, got {len(body['predictions'])}"
    for i, pred in enumerate(body["predictions"]):
        assert "prediction" in pred, f"prediction {i} missing 'prediction' field"
        assert "probability" in pred, f"prediction {i} missing 'probability' field"
        assert "label" in pred, f"prediction {i} missing 'label' field"


def test_missing_field_returns_400(base_url):
    """missing required field returns 400 with helpful message"""
    bad_order = VALID_ORDER.copy()
    del bad_order["total_order_value"]

    response = requests.post(f"{base_url}/predict", json=bad_order, timeout=30)
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    assert response.status_code == 400, f"expected 400, got {response.status_code}"
    body = response.json()
    assert "error" in body, "response missing 'error' field"
    assert "details" in body, "response missing 'details' field"
    assert "total_order_value" in body["details"], "error details should name the missing field"


def test_invalid_type_returns_400(base_url):
    """string where number expected returns 400"""
    bad_order = VALID_ORDER.copy()
    bad_order["total_order_value"] = "not a number"

    response = requests.post(f"{base_url}/predict", json=bad_order, timeout=30)
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.json()}")
    assert response.status_code == 400, f"expected 400, got {response.status_code}"
    body = response.json()
    assert "error" in body, "response missing 'error' field"
    assert "details" in body, "response missing 'details' field"
    assert "total_order_value" in body["details"], "error details should name the bad field"


# ---- main ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the HW4 Flask API")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5001",
        help="Base URL of the API (default: http://127.0.0.1:5001)"
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(f"Running API tests against {base_url}")
    print("=" * 60)

    run_test("Health check endpoint", test_health_check, base_url)
    run_test("Valid single prediction", test_valid_single_prediction, base_url)
    run_test("Valid batch of 5 records", test_valid_batch_prediction, base_url)
    run_test("Missing field returns 400", test_missing_field_returns_400, base_url)
    run_test("Invalid type returns 400", test_invalid_type_returns_400, base_url)

    print("\n" + "=" * 60)
    print(f"Results: {tests_passed}/{tests_run} tests passed")

    if tests_passed == tests_run:
        print("All tests passed")
        sys.exit(0)
    else:
        print("Some tests failed")
        sys.exit(1)