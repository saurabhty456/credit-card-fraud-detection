"""
=============================================================================
FRAUDGUARD - End-to-End Test Suite
File: test_app.py
Author: Senior ML & Full-Stack Developer / College Project Mentor
Description:
    Comprehensive verification script testing:
    1. Dataset accessibility and metrics
    2. Serialized model and scaler integrity
    3. Flask route response codes (Home, Predict, Dashboard, APIs)
    4. Prediction on Legitimate transaction sample
    5. Prediction on Fraudulent transaction sample
    6. Form error handling (empty fields, non-numeric input)
=============================================================================
"""

import os
import json
import numpy as np
from app import app, model, scaler, EXPECTED_FEATURES

def run_tests():
    print("=" * 70)
    print("PHASE 11: END-TO-END APPLICATION AND MODEL TESTING")
    print("=" * 70)

    # Test 1: Verify Artifact Files on Disk
    print("\n[TEST 1] Verifying Model, Scaler, and Metrics Artifacts...")
    required_files = [
        "models/fraud_model.pkl",
        "models/scaler.pkl",
        "models/model_metrics.json",
        "models/sample_transactions.json",
        "reports/eda_summary.json",
        "static/img/plots/confusion_matrices.png",
        "static/img/plots/roc_curves.png",
        "static/img/plots/class_distribution.png"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"
        print(f"  [OK] {rf} exists ({os.path.getsize(rf):,} bytes)")
    print(">> TEST 1 PASSED: All artifact files are present.")

    # Test 2: Model and Scaler Memory Check
    print("\n[TEST 2] Verifying Model and Scaler in Memory...")
    assert model is not None, "Model object is None!"
    assert scaler is not None, "Scaler object is None!"
    import pandas as pd
    dummy_input = pd.DataFrame(np.zeros((1, 30)), columns=EXPECTED_FEATURES)
    dummy_scaled = scaler.transform(dummy_input)
    assert dummy_scaled.shape == (1, 30), "Transformed dummy input shape mismatch!"
    pred = model.predict(dummy_scaled)
    prob = model.predict_proba(dummy_scaled)
    print(f"  [OK] Scaler transform successful. Model predict output: {pred}, Proba shape: {prob.shape}")
    print(">> TEST 2 PASSED: Model & Scaler inference functioning.")

    # Initialize Flask test client
    client = app.test_client()

    # Test 3: Home Page Route
    print("\n[TEST 3] Testing GET / (Home Page)...")
    res_home = client.get("/")
    assert res_home.status_code == 200, f"Expected 200, got {res_home.status_code}"
    assert b"FRAUDGUARD" in res_home.data
    assert b"284,807" in res_home.data or b"284807" in res_home.data
    print("  [OK] GET / returned HTTP 200 with project title and dataset statistics.")
    print(">> TEST 3 PASSED.")

    # Test 4: Dashboard Route
    print("\n[TEST 4] Testing GET /dashboard...")
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200, f"Expected 200, got {res_dash.status_code}"
    assert b"Model Comparison" in res_dash.data
    assert b"Random Forest" in res_dash.data
    assert b"confusion_matrices.png" in res_dash.data
    print("  [OK] GET /dashboard returned HTTP 200 with comparison table and plot references.")
    print(">> TEST 4 PASSED.")

    # Test 5: Predict Page GET Route
    print("\n[TEST 5] Testing GET /predict...")
    res_pred = client.get("/predict")
    assert res_pred.status_code == 200, f"Expected 200, got {res_pred.status_code}"
    assert b"Mode 1: Pre-fill Real Test Set Samples" in res_pred.data
    assert b"ANALYZE TRANSACTION" in res_pred.data
    print("  [OK] GET /predict returned HTTP 200 with transaction analysis form.")
    print(">> TEST 5 PASSED.")

    # Load Real Samples for Testing Predictions
    with open("models/sample_transactions.json", "r") as f:
        samples = json.load(f)

    legit_sample = samples["legitimate_samples"][0]["features"]
    fraud_sample = samples["fraud_samples"][0]["features"]

    # Test 6: POST /predict with Legitimate Sample
    print("\n[TEST 6] Testing POST /predict with Real Legitimate Transaction Sample...")
    res_legit = client.post("/predict", data=legit_sample, follow_redirects=True)
    assert res_legit.status_code == 200, f"Expected 200, got {res_legit.status_code}"
    assert b"LEGITIMATE TRANSACTION" in res_legit.data, "Expected LEGITIMATE TRANSACTION banner!"
    assert b"LOW" in res_legit.data, "Expected LOW risk level!"
    assert b"classified this transaction as legitimate" in res_legit.data
    print("  [OK] Correctly predicted Class 0: 'LEGITIMATE TRANSACTION', Risk Level: LOW.")
    print(">> TEST 6 PASSED.")

    # Test 7: POST /predict with Fraudulent Sample
    print("\n[TEST 7] Testing POST /predict with Real Fraudulent Transaction Sample...")
    res_fraud = client.post("/predict", data=fraud_sample, follow_redirects=True)
    assert res_fraud.status_code == 200, f"Expected 200, got {res_fraud.status_code}"
    assert b"POTENTIAL FRAUDULENT TRANSACTION" in res_fraud.data, "Expected FRAUDULENT TRANSACTION banner!"
    assert b"HIGH" in res_fraud.data, "Expected HIGH risk level!"
    assert b"classified this transaction as potentially fraudulent" in res_fraud.data
    print("  [OK] Correctly predicted Class 1: 'POTENTIAL FRAUDULENT TRANSACTION', Risk Level: HIGH.")
    print(">> TEST 7 PASSED.")

    # Test 8: Input Validation - Missing Fields Handling
    print("\n[TEST 8] Testing POST /predict with Missing Feature Fields...")
    incomplete_data = legit_sample.copy()
    del incomplete_data['Amount']
    del incomplete_data['V5']
    res_missing = client.post("/predict", data=incomplete_data, follow_redirects=True)
    assert res_missing.status_code == 200
    assert b"Missing input fields" in res_missing.data
    print("  [OK] Gracefully handled missing fields with a friendly alert banner.")
    print(">> TEST 8 PASSED.")

    # Test 9: Input Validation - Non-numeric Fields Handling
    print("\n[TEST 9] Testing POST /predict with Invalid Non-Numeric Fields...")
    invalid_data = legit_sample.copy()
    invalid_data['Amount'] = "invalid_text_input"
    res_invalid = client.post("/predict", data=invalid_data, follow_redirects=True)
    assert res_invalid.status_code == 200
    assert b"Non-numeric values in fields" in res_invalid.data
    print("  [OK] Gracefully caught non-numeric input without raising unhandled 500 error.")
    print(">> TEST 9 PASSED.")

    # Test 10: JSON API Endpoints
    print("\n[TEST 10] Testing REST API Endpoints (/api/samples & /api/metrics)...")
    res_api_samples = client.get("/api/samples")
    assert res_api_samples.status_code == 200
    assert "legitimate_samples" in res_api_samples.get_json()

    res_api_metrics = client.get("/api/metrics")
    assert res_api_metrics.status_code == 200
    assert "model_metrics" in res_api_metrics.get_json()
    print("  [OK] Both API helper endpoints returned valid JSON data.")
    print(">> TEST 10 PASSED.")

    print("\n" + "=" * 70)
    print("ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
