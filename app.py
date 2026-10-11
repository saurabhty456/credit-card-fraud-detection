"""
=============================================================================
FRAUDGUARD - Credit Card Fraud Detection Web Application
File: app.py
Author: Senior ML & Full-Stack Developer / College Project Mentor
Description:
    Flask backend application serving the modern FraudGuard interface.
    Handles data validation, feature scaling, model inference, error
    handling, and educational dashboards for college project demonstrations.
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "fraudguard_college_project_secret_key"

# Paths to models and metadata
MODEL_PATH = os.path.join("models", "fraud_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
PIPELINE_PATH = os.path.join("models", "pipeline.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")
EDA_SUMMARY_PATH = os.path.join("reports", "eda_summary.json")
SAMPLES_PATH = os.path.join("models", "sample_transactions.json")

# Global containers for loaded artifacts
model = None
scaler = None
model_metrics = {}
eda_summary = {}
sample_transactions = {}

# Expected 30 feature names in exact order
EXPECTED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

def load_artifacts():
    """Loads trained model, scaler, and evaluation metrics with graceful fallback."""
    global model, scaler, model_metrics, eda_summary, sample_transactions
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("[INFO] Model and Scaler loaded successfully.")
        else:
            print("[WARNING] Model or Scaler file missing. Please run train_model.py first.")

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                model_metrics = json.load(f)

        if os.path.exists(EDA_SUMMARY_PATH):
            with open(EDA_SUMMARY_PATH, "r") as f:
                eda_summary = json.load(f)

        if os.path.exists(SAMPLES_PATH):
            with open(SAMPLES_PATH, "r") as f:
                sample_transactions = json.load(f)

    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}")

# Initial load
load_artifacts()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Home Page: Overview of FraudGuard, live statistics, and quick navigation."""
    return render_template(
        "index.html",
        eda_summary=eda_summary,
        model_metrics=model_metrics,
        expected_features=EXPECTED_FEATURES
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Transaction Analysis Page:
    GET: Displays the input form with Mode 1 (Real Test Samples) and Mode 2 (Manual Input).
    POST: Validates input, scales features, runs inference, and renders the result page.
    """
    if request.method == "GET":
        return render_template(
            "predict.html",
            expected_features=EXPECTED_FEATURES,
            sample_transactions=sample_transactions,
            model_loaded=(model is not None and scaler is not None)
        )

    # POST Handling
    if model is None or scaler is None:
        flash("Model or Scaler not loaded on server. Please train the model first.", "error")
        return redirect(url_for("predict"))

    try:
        # Collect and validate feature values
        input_data = {}
        missing_fields = []
        invalid_fields = []

        # Check if request came as JSON or Form
        data_source = request.get_json(silent=True) if request.is_json else request.form

        for feat in EXPECTED_FEATURES:
            val_str = data_source.get(feat, "").strip() if isinstance(data_source.get(feat), str) else data_source.get(feat)
            if val_str is None or val_str == "":
                missing_fields.append(feat)
            else:
                try:
                    val_float = float(val_str)
                    input_data[feat] = val_float
                except (ValueError, TypeError):
                    invalid_fields.append(feat)

        if missing_fields:
            error_msg = f"Missing input fields: {', '.join(missing_fields[:5])}" + ("..." if len(missing_fields) > 5 else "")
            if request.is_json:
                return jsonify({"status": "error", "message": error_msg, "missing": missing_fields}), 400
            flash(error_msg, "error")
            return redirect(url_for("predict"))

        if invalid_fields:
            error_msg = f"Non-numeric values in fields: {', '.join(invalid_fields[:5])}"
            if request.is_json:
                return jsonify({"status": "error", "message": error_msg, "invalid": invalid_fields}), 400
            flash(error_msg, "error")
            return redirect(url_for("predict"))

        # Build DataFrame with explicit feature names to match StandardScaler training metadata
        feature_df = pd.DataFrame([[input_data[feat] for feat in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)

        # Scale features using the trained StandardScaler
        feature_vector_scaled = scaler.transform(feature_df)

        # Run prediction
        prediction_class = int(model.predict(feature_vector_scaled)[0])
        probabilities = model.predict_proba(feature_vector_scaled)[0]
        legit_prob = round(float(probabilities[0]) * 100, 2)
        fraud_prob = round(float(probabilities[1]) * 100, 2)

        # Formulate result dictionary
        result = {
            "prediction": prediction_class,
            "status_title": "POTENTIAL FRAUDULENT TRANSACTION" if prediction_class == 1 else "LEGITIMATE TRANSACTION",
            "risk_level": "HIGH" if prediction_class == 1 else "LOW",
            "risk_class": "danger" if prediction_class == 1 else "success",
            "message": (
                "The machine-learning model classified this transaction as potentially fraudulent."
                if prediction_class == 1
                else "The machine-learning model classified this transaction as legitimate."
            ),
            "fraud_probability": fraud_prob,
            "legit_probability": legit_prob,
            "amount": input_data.get("Amount", 0.0),
            "time": input_data.get("Time", 0.0),
            "raw_features": input_data,
            "model_name": model_metrics.get("selected_model", "Random Forest Classifier")
        }

        if request.is_json:
            return jsonify({"status": "success", "data": result}), 200

        return render_template("result.html", result=result)

    except Exception as e:
        error_msg = f"An unexpected error occurred during prediction: {str(e)}"
        print(f"[ERROR] Predict route exception: {e}")
        if request.is_json:
            return jsonify({"status": "error", "message": error_msg}), 500
        flash("An error occurred while processing your transaction. Please verify all feature values.", "error")
        return redirect(url_for("predict"))

@app.route("/dashboard")
def dashboard():
    """Dashboard Page: Comprehensive data insights, model evaluation metrics, and comparison plots."""
    return render_template(
        "dashboard.html",
        eda_summary=eda_summary,
        model_metrics=model_metrics
    )

@app.route("/api/samples")
def api_samples():
    """Helper API endpoint to provide real test set samples for frontend population."""
    return jsonify(sample_transactions)

@app.route("/api/metrics")
def api_metrics():
    """Helper API endpoint returning actual evaluation metrics."""
    return jsonify({
        "eda_summary": eda_summary,
        "model_metrics": model_metrics
    })

# ---------------------------------------------------------------------------
# Custom Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", error_banner="Page not found (404). Redirected to Home."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("index.html", error_banner="Internal server error encountered. Please check logs."), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ["1", "true", "True"]
    print("\n" + "=" * 60)
    print(f"FraudGuard Web Server Running on port {port}...")
    print(f"Access application at: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
