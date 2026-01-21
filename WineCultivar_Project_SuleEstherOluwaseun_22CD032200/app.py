import json
import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load saved artifacts at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "wine_cultivar_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_order.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
imputer = joblib.load(IMPUTER_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)["features"]


def parse_float(value: str):
    """Strict float parsing for safety."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=FEATURES, prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    values = []
    missing_fields = []

    for feat in FEATURES:
        raw = request.form.get(feat, "").strip()
        val = parse_float(raw)

        if val is None:
            missing_fields.append(feat)
        else:
            values.append(val)

    if missing_fields:
        return render_template(
            "index.html",
            features=FEATURES,
            prediction=None,
            error=f"Invalid or missing value(s) for: {', '.join(missing_fields)}"
        )

    # shape = (1, 6)
    X = np.array(values, dtype=float).reshape(1, -1)

    # Apply the same preprocessing used during training
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    pred = int(model.predict(X_scaled)[0])  # 1, 2, or 3

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=f"Cultivar {pred}",
        error=None
    )


if __name__ == "__main__":
    # Render uses PORT env var
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
