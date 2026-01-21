import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Only allowed feature names (from your assignment)
ALLOWED_FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

#Choose EXACTLY six (6) input features
FEATURES = [
    "alcohol",
    "malic_acid",
    "alcalinity_of_ash",
    "total_phenols",
    "flavanoids",
    "proline",
]

def main():
    # Safety check: ensure exactly 6 and in allowed list
    if len(FEATURES) != 6:
        raise ValueError("You must select EXACTLY six (6) features.")
    for f in FEATURES:
        if f not in ALLOWED_FEATURES:
            raise ValueError(f"Feature '{f}' is not allowed by the project instructions.")

    # 1) Load the Wine dataset
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()

    # Rename columns to exactly match assignment names
    rename_map = {
        "od280/od315_of_diluted_wines": "od280/od315_of_diluted_wines",  # same
        "alcalinity_of_ash": "alcalinity_of_ash",                        # same
        "malic_acid": "malic_acid",                                      # same
        "color_intensity": "color_intensity",                            # same
        "total_phenols": "total_phenols",                                # same
    }
    df.rename(columns=rename_map, inplace=True)

    # Target is cultivar (class label)
    df["cultivar"] = df["target"] + 1  # convert 0/1/2 into 1/2/3
    df.drop(columns=["target"], inplace=True)

    # 2) Preprocessing
    # Missing values handling (Wine dataset typically has none, but we still implement it)
    X = df[FEATURES].copy()
    y = df["cultivar"].copy()

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Impute missing values (if any)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 3) Feature scaling (mandatory)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 4) Train model (Logistic Regression)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro):    {rec:.4f}")
    print(f"F1-score (macro):  {f1:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6) Save artifacts
    out_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(out_dir, "wine_cultivar_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))

    with open(os.path.join(out_dir, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES}, f, indent=2)

    print("\n Saved:")
    print("- model/wine_cultivar_model.pkl")
    print("- model/scaler.pkl")
    print("- model/imputer.pkl")
    print("- model/feature_order.json")

if __name__ == "__main__":
    main()
