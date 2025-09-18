# src/evaluate.py

import argparse
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Evaluate a trained model")
parser.add_argument("--test", required=True, help="Path to test CSV file with features + labels")
parser.add_argument("--model", required=True, help="Path to trained model file (.pkl or .h5)")
args = parser.parse_args()

# --- Load Test Data ---
if not os.path.exists(args.test):
    raise FileNotFoundError(f"Test file not found: {args.test}")

df = pd.read_csv(args.test)

if "label" not in df.columns:
    raise ValueError("Test CSV must include a 'label' column")

X_test = df.drop(columns=["label"]).values
y_test = df["label"].values

# --- Load Model ---
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")

model = joblib.load(args.model)

# --- Make Predictions ---
y_pred = model.predict(X_test)

# --- Decode Labels if Encoder Exists ---
encoder_path = "models/label_encoder.pkl"
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)
    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)

# --- Evaluation Metrics ---
print("✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

