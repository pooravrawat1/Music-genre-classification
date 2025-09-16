# src/predict.py

import argparse
import pandas as pd
import joblib
import os

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Make predictions with trained models")
parser.add_argument("--input", required=True, help="Path to input CSV with features")
parser.add_argument("--model", required=True, help="Path to trained model file (.pkl)")
parser.add_argument("--ytrue", help="(Optional) Path to CSV with true labels for comparison")
args = parser.parse_args()

# --- Load Input Features ---
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

X_new = pd.read_csv(args.input)

# Remove label column if present (e.g. features.csv)
if 'label' in X_new.columns:
    X_new = X_new.drop(columns=['label'])

# --- Load Model ---
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")

model = joblib.load(args.model)

# Convert features to NumPy array
X_new = X_new.values

# --- Make Predictions ---
predictions = model.predict(X_new)

# Load label encoder if available
encoder_path = "models/label_encoder.pkl"
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)
    predicted_labels = encoder.inverse_transform(predictions)
else:
    predicted_labels = predictions  # fallback to numeric labels

print("✅ Predictions:")
for i, label in enumerate(predicted_labels, 1):
    print(f"Sample {i}: {label}")

# --- Optional: Compare with ground truth ---
if args.ytrue and os.path.exists(args.ytrue):
    y_true = pd.read_csv(args.ytrue).squeeze()
    if os.path.exists(encoder_path):
        y_true = encoder.inverse_transform(y_true)

    print("\n📊 Comparison with Ground Truth:")
    for i, (pred, true) in enumerate(zip(predicted_labels, y_true), 1):
        print(f"Sample {i}: predicted={pred}, actual={true}")
