# src/predict.py

import argparse
import pandas as pd
import joblib
import os

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Make predictions with trained models")
parser.add_argument("--input", required=True, help="Path to input CSV with features")
parser.add_argument("--model", required=True, help="Path to trained model file (.pkl or .h5)")
args = parser.parse_args()

# --- Load Input Features ---
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

X_new = pd.read_csv(args.input)

# Remove label column if present
if 'label' in X_new.columns:
    X_new = X_new.drop(columns=['label'])

# --- Load Model ---
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")

model = joblib.load(args.model)

X_new = pd.read_csv(args.input)

# Ensure columns match training (drop names, keep only values)
X_new = X_new.values

# --- Make Predictions ---
predictions = model.predict(X_new)

print("✅ Predictions:")
for i, pred in enumerate(predictions, 1):
    print(f"Sample {i}: {pred}")
