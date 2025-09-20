# src/predict.py

import argparse
import joblib
import os
import numpy as np
import librosa

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Predict genre from an audio file")
parser.add_argument("--audio", required=True, help="Path to input audio file (.wav)")
parser.add_argument("--model", required=True, help="Path to trained model file (.pkl or .h5)")
args = parser.parse_args()

# --- Load Audio & Extract Features ---
def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)

    # Example features (same as in feature_extraction.py)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Flatten all features into one vector
    features = np.hstack([mfcc_mean, mfcc_std, chroma_mean, chroma_std])
    return features.reshape(1, -1)  # 2D for sklearn

print("🎵 Extracting features...")
X_new = extract_features(args.audio)

# --- Load Model ---
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")

model = joblib.load(args.model)

# --- Predict ---
predictions = model.predict(X_new)

# Load label encoder (if available)
encoder_path = "models/label_encoder.pkl"
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)
    predicted_labels = encoder.inverse_transform(predictions)
    print(f"✅ Predicted Genre: {predicted_labels[0]}")
else:
    print(f"✅ Predicted Genre (numeric): {predictions[0]}")
