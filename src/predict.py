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
    y, sr = librosa.load(file_path, sr=sr, duration=30)
    features = []

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs.T, axis=0))
    features.extend(np.std(mfccs.T, axis=0))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma.T, axis=0))
    features.extend(np.std(chroma.T, axis=0))

    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(spec_contrast.T, axis=0))
    features.extend(np.std(spec_contrast.T, axis=0))

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features.extend(np.mean(tonnetz.T, axis=0))
    features.extend(np.std(tonnetz.T, axis=0))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Centroid / Bandwidth / Rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    for feat in [centroid, bandwidth, rolloff]:
        features.append(np.mean(feat))
        features.append(np.std(feat))

    return np.array(features).reshape(1, -1)

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
