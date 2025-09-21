#!/usr/bin/env python3
"""
Simple Music Genre Predictor
Just run this script and it will ask for the audio file path.
"""

import joblib
import os
import sys
import numpy as np
import librosa
from pathlib import Path

def extract_features(file_path, sr=22050):
    """Extract audio features for genre classification"""
    try:
        print(f"🎵 Loading audio file: {Path(file_path).name}")
        y, sr = librosa.load(file_path, sr=sr, duration=30)
        
        print("🔄 Extracting features...")
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

        print(f"✅ Extracted {len(features)} features")
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        print(f"❌ Error extracting features: {str(e)}")
        return None

def predict_genre(audio_path, model_path="models/random_forest.pkl"):
    """Predict genre from audio file"""
    
    # Validate inputs
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print("💡 Make sure you've trained a model first by running: python src/models/train.py")
        return None

    # Extract features
    features = extract_features(audio_path)
    if features is None:
        return None

    # Load model
    try:
        print(f"🤖 Loading model: {Path(model_path).name}")
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

    # Make prediction
    try:
        print("🔮 Making prediction...")
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities for confidence scores
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]

        # Load label encoder for genre names
        encoder_path = "models/label_encoder.pkl"
        genre_names = None
        
        if os.path.exists(encoder_path):
            try:
                encoder = joblib.load(encoder_path)
                genre_names = encoder.classes_
                predicted_genre = encoder.inverse_transform([prediction])[0]
            except Exception as e:
                print(f"⚠️  Warning: Could not load label encoder: {str(e)}")
                predicted_genre = f"Genre_{prediction}"
        else:
            predicted_genre = f"Genre_{prediction}"
            print("⚠️  Label encoder not found. Showing numeric prediction.")

        # Display results
        print("\n" + "="*50)
        print("🎯 PREDICTION RESULTS")
        print("="*50)
        print(f"🎵 Audio File: {Path(audio_path).name}")
        print(f"🏆 Predicted Genre: {predicted_genre}")
        
        # Show confidence scores if available
        if probabilities is not None and genre_names is not None:
            print("\n📊 Top 5 Predictions:")
            print("-" * 30)
            
            # Sort by probability and show top 5
            genre_probs = list(zip(genre_names, probabilities))
            genre_probs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (genre, prob) in enumerate(genre_probs[:5]):
                confidence = prob * 100
                bar_length = int(confidence / 5)  # Scale to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"{i+1:2d}. {genre:12s} {bar} {confidence:6.2f}%")
        
        print("="*50)
        return predicted_genre
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return None

def main():
    """Interactive main function"""
    print("🎵 Music Genre Classifier")
    print("=" * 30)
    
    while True:
        # Get audio file path from user
        audio_path = input("\n📁 Enter path to audio file (or 'quit' to exit): ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        # Remove quotes if user added them
        audio_path = audio_path.strip('"\'')
        
        if not audio_path:
            print("⚠️  Please enter a valid file path")
            continue
        
        # Predict genre
        result = predict_genre(audio_path)
        
        if result:
            print(f"\n🎉 Successfully predicted genre: {result}")
        
        # Ask if user wants to predict another file
        again = input("\n🔄 Predict another file? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()