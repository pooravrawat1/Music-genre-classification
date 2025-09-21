import argparse
import joblib
import os
import sys
import numpy as np
import librosa
from pathlib import Path

def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="🎵 Music Genre Classifier - Predict genre from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict.py --audio "path/to/song.wav"
  python src/predict.py --audio "path/to/song.wav" --model "models/custom_model.pkl"
  python src/predict.py --audio "path/to/song.wav" --top 5
        """
    )
    parser.add_argument("--audio", required=True, help="Path to input audio file (.wav, .mp3)")
    parser.add_argument("--model", default="models/random_forest.pkl", help="Path to trained model file (default: models/random_forest.pkl)")
    parser.add_argument("--top", type=int, default=3, help="Show top N predictions with confidence scores (default: 3)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.audio):
        print(f"❌ =Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print("💡 Make sure you've trained a model first by running: python src/models/train.py")
        sys.exit(1)

    # --- Load Audio & Extract Features ---
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
            sys.exit(1)

    # Extract features from audio
    X_new = extract_features(args.audio)

    # --- Load Model ---
    try:
        print(f"🤖 Loading model: {Path(args.model).name}")
        model = joblib.load(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)

    # --- Make Prediction ---
    try:
        print("🔮 Making prediction...")
        
        # Get prediction
        prediction = model.predict(X_new)[0]
        
        # Get prediction probabilities for confidence scores
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new)[0]
        else:
            print("⚠️  Model doesn't support probability predictions")
            probabilities = None

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
        print(f"🎵 Audio File: {Path(args.audio).name}")
        print(f"🏆 Predicted Genre: {predicted_genre}")
        
        # Show confidence scores if available
        if probabilities is not None and genre_names is not None:
            print(f"\n📊 Top {min(args.top, len(genre_names))} Predictions:")
            print("-" * 30)
            
            # Sort by probability and show top N
            genre_probs = list(zip(genre_names, probabilities))
            genre_probs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (genre, prob) in enumerate(genre_probs[:args.top]):
                confidence = prob * 100
                bar_length = int(confidence / 5)  # Scale to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"{i+1:2d}. {genre:12s} {bar} {confidence:6.2f}%")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()