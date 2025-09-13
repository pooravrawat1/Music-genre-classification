import os
import librosa
import numpy as np
import pandas as pd

# Paths
DATASET_PATH = "data/raw/genres_original"
OUTPUT_PATH = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "features.csv")


def extract_features(file_path, n_mfcc=20):
    """
    Extract MFCC features from an audio file.
    Returns a vector of size n_mfcc (mean across time).
    """
    try:
        y, sr = librosa.load(file_path, duration=30)  # load 30 sec clip
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

        return np.array(features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    genres = os.listdir(DATASET_PATH)
    data = []

    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                features = extract_features(file_path)
                if features is not None:
                    row = [genre] + features.tolist()
                    data.append(row)

    # Create dataframe
    # Calculate the number of features for each type
    n_mfcc = 13
    n_chroma = 12
    n_spec_contrast = 7
    n_tonnetz = 6
    # 2 stats (mean, std) for each feature vector, except zcr, centroid, bandwidth, rolloff (which are 1D, so 2 each)
    columns = ["label"]
    columns += [f"mfcc_mean_{i+1}" for i in range(n_mfcc)]
    columns += [f"mfcc_std_{i+1}" for i in range(n_mfcc)]
    columns += [f"chroma_mean_{i+1}" for i in range(n_chroma)]
    columns += [f"chroma_std_{i+1}" for i in range(n_chroma)]
    columns += [f"spec_contrast_mean_{i+1}" for i in range(n_spec_contrast)]
    columns += [f"spec_contrast_std_{i+1}" for i in range(n_spec_contrast)]
    columns += [f"tonnetz_mean_{i+1}" for i in range(n_tonnetz)]
    columns += [f"tonnetz_std_{i+1}" for i in range(n_tonnetz)]
    columns += ["zcr_mean", "zcr_std"]
    columns += ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std", "rolloff_mean", "rolloff_std"]
    df = pd.DataFrame(data, columns=columns)

    # Save CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Features saved to {OUTPUT_FILE}")
    print(f"Shape of dataset: {df.shape}")

if __name__ == "__main__":
    main()

