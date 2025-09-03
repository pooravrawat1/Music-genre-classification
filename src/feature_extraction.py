import os
import librosa
import numpy as np
import pandas as pd

# Paths
DATASET_PATH = "data/raw/genres"
OUTPUT_PATH = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "features.csv")

def extract_features(file_path, n_mfcc=20):
    """
    Extract MFCC features from an audio file.
    Returns a vector of size n_mfcc (mean across time).
    """
    try:
        y, sr = librosa.load(file_path, duration=30)  # load 30 sec clip
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = np.mean(mfcc, axis=1)  # mean across time frames
        return mfcc_means
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
    columns = ["label"] + [f"mfcc_{i+1}" for i in range(20)]
    df = pd.DataFrame(data, columns=columns)

    # Save CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Features saved to {OUTPUT_FILE}")
    print(f"Shape of dataset: {df.shape}")

if __name__ == "__main__":
    main()

