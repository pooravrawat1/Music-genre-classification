# Music-genre-classification

## Feature Extraction with MFCCs

To classify music tracks into genres, we extract MFCC (Mel-Frequency Cepstral Coefficients) from each audio file. MFCCs are a compact representation of an audio signal that approximates how humans perceive sound. They capture the timbre and tonal characteristics of music, which differ across genres.
Used more features, counting upto 84

## How it works:

- Audio is divided into short frames (~20–40 ms).

- Each frame is transformed into a frequency spectrum.

- Frequencies are mapped to the Mel scale (perceptually relevant).

- The spectrum is summarized into 13–20 coefficients per frame.

- For each track, we aggregate the coefficients (mean over time) to create a single feature vector.

## Why MFCCs:

- Capture tonal textures unique to each genre.

- Reduce dimensionality compared to raw audio or spectrograms.

- Provide a robust input for machine learning models like Random Forests.

## Outcome:

## Model Comparison: Random Forest vs KNN

We compared Random Forest and K-Nearest Neighbors (KNN) classifiers for music genre classification. Random Forest achieved higher accuracy (0.64) compared to KNN (0.54). This improvement is likely due to Random Forest's ability to handle high-dimensional audio features more effectively than KNN.

Random Forest was therefore selected as the primary model for this project.

The resulting CSV (data/processed/features.csv) contains one row per track with 20 MFCC features plus the genre label, ready for training and evaluation.
