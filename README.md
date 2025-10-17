# Music Genre Classification

A compact music-genre classification project that extracts audio features from songs, trains machine learning models, and provides tools and a Streamlit web app to predict genres for new audio files.

---

## What this project does

- Extracts audio features (MFCC, chroma, spectral contrast, tonnetz, zero-crossing rate, centroid/bandwidth/rolloff) from audio files using [`src/feature_extraction.py`](src/feature_extraction.py) and related extraction helpers (also used in [`src/predict.py`](src/predict.py), [`src/predict_simple.py`](src/predict_simple.py), and `app_streamlit.py`).
- Preprocesses features (label encoding + scaling) with [`src/data_preprocessing.py`](src/data_preprocessing.py).
- Trains classifiers (Random Forest by default, plus KNN/XGBoost/Neural Net implementations) in [`src/models/train.py`](src/models/train.py) and model modules: [`src/models/random_forest_model.py`](src/models/random_forest_model.py), [`src/models/knn_model.py`](src/models/knn_model.py), [`src/models/neural_net_model.py`](src/models/neural_net_model.py).
- Evaluates models using accuracy, classification report and confusion matrix via [`src/evaluate.py`](src/evaluate.py).
- Provides CLI prediction scripts [`src/predict.py`](src/predict.py) and [`src/predict_simple.py`](src/predict_simple.py).
- Serves a web interface with Streamlit in [app_streamlit.py](app_streamlit.py) (model loader: [`load_model`](app_streamlit.py), web prediction: [`predict_genre`](app_streamlit.py), UI feature extraction: [`extract_features`](app_streamlit.py)).

---

## How it works — pipeline (step by step)

1. Data
   - Raw audio organized in `data/raw/genres_original` (default dataset path used by [`src/feature_extraction.py`](src/feature_extraction.py)).
   - Processed CSV saved to `data/processed/features.csv` (see `OUTPUT_FILE` in [`src/feature_extraction.py`](src/feature_extraction.py)).

2. Feature extraction
   - Run the extractor in [`src/feature_extraction.py`](src/feature_extraction.py). For each audio file it:
     - Loads the first 30 seconds with librosa.
     - Computes MFCC (mean & std), chroma (mean & std), spectral contrast (mean & std), tonnetz (mean & std), zero-crossing rate (mean & std), spectral centroid, bandwidth and rolloff (mean & std).
   - Result: a fixed-length numeric feature vector per audio file (the project documents ~84 features in [`WEB_APP_GUIDE.md`](WEB_APP_GUIDE.md)).

3. Preprocessing
   - [`src/data_preprocessing.py`](src/data_preprocessing.py) reads `data/processed/features.csv` and:
     - Separates X/y.
     - Encodes labels with `LabelEncoder` (see `encoder.classes_` usage).
     - Scales features with `StandardScaler`.
     - Splits into train/test using `train_test_split` (stratified, test_size=0.2, random_state=42).
     - Saves split CSVs to `data/processed/` for downstream training.

4. Training
   - Default quick-train script: [`src/models/train.py`](src/models/train.py) — trains a `RandomForestClassifier` and saves `models/random_forest.pkl`.
   - Modular model implementations are in:
     - [`src/models/random_forest_model.py`](src/models/random_forest_model.py) — includes hyperparameter tuning with `GridSearchCV` and saves model + artifacts.
     - [`src/models/knn_model.py`](src/models/knn_model.py) — KNN training and saving.
     - [`src/models/neural_net_model.py`](src/models/neural_net_model.py) — small Keras MLP that saves to `models/neural_net.h5`.
   - Saved artifacts: models (e.g., `models/random_forest.pkl`), optionally `models/label_encoder.pkl`, `models/scaler.pkl`, and `models/feature_names.pkl`.

5. Evaluation
   - Use [`src/evaluate.py`](src/evaluate.py) to run evaluation on a test CSV with a saved model to print accuracy, classification report and confusion matrix. If a label encoder exists, it inverse-transforms numeric predictions back to human-readable labels.

6. Prediction (CLI & Web)
   - CLI: [`src/predict.py`](src/predict.py) and [`src/predict_simple.py`](src/predict_simple.py) load saved models (`models/random_forest.pkl` by default), extract features from a supplied audio file, run predict / predict_proba, and display top-N genre probabilities.
   - Web: [app_streamlit.py](app_streamlit.py)
     - Cached model loader: [`load_model`](app_streamlit.py).
     - Cached feature extractor: [`extract_features`](app_streamlit.py).
     - Web predictor: [`predict_genre`](app_streamlit.py) — saves uploaded file temporarily, extracts features, runs model.predict and model.predict_proba (if available), and displays results in the UI.

---

## Quickstart

1. Install dependencies:
```sh
pip install -r requirements.txt
```
(see [requirements.txt](requirements.txt))

2. Extract features (from raw dataset):
```sh
python src/feature_extraction.py
```
(creates `data/processed/features.csv` — see [`src/feature_extraction.py`](src/feature_extraction.py))

3. Preprocess:
```sh
python src/data_preprocessing.py
```
(creates train/test CSVs in `data/processed/` — see [`src/data_preprocessing.py`](src/data_preprocessing.py))

4. Train:
```sh
python src/models/train.py
```
(saves model to `models/random_forest.pkl` — see [`src/models/train.py`](src/models/train.py) and [`src/models/random_forest_model.py`](src/models/random_forest_model.py))

5. Run web app:
```sh
streamlit run app_streamlit.py
```
(Open at http://localhost:8501; guide: [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md))

6. Or use CLI prediction:
```sh
python src/predict.py --audio "path/to/song.wav"
# or
python src/predict_simple.py
```

---

## Key files & symbols

- [app_streamlit.py](app_streamlit.py) — Streamlit UI; see [`load_model`](app_streamlit.py), [`extract_features`](app_streamlit.py), [`predict_genre`](app_streamlit.py).
- [src/feature_extraction.py](src/feature_extraction.py) — main extraction logic: [`extract_features`](src/feature_extraction.py).
- [src/data_preprocessing.py](src/data_preprocessing.py) — preprocessing pipeline: [`preprocess`](src/data_preprocessing.py).
- [src/models/train.py](src/models/train.py) — example training flow that saves `models/random_forest.pkl`.
- [src/models/random_forest_model.py](src/models/random_forest_model.py) — RF trainer and tuning: [`train_random_forest`](src/models/random_forest_model.py).
- [src/models/neural_net_model.py](src/models/neural_net_model.py) — neural net trainer: [`train_neural_net`](src/models/neural_net_model.py).
- [src/models/knn_model.py](src/models/knn_model.py) — KNN trainer: [`train_knn`](src/models/knn_model.py).
- [src/predict.py](src/predict.py) and [src/predict_simple.py](src/predict_simple.py) — CLI prediction scripts.
- [src/evaluate.py](src/evaluate.py) — evaluation runner.
- [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) — web app usage & troubleshooting.
- [notebooks/](notebooks/) — Jupyter notebooks for exploration: `01_data_exploration.ipynb`, `02_feature_extraction.ipynb`, `03_model_training.ipynb`.

---

## Tips & troubleshooting

- "Model file not found": train first (`python src/models/train.py`) or ensure `models/random_forest.pkl` exists (see [src/models/train.py](src/models/train.py)).
- "Error extracting features": ensure file is valid (WAV preferred) or convert your file to WAV (see feature extraction code in [`src/feature_extraction.py`](src/feature_extraction.py)).
- First web prediction is slower due to model loading and caching — subsequent predictions are faster (see caching in [app_streamlit.py](app_streamlit.py)).
- If you modify features, regenerate `data/processed/features.csv` and retrain to keep model / feature alignment.

---

## License & notes

- This repo is a compact educational pipeline for audio-based genre classification. See individual source files for implementation details.
- For a user guide and troubleshooting, consult [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md).

Happy experimenting — upload a 30s clip and try it out!
