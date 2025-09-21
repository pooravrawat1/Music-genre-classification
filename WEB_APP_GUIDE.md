# 🎵 Music Genre Classifier Web App

A beautiful, interactive web application for predicting music genres using machine learning!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

### 2. Train the Model (if not done already)

```bash
python src/models/train.py
```

### 3. Run the Web App

```bash
streamlit run app_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

## 🌟 Features

### ✨ Beautiful Interface

- Clean, modern design with custom styling
- Interactive charts and visualizations
- Real-time audio preview

### 🎯 Smart Predictions

- Upload audio files (.wav, .mp3, .m4a)
- Get instant genre predictions
- View confidence scores for all genres
- Interactive bar charts showing top predictions

### 📊 Detailed Analysis

- Shows top 5 genre predictions
- Confidence percentage for each prediction
- Model insights (high/medium/low confidence)
- File information display

### 🎵 Supported Genres

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 🎬 How to Use

1. **Launch the app**: Run `streamlit run app_streamlit.py`
2. **Upload audio**: Click "Browse files" and select your music file
3. **Preview**: Listen to the audio preview to confirm it's correct
4. **Predict**: Click the "🔮 Predict Genre" button
5. **Analyze**: View the prediction results and confidence scores

## 📱 Screenshots

The app includes:

- 📁 **File Upload Section**: Drag & drop or browse for audio files
- 🎧 **Audio Preview**: Built-in audio player
- 🎯 **Prediction Results**: Beautiful gradient boxes showing the predicted genre
- 📊 **Interactive Charts**: Plotly charts showing confidence scores
- 📋 **Detailed Table**: Complete breakdown of all genre probabilities

## 🔧 Technical Details

### Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 84 audio features including:
  - MFCC (Mel-frequency cepstral coefficients)
  - Chroma features
  - Spectral contrast, centroid, bandwidth, rolloff
  - Tonnetz (tonal centroid features)
  - Zero crossing rate

### Performance Caching

- Model loading is cached for faster predictions
- Feature extraction is cached to improve performance
- Optimized for multiple predictions

## 🛠️ Customization

### Styling

The app uses custom CSS for beautiful styling. You can modify the colors and layout in the `st.markdown()` sections of `app_streamlit.py`.

### Adding New Features

- Modify `extract_features()` to add new audio features
- Update the model training script to include new features
- The web app will automatically adapt to new model outputs

## 📦 Dependencies

Key packages required:

- `streamlit`: Web app framework
- `librosa`: Audio processing
- `scikit-learn`: Machine learning
- `plotly`: Interactive charts
- `pandas`: Data handling
- `numpy`: Numerical computing

## 🎯 Tips for Best Results

1. **Audio Quality**: Use clear, good-quality audio files
2. **File Length**: 30-second clips work best (the model analyzes first 30 seconds)
3. **Supported Formats**: WAV files generally work best, but MP3 and M4A are also supported
4. **File Size**: Keep files under 16MB for faster processing

## 🚀 Deployment Options

### Local Development

```bash
streamlit run app_streamlit.py
```

### Streamlit Cloud (Free Hosting)

1. Push your code to GitHub
2. Connect your repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click!

### Custom Domain

```bash
streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
```

## 🎉 Example Usage

```bash
# Start the app
streamlit run app_streamlit.py

# Open browser to http://localhost:8501
# Upload a rock song
# See prediction: "🏆 Predicted Genre: ROCK" with 87% confidence!
```

## 🛟 Troubleshooting

### Common Issues

**"Model file not found"**

- Run `python src/models/train.py` first to train the model

**"Error extracting features"**

- Check if your audio file is valid
- Try converting to WAV format
- Ensure file is not corrupted

**"App won't start"**

- Check if all dependencies are installed: `pip install -r requirement.txt`
- Ensure you're in the correct directory

**"Slow predictions"**

- Large files take longer to process
- First prediction is slower due to model loading
- Subsequent predictions are faster due to caching

---

🎵 **Enjoy classifying music genres with AI!** 🎵
