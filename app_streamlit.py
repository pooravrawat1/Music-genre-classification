import streamlit as st
import joblib
import numpy as np
import librosa
import pandas as pd
import os
import tempfile
import yt_dlp

# Basic page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵"
)

# Very basic CSS styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    .confidence-box {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 3px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your music file or import from YouTube to discover its genre</p>', unsafe_allow_html=True)

def display_results(prediction, probabilities):
    """Display prediction results"""
    # Genre names
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
             'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    if probabilities is not None:
        # Create results
        results = pd.DataFrame({
            'Genre': genres,
            'Confidence': probabilities * 100
        }).sort_values('Confidence', ascending=False)
        
        top_genre = results.iloc[0]['Genre']
        top_confidence = results.iloc[0]['Confidence']
        
        # Display result
        st.markdown(f"""
        <div class="result-box">
            <h3>🎵 Predicted Genre: {top_genre.upper()}</h3>
            <p>Confidence: {top_confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show top 3
        st.write("### Top 3 Predictions:")
        for i, (_, row) in enumerate(results.head(3).iterrows()):
            st.markdown(f"""
            <div class="confidence-box">
                {i+1}. {row['Genre'].title()}: {row['Confidence']:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Show all results
        st.write("### All Results:")
        st.dataframe(results.round(1), use_container_width=True)
    
    else:
        st.markdown(f"""
        <div class="result-box">
            <h3>🎵 Predicted Genre: {prediction}</h3>
            <p>Confidence scores not available</p>
        </div>
        """, unsafe_allow_html=True)

def download_youtube_audio(url):
    """Download audio from YouTube URL"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            
        # Find the downloaded file
        for file in os.listdir('.'):
            if file.startswith('temp_audio') and file.endswith('.wav'):
                return file, title
        
        return None, None
    except Exception as e:
        st.error(f"Error downloading from YouTube: {e}")
        return None, None

@st.cache_data
def extract_features(file_path):
    """Extract basic audio features"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30)
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

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        for feat in [centroid, bandwidth, rolloff]:
            features.append(np.mean(feat))
            features.append(np.std(feat))

        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the model"""
    try:
        model_path = "models/random_forest.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.error("Model file not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_genre(audio_file, model):
    """Make prediction"""
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Extract features
        features = extract_features(tmp_path)
        if features is None:
            return None, None
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        # Cleanup
        os.unlink(tmp_path)
        
        return prediction, probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main content
st.write("## Choose Input Method")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload File", "🎬 YouTube"])

# Load model
model = load_model()
if model is None:
    st.stop()

with tab1:
    st.write("### Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a']
    )
    
    if uploaded_file:
        # Show file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        
        # Audio player
        st.audio(uploaded_file)
        
        # Predict button
        if st.button("🎯 Analyze Genre", key="upload"):
            with st.spinner("Analyzing..."):
                prediction, probabilities = predict_genre(uploaded_file, model)
            
            if prediction is not None:
                display_results(prediction, probabilities)

with tab2:
    st.write("### Import from YouTube")
    st.info("📝 **Note:** You need to install yt-dlp and ffmpeg for this feature to work.")
    
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if youtube_url and st.button("🎬 Download & Analyze", key="youtube"):
        if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
            with st.spinner("Downloading from YouTube..."):
                audio_file, title = download_youtube_audio(youtube_url)
            
            if audio_file and os.path.exists(audio_file):
                st.success(f"Downloaded: {title}")
                
                # Show audio player
                with open(audio_file, 'rb') as f:
                    st.audio(f.read())
                
                with st.spinner("Analyzing..."):
                    # Create a file-like object for the downloaded audio
                    with open(audio_file, 'rb') as f:
                        audio_bytes = f.read()
                    
                    # Create a temporary file object
                    class AudioFile:
                        def __init__(self, data):
                            self.data = data
                        def read(self):
                            return self.data
                    
                    audio_obj = AudioFile(audio_bytes)
                    prediction, probabilities = predict_genre(audio_obj, model)
                
                if prediction is not None:
                    display_results(prediction, probabilities)
                
                # Cleanup
                os.unlink(audio_file)
            else:
                st.error("Failed to download audio from YouTube")
        else:
            st.error("Please enter a valid YouTube URL")

# Footer
st.write("---")
st.write("Simple Music Genre Classifier")
