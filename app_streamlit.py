import streamlit as stimport streamlit as stimport streamlit as st

import joblib

import numpy as npimport joblibimport joblib

import librosa

import pandas as pdimport numpy as npimport numpy as np

import os

import tempfileimport librosaimport librosa



# Set page configimport pandas as pdimport pandas as pd

st.set_page_config(

    page_title="Music Genre Classifier",import plotly.express as pximport plotly.express as px

    page_icon="🎵",

    layout="wide"import plotly.graph_objects as goimport os

)

import osimport tempfile

# Simple header

st.title("🎵 Music Genre Classifier")import tempfilefrom pathlib import Path

st.write("Upload your music and find out its genre")

from pathlib import Path

@st.cache_data

def extract_features(file_path, sr=22050):# Set page config

    """Extract audio features for genre classification"""

    try:# Set page configst.set_page_config(

        # Load audio file

        y, sr = librosa.load(file_path, sr=sr, duration=30)st.set_page_config(    page_title="🎵 Music Genre Classifier",

        features = []

    page_title="🎵 Music Genre Classifier",    page_icon="🎵",

        # MFCC features

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)    page_icon="🎵",    layout="wide",

        features.extend(np.mean(mfccs.T, axis=0))

        features.extend(np.std(mfccs.T, axis=0))    layout="wide",    initial_sidebar_state="collapsed"



        # Chroma features    initial_sidebar_state="collapsed")

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        features.extend(np.mean(chroma.T, axis=0)))

        features.extend(np.std(chroma.T, axis=0))

# Custom CSS for clean styling

        # Spectral Contrast

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)# Simple, clean CSSst.markdown("""

        features.extend(np.mean(spec_contrast.T, axis=0))

        features.extend(np.std(spec_contrast.T, axis=0))st.markdown("""<style>



        # Tonnetz<style>    /* Clean, minimal styling */

        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        features.extend(np.mean(tonnetz.T, axis=0))    .stApp {    .stApp {

        features.extend(np.std(tonnetz.T, axis=0))

        background-color: #f8f9fa;        background-color: #f8f9fa;

        # Zero Crossing Rate

        zcr = librosa.feature.zero_crossing_rate(y)    }        color: #333;

        features.append(np.mean(zcr))

        features.append(np.std(zcr))        }



        # Spectral features    .main-header {    

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)        background: linear-gradient(90deg, #007bff, #0056b3);    /* Hide default Streamlit elements */

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        padding: 2rem;    #MainMenu {visibility: hidden;}

        for feat in [centroid, bandwidth, rolloff]:

            features.append(np.mean(feat))        border-radius: 10px;    footer {visibility: hidden;}

            features.append(np.std(feat))

        text-align: center;    header {visibility: hidden;}

        return np.array(features).reshape(1, -1)

            margin-bottom: 2rem;    

    except Exception as e:

        st.error(f"Error extracting features: {str(e)}")        color: white;    /* Simple header */

        return None

    }    .main-header {

@st.cache_resource

def load_model():            background-color: #007bff;

    """Load the trained model"""

    try:    .prediction-card {        padding: 1.5rem;

        model_path = "models/random_forest.pkl"

                background: white;        text-align: center;

        if not os.path.exists(model_path):

            st.error(f"Model file not found: {model_path}")        border: 1px solid #dee2e6;        border-radius: 10px;

            return None

                border-radius: 10px;        margin-bottom: 2rem;

        model = joblib.load(model_path)

        return model        padding: 1.5rem;    }

    

    except Exception as e:        margin: 1rem 0;    

        st.error(f"Error loading model: {str(e)}")

        return None        box-shadow: 0 2px 4px rgba(0,0,0,0.1);    .main-header h1 {



def predict_genre(audio_file, model):    }        color: white;

    """Predict genre from uploaded audio file"""

    try:            margin: 0;

        # Save uploaded file temporarily

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:    .confidence-card {        font-size: 2.5rem;

            tmp_file.write(audio_file.read())

            tmp_path = tmp_file.name        background: #e3f2fd;    }

        

        # Extract features        border-left: 4px solid #2196f3;    

        features = extract_features(tmp_path)

        if features is None:        padding: 1rem;    .main-header p {

            return None, None

                margin: 0.5rem 0;        color: white;

        # Make prediction

        prediction = model.predict(features)[0]        border-radius: 5px;        margin: 0.5rem 0 0 0;

        

        # Get probabilities if available    }        opacity: 0.9;

        probabilities = None

        if hasattr(model, 'predict_proba'):        }

            probabilities = model.predict_proba(features)[0]

            .top-prediction {    

        # Clean up temporary file

        os.unlink(tmp_path)        background: linear-gradient(90deg, #28a745, #20c997);    /* Simple card styling */

        

        return prediction, probabilities        color: white;    .simple-card {

        

    except Exception as e:        padding: 2rem;        background: white;

        st.error(f"Error during prediction: {str(e)}")

        return None, None        border-radius: 10px;        border: 1px solid #dee2e6;



# Main sections using columns        text-align: center;        border-radius: 8px;

col1, col2 = st.columns([3, 1])

        margin: 1rem 0;        padding: 1.5rem;

with col1:

    # Load model    }        margin: 1rem 0;

    model = load_model()

    </style>        box-shadow: 0 2px 4px rgba(0,0,0,0.1);

    if model is None:

        st.error("❌ Could not load the model. Please check if the model file exists.")""", unsafe_allow_html=True)    }

        st.stop()

        

    # File upload section

    uploaded_file = st.file_uploader(@st.cache_data    /* Prediction result */

        "Choose an audio file",

        type=['wav', 'mp3', 'm4a'],def extract_features(file_path, sr=22050):    .prediction-result {

        help="Supported formats: WAV, MP3, M4A"

    )    """Extract audio features for genre classification"""        background-color: #28a745;

    

    if uploaded_file is not None:    try:        padding: 1.5rem;

        # Audio player

        st.write("### Preview")        # Load audio file        border-radius: 8px;

        st.audio(uploaded_file)

                y, sr = librosa.load(file_path, sr=sr, duration=30)        text-align: center;

        # Prediction button

        if st.button("Analyze Genre", type="primary"):        features = []        margin: 1rem 0;

            with st.spinner("Analyzing..."):

                prediction, probabilities = predict_genre(uploaded_file, model)    }

            

            if prediction is not None:        # MFCC features    

                # Define genre names (adjust based on your model)

                genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop',         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)    .prediction-result h2 {

                              'jazz', 'metal', 'pop', 'reggae', 'rock']

                        features.extend(np.mean(mfccs.T, axis=0))        color: white;

                if probabilities is not None:

                    # Create results dataframe        features.extend(np.std(mfccs.T, axis=0))        margin: 0;

                    results_df = pd.DataFrame({

                        'Genre': genre_names,        font-size: 2rem;

                        'Confidence': probabilities * 100

                    }).sort_values('Confidence', ascending=False)        # Chroma features    }

                    

                    # Get top prediction        chroma = librosa.feature.chroma_stft(y=y, sr=sr)    

                    top_genre = results_df.iloc[0]['Genre']

                    top_confidence = results_df.iloc[0]['Confidence']        features.extend(np.mean(chroma.T, axis=0))    .prediction-result p {

                    

                    # Display main prediction        features.extend(np.std(chroma.T, axis=0))        color: white;

                    st.success(f"### Predicted Genre: {top_genre.upper()}")

                    st.info(f"Confidence: {top_confidence:.2f}%")        margin: 0.5rem 0 0 0;

                    

                    # Show all results in a table        # Spectral Contrast    }

                    st.write("### All Confidence Scores")

                    st.dataframe(        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)    

                        results_df.round(2),

                        use_container_width=True,        features.extend(np.mean(spec_contrast.T, axis=0))    /* Button styling */

                        hide_index=True

                    )        features.extend(np.std(spec_contrast.T, axis=0))    .stButton > button {

                    

                    # Show top 3 in a clearer format        background-color: #007bff;

                    st.write("### Top 3 Predictions")

                    for i, (_, row) in enumerate(results_df.head(3).iterrows()):        # Tonnetz        color: white;

                        st.write(f"{i+1}. {row['Genre'].title()}: {row['Confidence']:.2f}%")

                            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)        border: none;

                else:

                    st.success(f"### Predicted Genre: {prediction}")        features.extend(np.mean(tonnetz.T, axis=0))        border-radius: 5px;

                    st.warning("Confidence scores not available")

            else:        features.extend(np.std(tonnetz.T, axis=0))        padding: 0.5rem 1rem;

                st.error("Failed to analyze the audio file. Please try again.")

        font-weight: 500;

with col2:

    # Simple sidebar with info        # Zero Crossing Rate    }

    st.write("### About")

    st.write("""        zcr = librosa.feature.zero_crossing_rate(y)    

    This app uses a Random Forest model to classify music into genres based on audio features.

            features.append(np.mean(zcr))    .stButton > button:hover {

    **Supported genres:**

    - Blues        features.append(np.std(zcr))        background-color: #0056b3;

    - Classical

    - Country    }

    - Disco

    - Hip-hop        # Spectral features    

    - Jazz

    - Metal        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)    /* Metric cards */

    - Pop

    - Reggae        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)    .metric-card {

    - Rock

    """)        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)        background: white;



# Simple footer        border: 1px solid #dee2e6;

st.write("---")

st.caption("Music Genre Classifier • Built with Streamlit and Machine Learning")        for feat in [centroid, bandwidth, rolloff]:        border-radius: 8px;

            features.append(np.mean(feat))        padding: 1rem;

            features.append(np.std(feat))        text-align: center;

    }

        return np.array(features).reshape(1, -1)    

        .metric-card h3 {

    except Exception as e:        color: #007bff;

        st.error(f"Error extracting features: {str(e)}")        font-size: 1.8rem;

        return None        margin: 0;

    }

@st.cache_resource    

def load_model():    .metric-card p {

    """Load the trained model"""        color: #6c757d;

    try:        font-size: 0.9rem;

        model_path = "models/random_forest.pkl"        margin: 0.5rem 0 0 0;

            }

        if not os.path.exists(model_path):</style>

            st.error(f"Model file not found: {model_path}")""", unsafe_allow_html=True)

            return None

        @st.cache_data

        model = joblib.load(model_path)def extract_features(file_path, sr=22050):

        return model    """Extract audio features for genre classification"""

        try:

    except Exception as e:        y, sr = librosa.load(file_path, sr=sr, duration=30)

        st.error(f"Error loading model: {str(e)}")        features = []

        return None

        # MFCC

def predict_genre(audio_file, model):        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    """Predict genre from uploaded audio file"""        features.extend(np.mean(mfccs.T, axis=0))

    try:        features.extend(np.std(mfccs.T, axis=0))

        # Save uploaded file temporarily

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:        # Chroma

            tmp_file.write(audio_file.read())        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            tmp_path = tmp_file.name        features.extend(np.mean(chroma.T, axis=0))

                features.extend(np.std(chroma.T, axis=0))

        # Extract features

        features = extract_features(tmp_path)        # Spectral Contrast

        if features is None:        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

            return None, None        features.extend(np.mean(spec_contrast.T, axis=0))

                features.extend(np.std(spec_contrast.T, axis=0))

        # Make prediction

        prediction = model.predict(features)[0]        # Tonnetz

                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        # Get probabilities if available        features.extend(np.mean(tonnetz.T, axis=0))

        probabilities = None        features.extend(np.std(tonnetz.T, axis=0))

        if hasattr(model, 'predict_proba'):

            probabilities = model.predict_proba(features)[0]        # Zero Crossing Rate

                zcr = librosa.feature.zero_crossing_rate(y)

        # Clean up temporary file        features.append(np.mean(zcr))

        os.unlink(tmp_path)        features.append(np.std(zcr))

        

        return prediction, probabilities        # Centroid / Bandwidth / Rolloff

                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    except Exception as e:        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        st.error(f"Error during prediction: {str(e)}")        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        return None, None

        for feat in [centroid, bandwidth, rolloff]:

def display_results(prediction, probabilities):            features.append(np.mean(feat))

    """Display prediction results with confidence scores"""            features.append(np.std(feat))

    

    # Define genre names (adjust based on your model)        return np.array(features).reshape(1, -1)

    genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop',     except Exception as e:

                   'jazz', 'metal', 'pop', 'reggae', 'rock']        st.error(f"Error extracting features: {str(e)}")

            return None

    if probabilities is not None:

        # Create results dataframe@st.cache_resource

        results_df = pd.DataFrame({def load_model_and_encoder():

            'Genre': genre_names,    """Load the trained model and label encoder"""

            'Confidence': probabilities * 100    try:

        }).sort_values('Confidence', ascending=False)        model_path = "models/random_forest.pkl"

                scaler_path = "models/scaler.pkl"

        # Get top prediction        encoder_path = "models/label_encoder.pkl"

        top_genre = results_df.iloc[0]['Genre']        

        top_confidence = results_df.iloc[0]['Confidence']        if not os.path.exists(model_path):

                    st.error(f"Model file not found: {model_path}")

        # Display main prediction            st.info("Please train a model first by running: `python src/models/train.py`")

        st.markdown(f"""            return None, None, None

        <div class="top-prediction">        

            <h1>🎯 {top_genre.upper()}</h1>        model = joblib.load(model_path)

            <h2>{top_confidence:.1f}% Confidence</h2>        

        </div>        # Load scaler if it exists

        """, unsafe_allow_html=True)        scaler = None

                if os.path.exists(scaler_path):

        # Display top 3 predictions            scaler = joblib.load(scaler_path)

        st.markdown("### 🏆 Top 3 Predictions")            st.info("✅ Scaler loaded")

                else:

        col1, col2, col3 = st.columns(3)            st.warning("⚠️ No scaler found - using raw features")

        medals = ["🥇", "🥈", "🥉"]        

                # Load encoder if it exists

        for i, (col, (_, row)) in enumerate(zip([col1, col2, col3], results_df.head(3).iterrows())):        encoder = None

            with col:        if os.path.exists(encoder_path):

                st.markdown(f"""            encoder = joblib.load(encoder_path)

                <div class="confidence-card">            st.info("✅ Label encoder loaded")

                    <h3>{medals[i]} {row['Genre'].title()}</h3>        else:

                    <h2>{row['Confidence']:.1f}%</h2>            st.warning("⚠️ No label encoder found - using raw predictions")

                </div>        

                """, unsafe_allow_html=True)        return model, encoder, scaler

            except Exception as e:

        # Show detailed confidence scores        st.error(f"Error loading model: {str(e)}")

        st.markdown("### 📊 All Genre Confidence Scores")        return None, None, None

        

        # Create bar chartdef predict_genre(audio_file, model, encoder, scaler=None):

        fig = px.bar(    """Predict genre from uploaded audio file"""

            results_df,    try:

            x='Confidence',        # Save uploaded file temporarily

            y='Genre',        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:

            orientation='h',            tmp_file.write(audio_file.read())

            title="Confidence Scores for All Genres",            tmp_path = tmp_file.name

            color='Confidence',        

            color_continuous_scale='Blues'        # Extract features

        )        features = extract_features(tmp_path)

                if features is None:

        fig.update_layout(            return None, None

            height=400,        

            showlegend=False,        # Scale features if scaler is available

            title_x=0.5        if scaler is not None:

        )            features = scaler.transform(features)

                

        st.plotly_chart(fig, use_container_width=True)        # Make prediction

                prediction = model.predict(features)[0]

        # Show data table        probabilities = None

        st.markdown("### 📋 Detailed Results")        

        st.dataframe(        if hasattr(model, 'predict_proba'):

            results_df.round(2),            probabilities = model.predict_proba(features)[0]

            use_container_width=True,        

            hide_index=True        # Get genre name

        )        if encoder is not None:

                    try:

    else:                predicted_genre = encoder.inverse_transform([prediction])[0]

        # Fallback if no probabilities available                genre_names = encoder.classes_

        st.markdown(f"""            except Exception as e:

        <div class="top-prediction">                st.warning(f"Label encoder error: {e}")

            <h1>🎯 GENRE {prediction}</h1>                predicted_genre = f"Genre_{prediction}"

            <p>Confidence scores not available</p>                genre_names = [f"Genre_{i}" for i in range(len(probabilities))] if probabilities is not None else None

        </div>        else:

        """, unsafe_allow_html=True)            predicted_genre = f"Genre_{prediction}"

            # If no encoder, create generic genre names

def main():            if probabilities is not None:

    # Header                genre_names = [f"Genre_{i}" for i in range(len(probabilities))]

    st.markdown("""            else:

    <div class="main-header">                genre_names = None

        <h1>🎵 Music Genre Classifier</h1>        

        <p>Upload your music and discover its genre with AI</p>        # Cleanup

    </div>        os.unlink(tmp_path)

    """, unsafe_allow_html=True)        

            return predicted_genre, (genre_names, probabilities)

    # Load model        

    with st.spinner("Loading AI model..."):    except Exception as e:

        model = load_model()        st.error(f"Error during prediction: {str(e)}")

            return None, None

    if model is None:

        st.error("❌ Could not load the model. Please check if the model file exists.")def predict_and_display_results(uploaded_file, model, encoder, scaler=None):

        st.stop()    """Handle prediction and display results with confidence"""

        with st.spinner("🎵 Analyzing your track..."):

    st.success("✅ Model loaded successfully!")        predicted_genre, confidence_data = predict_genre(uploaded_file, model, encoder, scaler)

        

    # File upload section    if predicted_genre is not None:

    st.markdown("### 🎧 Upload Your Music File")        # Debug: Show what type of model we have

            st.info(f"Model type: {type(model).__name__}")

    uploaded_file = st.file_uploader(        st.info(f"Has predict_proba: {hasattr(model, 'predict_proba')}")

        "Choose an audio file",        

        type=['wav', 'mp3', 'm4a'],        # Check if we have confidence data

        help="Supported formats: WAV, MP3, M4A"        if confidence_data[0] is not None and confidence_data[1] is not None:

    )            genre_names, probabilities = confidence_data

                

    if uploaded_file is not None:            # Debug: Show raw probabilities

        # Display file info            st.write("Raw probabilities:", probabilities)

        col1, col2, col3 = st.columns(3)            st.write("Genre names:", genre_names)

                    

        with col1:            # Create sorted results dataframe

            st.metric("File Name", uploaded_file.name)            results_df = pd.DataFrame({

        with col2:                'Genre': genre_names,

            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")                'Confidence': probabilities * 100

        with col3:            }).sort_values('Confidence', ascending=False)

            st.metric("File Type", uploaded_file.type)            

                    # Show all confidence scores

        # Audio player            st.write("All confidence scores:")

        st.markdown("### 🎵 Preview")            st.dataframe(results_df)

        st.audio(uploaded_file)            

                    # Get top 3 predictions

        # Prediction button            top_3 = results_df.head(3)

        if st.button("🚀 Analyze Genre", type="primary", use_container_width=True):            main_confidence = top_3.iloc[0]['Confidence']

            with st.spinner("Analyzing your music..."):            

                prediction, probabilities = predict_genre(uploaded_file, model)            # Main prediction result with confidence

                        st.markdown(f"""

            if prediction is not None:            <div class="prediction-result">

                display_results(prediction, probabilities)                <h2>🎯 {predicted_genre.upper()}</h2>

            else:                <p>Primary prediction with {main_confidence:.1f}% confidence</p>

                st.error("Failed to analyze the audio file. Please try again.")            </div>

                """, unsafe_allow_html=True)

    # About section            

    st.markdown("---")            # Top 3 predictions in prominent cards

    with st.expander("ℹ️ About this app"):            st.markdown("### 🏆 Top 3 Predictions")

        st.markdown("""            

        This app uses machine learning to classify music genres based on audio features.            # Create three columns for top 3

                    col1, col2, col3 = st.columns(3)

        **Features analyzed:**            

        - MFCC (Mel-frequency cepstral coefficients)            # Medal emojis for top 3

        - Chroma features            medals = ["🥇", "🥈", "🥉"]

        - Spectral contrast            colors = ["#FFD700", "#C0C0C0", "#CD7F32"]  # Gold, Silver, Bronze

        - Tonnetz            

        - Zero crossing rate            for i, (col, (_, row)) in enumerate(zip([col1, col2, col3], top_3.iterrows())):

        - Spectral centroid, bandwidth, and rolloff                with col:

                            st.markdown(f"""

        **Supported genres:**                    <div class="simple-card" style="text-align: center; border-left: 4px solid {colors[i]};">

        Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock                        <h2 style="color: {colors[i]}; margin: 0;">{medals[i]}</h2>

        """)                        <h3 style="margin: 0.5rem 0;">{row['Genre'].upper()}</h3>

                        <h2 style="color: #007bff; margin: 0;">{row['Confidence']:.1f}%</h2>

if __name__ == "__main__":                        <p style="margin: 0.5rem 0; opacity: 0.8;">Confidence</p>

    main()                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced confidence visualization
            st.markdown("### 📊 Detailed Confidence Analysis")
            
            # Create two types of charts side by side
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Horizontal bar chart for all genres
                fig_bar = px.bar(
                    results_df.tail(8),  # Top 8 for better readability
                    x='Confidence',
                    y='Genre',
                    orientation='h',
                    title="All Genre Confidence Scores",
                    color='Confidence',
                    color_continuous_scale=['#e9ecef', '#007bff', '#0056b3'],
                    text='Confidence'
                )
                
                fig_bar.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    textfont_color='white',
                    textfont_size=10
                )
                
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_size=14,
                    title_font_color='white',
                    xaxis={'gridcolor': 'rgba(255,255,255,0.2)', 'color': 'white', 'title': 'Confidence %'},
                    yaxis={'gridcolor': 'rgba(255,255,255,0.2)', 'color': 'white'},
                    coloraxis_colorbar={'title': 'Confidence %', 'titlefont_color': 'white', 'tickfont_color': 'white'},
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with chart_col2:
                # Pie chart for top 5 predictions
                top_5 = results_df.head(5)
                
                fig_pie = px.pie(
                    top_5,
                    values='Confidence',
                    names='Genre',
                    title="Top 5 Genre Distribution",
                    color_discrete_sequence=['#007bff', '#0056b3', '#28a745', '#17a2b8', '#ffc107']
                )
                
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_color='white',
                    textfont_size=10
                )
                
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_size=14,
                    title_font_color='white',
                    height=350,
                    showlegend=True,
                    legend={'font_color': 'white'}
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Confidence insights with enhanced styling
            st.markdown("### 🧠 AI Analysis Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                if main_confidence > 70:
                    st.success(f"🎯 **High Confidence Prediction**\n\nThe AI is {main_confidence:.1f}% confident this is **{predicted_genre}**")
                elif main_confidence > 50:
                    st.warning(f"⚠️ **Moderate Confidence**\n\nThe AI is {main_confidence:.1f}% confident, suggesting possible genre blending")
                else:
                    st.info(f"🤔 **Low Confidence**\n\nThe AI is only {main_confidence:.1f}% confident - this might be a genre fusion or experimental track")
            
            with insight_col2:
                # Additional insights based on top 3
                second_choice = top_3.iloc[1]
                third_choice = top_3.iloc[2]
                
                st.markdown(f"""
                **Alternative Predictions:**
                - **2nd Choice:** {second_choice['Genre']} ({second_choice['Confidence']:.1f}%)
                - **3rd Choice:** {third_choice['Genre']} ({third_choice['Confidence']:.1f}%)
                
                **Genre Certainty:** {main_confidence - second_choice['Confidence']:.1f}% gap between 1st and 2nd choice
                """)
            
            # Detailed results table with enhanced styling
            st.markdown("### 📋 Complete Analysis Table")
            
            # Format the dataframe for display
            display_df = results_df.copy()
            display_df['Confidence (%)'] = display_df['Confidence'].round(1)
            display_df['Rank'] = range(1, len(display_df) + 1)
            display_df['Probability Bar'] = display_df['Confidence'].apply(
                lambda x: '█' * int(x/5) + '░' * (20 - int(x/5))
            )
            
            # Reorder columns
            display_df = display_df[['Rank', 'Genre', 'Confidence (%)', 'Probability Bar']]
            
            st.dataframe(
                display_df.reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
        
        else:
            # Fallback if no confidence data
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎯 {predicted_genre.upper()}</h2>
                <p>Genre detected (confidence data unavailable)</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header with Spotify-style design
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Genre Classifier</h1>
        <p>Discover your music's genre with AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🎵 Loading AI model..."):
        model, encoder, scaler = load_model_and_encoder()
    
    if model is None:
        st.error("❌ Model not found. Please train the model first.")
        st.code("python src/models/train.py", language="bash")
        st.stop()
    
    st.success("✅ AI model loaded and ready!")
    
    # Main content in Spotify-style cards
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Upload section
        st.markdown("### 🎧 Upload Your Track")
        st.markdown('<div class="simple-card">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your music file here",
            type=['wav', 'mp3', 'm4a'],
            help="Supports WAV, MP3, and M4A formats",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # File info in a clean card
            st.markdown("#### 📊 Track Information")
            file_col1, file_col2, file_col3 = st.columns(3)
            
            with file_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{uploaded_file.name[:15]}{'...' if len(uploaded_file.name) > 15 else ''}</h3>
                    <p>Filename</p>
                </div>
                """, unsafe_allow_html=True)
            
            with file_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{uploaded_file.size / 1024:.1f} KB</h3>
                    <p>File Size</p>
                </div>
                """, unsafe_allow_html=True)
            
            with file_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{uploaded_file.type.split('/')[-1].upper()}</h3>
                    <p>Format</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Audio player with custom styling
            st.markdown("#### 🎵 Preview")
            st.audio(uploaded_file, format='audio/wav')
            
            # Predict button with Spotify style
            if st.button("🚀 Analyze Genre", type="primary", use_container_width=True):
                predict_and_display_results(uploaded_file, model, encoder)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar info
        st.markdown("### 🎯 About")
        st.markdown("""
        <div class="simple-card">
            <h4>🤖 AI-Powered Analysis</h4>
            <p>Our advanced machine learning model analyzes 84+ audio features to predict your music's genre with high accuracy.</p>
            
            <h4>🎵 Supported Genres</h4>
            <p>Blues • Classical • Country • Disco • Hip-hop • Jazz • Metal • Pop • Reggae • Rock</p>
            
            <h4>⚡ Features</h4>
            <p>• Real-time analysis<br>
            • Confidence scoring<br>
            • Interactive visualizations<br>
            • Multiple format support</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; opacity: 0.7;">
        <p>Made with ❤️ using Streamlit & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()