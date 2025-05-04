import streamlit as st
from utils.facial_utils import extract_action_units, predict_from_aus
from utils.text_utils import predict_from_text
from pathlib import Path
import tempfile

# ✅ ADD: voice utils import
from utils.voice_utils import predict_voice_disease
import os

# ======================== CONFIG ======================== #
st.set_page_config(page_title="RARE AI - Multimodal Health Predictor", page_icon="🧠")

openface_path = r"C:\Users\HP\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
model_path = "models/model_simplelstm.pth"
scaler_path = "models/scaler.pkl"

# ======================== TITLE ======================== #
st.title("🧠 RARE AI - Multimodal Health Predictor")

# ======================== FACIAL SECTION ======================== #
st.header("📹  Psychological distress conditions Predictor")
uploaded_file = st.file_uploader("Upload a face video", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.info("Extracting facial features via OpenFace...")
    try:
        df_filtered = extract_action_units(video_path, openface_path)
        st.success("Facial features extracted.")

        # st.write("Preview of extracted features:")
        # st.dataframe(df_filtered.head())

        st.info("Predicting emotion from first frame...")
        prediction = predict_from_aus(df_filtered.iloc[0], model_path, scaler_path)
        if prediction == 0:
            st.success(f"🎯 You are not suffering form dipression/anxiety")
        else:
            st.success(f" You are suffering form dipression/anxiety ")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ======================== TEXT SECTION ======================== #
st.header("💬 Rare Disease Prediction from Medical Symptoms")

symptom_input = st.text_area("Enter medical symptoms (comma-separated):", height=150)

if st.button("🔍 Predict Disease"):
    if symptom_input.strip() == "":
        st.warning("Please enter some symptoms to proceed.")
    else:
        st.info("Predicting disease...")
        try:
            disease = predict_from_text(symptom_input)
            st.success(f"🩺 Predicted Disease: **{disease}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ======================== VOICE SECTION ======================== #
st.header("🎙️ Voice-based Disease Detection")
st.write("Upload a WAV file (16kHz) of the patient’s voice for disease detection.")

uploaded_audio = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(uploaded_audio.read())
        temp_audio_path = tmp_audio.name

    try:
        result = predict_voice_disease(temp_audio_path)
        st.success("✅ Prediction Successful!")
        st.write(f"**You may be suffering from :** {result['class']} rare disease")
        st.write(f"**Confidence:** {result['confidence']:.2f}")
    except Exception as e:
        st.error(f"❌ Error during voice prediction: {e}")
    finally:
        os.remove(temp_audio_path)
