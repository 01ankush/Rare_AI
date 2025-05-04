# utils/voice_utils.py

import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Load the trained model
MODEL_PATH = "models/voice_disease_detection_model2.h5"
model = load_model(MODEL_PATH)

# Class labels
CLASSES = ['Normal', 'Vox_senilis', 'Laryngozele']


def preprocess_audio(file_path, max_pad_len=1024):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    if len(mfcc_scaled) > max_pad_len:
        mfcc_scaled = mfcc_scaled[:max_pad_len]
    else:
        mfcc_scaled = np.pad(mfcc_scaled, (0, max_pad_len - len(mfcc_scaled)), mode='constant')

    return mfcc_scaled


def predict_voice_disease(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ File not found. Please check the path.")

    features = preprocess_audio(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)

    return {
        "class": CLASSES[predicted_index],
        "confidence": float(prediction[0][predicted_index])
    }
