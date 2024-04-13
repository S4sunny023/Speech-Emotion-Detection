import streamlit as st
import numpy as np
import librosa  
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('F:\Speech-Emotion-recognition\speech_model.h5')

def extract_features(audio, sample_rate):
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return np.expand_dims(mfccs_processed, axis=0)  

emotion_mapping = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

st.title('Speech Emotion Recognition')


file = st.file_uploader("Upload an audio file", type=["wav"])

if file is not None:
    
    st.audio(file)
 
    audio, sample_rate = librosa.load(file, sr=None)
    
   
    features = extract_features(audio, sample_rate)
    
    
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)
    
  
    predicted_emotion = emotion_mapping.get(predicted_class, "Unknown")
    
    st.write(f'Predicted Emotion: {predicted_emotion}')
