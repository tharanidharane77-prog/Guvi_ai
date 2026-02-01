import joblib
import os
import numpy as np
from utils.audio_processor import AudioProcessor

def debug_prediction():
    print("--- Debugging Prediction ---")
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    processor = AudioProcessor()
    
    feat = processor.extract_features('sample.mp3')
    scaled_feat = scaler.transform(feat)
    
    pred = model.predict(scaled_feat)[0]
    probs = model.predict_proba(scaled_feat)[0]
    
    classes = model.classes_
    print(f"Classes: {classes}")
    print(f"Prediction Value: {pred}")
    print(f"Probabilities: {probs}")
    
    # Check if 1 is AI and 0 is Human
    label = "AI_GENERATED" if pred == 1 else "HUMAN"
    print(f"Result Label: {label}")

if __name__ == "__main__":
    debug_prediction()
