import os
import librosa
import numpy as np
from utils.audio_processor import AudioProcessor
from utils.model_handler import ModelHandler
import soundfile as sf

def test_on_short_segments():
    processor = AudioProcessor()
    handler = ModelHandler()
    
    print("--- Verifying Advanced Detection Engine ---")
    
    # 1. Test AI Source (Short Segment)
    ai_file = 'sample.wav'
    if os.path.exists(ai_file):
        y, sr = librosa.load(ai_file, sr=22050)
        # Extract a 0.5s segment (like a single word)
        start = int(2.0 * sr)
        end = int(2.5 * sr)
        segment = y[start:end]
        
        sf.write('temp_ai_word.wav', segment, sr)
        feats = processor.extract_features('temp_ai_word.wav')
        result, error = handler.predict(feats)
        
        print(f"AI Sample (0.5s segment): {result['classification']} (Confidence: {result['confidence']})")
        if os.path.exists('temp_ai_word.wav'): os.remove('temp_ai_word.wav')

    # 2. Test Human Source (Short Segment)
    human_file = 'test3.wav'
    if os.path.exists(human_file):
        y, sr = librosa.load(human_file, sr=22050)
        # Extract a 0.5s segment
        start = int(10.0 * sr)
        end = int(10.5 * sr)
        segment = y[start:end]
        
        sf.write('temp_human_word.wav', segment, sr)
        feats = processor.extract_features('temp_human_word.wav')
        result, error = handler.predict(feats)
        
        print(f"Human Sample (0.5s segment): {result['classification']} (Confidence: {result['confidence']})")
        if os.path.exists('temp_human_word.wav'): os.remove('temp_human_word.wav')

if __name__ == "__main__":
    test_on_short_segments()
