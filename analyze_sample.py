import librosa
import numpy as np
import os
from utils.audio_processor import AudioProcessor

def analyze_audio(path):
    print(f"Analyzing {path}...")
    processor = AudioProcessor()
    
    # Load and extract
    try:
        y, sr = librosa.load(path, sr=None)
        features = processor.extract_features(path)
        
        # Analyze stability
        # MFCC stds are indices 20-39
        mfcc_stds = features[0][20:40]
        avg_mfcc_std = np.mean(mfcc_stds)
        
        # Spectral centroids (indices 40,41 - mean, std)
        centroid_std = features[0][41]
        
        # RMS energy std (index 72)
        rms_std = features[0][72]
        
        print(f"Features Summary:")
        print(f"- Features First 5: {features[0][:5]}")
        print(f"- MFCC Avg Std: {avg_mfcc_std:.4f}")
        print(f"- Centroid Std: {centroid_std:.4f}")
        print(f"- RMS Energy Std: {rms_std:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

import sys

if __name__ == "__main__":
    target = 'sample.mp3'
    if len(sys.argv) > 1:
        target = sys.argv[1]
        
    if os.path.exists(target):
        analyze_audio(target)
    else:
        print(f"{target} not found")
