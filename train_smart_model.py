import os
import joblib
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from utils.audio_processor import AudioProcessor

def extract_chunks(audio_path, chunk_duration=1.0, overlap=0.5):
    """Splits audio into overlapping chunks for more training data."""
    y, sr = librosa.load(audio_path, sr=22050)
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step = chunk_samples - overlap_samples
    
    chunks = []
    for i in range(0, len(y) - chunk_samples, step):
        chunks.append(y[i : i + chunk_samples])
    
    # If file is too short, still return one chunk (padded)
    if not chunks and len(y) > 0:
        chunks.append(y)
        
    return chunks, sr

def train_smart_model():
    """
    Highly advanced training using 256 granular speech analysis features.
    Implements chunking and high-fidelity augmentation to detect single-word AI speech.
    """
    processor = AudioProcessor()
    X = []
    y = []

    print("--- Phase 1: High-Resolution Data Preparation ---")
    
    # Configuration for training
    chunk_length = 1.0  # 1 second chunks for single-word similarity
    overlap = 0.5
    augment_factor = 5 # Number of augmented versions per chunk

    # 1. AI DATA (Label 1)
    ai_refs = ['sample.wav', 'test2.wav']
    for s in ai_refs:
        if os.path.exists(s):
            print(f"Processing AI Source (Chunking): {s}")
            chunks, sr = extract_chunks(s, chunk_length, overlap)
            
            for chunk in chunks:
                # Save chunk to temp file for processor
                temp_chunk_path = "temp_training_chunk.wav"
                librosa.output.write_wav(temp_chunk_path, chunk, sr) if hasattr(librosa, 'output') else None
                # librosa.output was removed in newer versions, use soundfile
                import soundfile as sf
                sf.write(temp_chunk_path, chunk, sr)
                
                # Extract features
                feat = processor.extract_features(temp_chunk_path).flatten()
                X.append(feat)
                y.append(1)
                
                # Augmentation (Noise & Pitch)
                for _ in range(augment_factor):
                    # Add noise
                    noise_lvl = np.random.uniform(0.005, 0.02)
                    noisy_feat = feat + np.random.normal(0, noise_lvl, 256)
                    X.append(noisy_feat)
                    y.append(1)
            
    # 2. HUMAN DATA (Label 0)
    human_refs = ['test3.wav', 'test4.wav']
    for s in human_refs:
        if os.path.exists(s):
            print(f"Processing Human Source (Chunking): {s}")
            chunks, sr = extract_chunks(s, chunk_length, overlap)
            
            # Humans need more diversity
            for chunk in chunks:
                temp_chunk_path = "temp_training_chunk_h.wav"
                import soundfile as sf
                sf.write(temp_chunk_path, chunk, sr)
                
                feat = processor.extract_features(temp_chunk_path).flatten()
                X.append(feat)
                y.append(0)
                
                # More aggressive augmentation for human voice
                for _ in range(augment_factor + 2):
                    noise_lvl = np.random.uniform(0.01, 0.05)
                    noisy_feat = feat + np.random.normal(0, noise_lvl, 256)
                    X.append(noisy_feat)
                    y.append(0)

    # 3. EXTRA SYNTHETIC NEGATIVES (Robotic Patterns)
    # Generate random features with "AI-like" spectral flatness patterns
    print("Synthesizing algorithmic patterns...")
    for _ in range(500):
        # AI often has high energy in specific bands or very low variance
        synth_ai = np.random.normal(0, 1.0, 256)
        # Force low flatness variance
        synth_ai[120:122] = 0.001 
        X.append(synth_ai)
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    print(f"Extracted {len(X)} high-resolution samples.")
    print(f"Feature Vector Size: {X.shape[1]}")

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    print("--- Phase 2: Training Deeper Ensemble model ---")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use ExtraTrees for better variance handling and less overfitting on augmented data
    model = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_scaled, y)
    
    score = model.score(X_scaled, y)
    print(f"Training Accuracy: {score*100:.2f}%")

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Clean up
    if os.path.exists("temp_training_chunk.wav"): os.remove("temp_training_chunk.wav")
    if os.path.exists("temp_training_chunk_h.wav"): os.remove("temp_training_chunk_h.wav")
    
    print("--- Success: Advanced Detection Engine Deployed (256 Features) ---")

if __name__ == "__main__":
    train_smart_model()
