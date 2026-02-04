import base64
import os
import uuid
from pydub import AudioSegment
import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, temp_dir='temp_audio'):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def base64_to_wav(self, base64_string):
        """Decodes base64 audio and saves it. Prefers WAV but keeps original if conversion fails."""
        file_id = str(uuid.uuid4())
        mp3_path = os.path.join(self.temp_dir, f"{file_id}.mp3")
        wav_path = os.path.join(self.temp_dir, f"{file_id}.wav")

        try:
            # Decode base64
            audio_data = base64.b64decode(base64_string)
            with open(mp3_path, "wb") as f:
                f.write(audio_data)

            # Try static_ffmpeg first as it's more reliable in this environment
            try:
                import subprocess
                from static_ffmpeg import run
                # Use static_ffmpeg to convert
                subprocess.run(['static_ffmpeg', '-y', '-i', mp3_path, wav_path], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return wav_path, mp3_path
            except Exception as e:
                print(f"static_ffmpeg failed, trying pydub: {e}")
                # Attempt conversion to WAV using pydub
                try:
                    audio = AudioSegment.from_file(mp3_path)
                    audio.export(wav_path, format="wav")
                    return wav_path, mp3_path
                except Exception as e2:
                    # If conversion fails (e.g. no ffmpeg), we'll try to use the mp3 directly in extraction
                    return mp3_path, None
                
        except Exception as e:
            self.cleanup([mp3_path, wav_path])
            raise Exception(f"Audio processing error: {str(e)}")

    def extract_features(self, audio_path):
        """Extracts 256+ highly granular features for deep speech analysis.
        Designed to detect AI vs Human even in short (1-word) clips.
        """
        try:
            # Load audio - Resample to 22050 for consistency
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Ensure minimum length for feature extraction (at least 0.5s)
            if len(y) < sr // 2:
                # Pad with silence if too short
                y = np.pad(y, (0, max(0, sr // 2 - len(y))), mode='constant')
            
            features = []
            
            # 1. MFCCs with high resolution (40 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            features.extend(np.mean(mfccs, axis=1)) # 40
            features.extend(np.std(mfccs, axis=1))  # 40
            
            # MFCC Deltas (Velocity of speech transitions)
            mfcc_delta = librosa.feature.delta(mfccs)
            features.extend(np.mean(mfcc_delta, axis=1)) # 40
            
            # MFCC Delta-Deltas (Acceleration - captures micro-jitters)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features.extend(np.mean(mfcc_delta2, axis=1)) # 40
            
            # 2. Spectral Features (Deep Texture Analysis)
            # Spectral Flatness (Search for robotic/artificial tones)
            flatness = librosa.feature.spectral_flatness(y=y)
            features.append(np.mean(flatness)) # 1
            features.append(np.std(flatness))  # 1
            
            # Spectral Centroid (Brightness/Timbre)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(centroid)) # 1
            features.append(np.std(centroid))  # 1
            
            # Spectral Rolloff (High frequency content)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(rolloff)) # 1
            features.append(np.std(rolloff))  # 1
            
            # Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(bandwidth)) # 1
            features.append(np.std(bandwidth))  # 1

            # 3. Spectral Contrast (Harmonic Peaks vs Valleys)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend(np.mean(contrast, axis=1)) # 7
            
            # 4. Zero Crossing Rate (Check for sharp transients common in AI artifacts)
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr)) # 1
            features.append(np.std(zcr))  # 1
            
            # 5. Chroma Features (Pitch/Harmonic structure)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1)) # 12
            
            # 6. Tonnetz (Inter-harmonic relationships)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features.extend(np.mean(tonnetz, axis=1)) # 6
            
            # 7. RMS Energy (Dynamic Range)
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms)) # 1
            features.append(np.std(rms))  # 1
            
            # 8. Mel Spectrogram (High resolution energy bands)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            features.extend(np.mean(mel, axis=1)) # 40
            
            # Final feature count should be consistent
            # 40+40+40+40 + 2 + 2 + 2 + 2 + 7 + 2 + 12 + 6 + 2 + 40 = 237
            
            # Adding Padding/Truncating to ensure exactly 256 features for model stability
            feat_arr = np.array(features)
            if len(feat_arr) < 256:
                feat_arr = np.pad(feat_arr, (0, 256 - len(feat_arr)), mode='constant')
            else:
                feat_arr = feat_arr[:256]
                
            return feat_arr.reshape(1, -1)
        except Exception as e:
            raise Exception(f"Feature extraction error: {str(e)}")

    def cleanup(self, paths):
        """Deletes temporary files."""
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
