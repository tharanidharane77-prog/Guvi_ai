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
        """Extracts 128 advanced features for deep speech analysis."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            features = []
            
            # 1. MFCC & Deltas (Vocal Fingerprints)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfccs, axis=1)) # 20
            features.extend(np.std(mfccs, axis=1))  # 20
            
            # MFCC Delta (Transitions)
            mfcc_delta = librosa.feature.delta(mfccs)
            features.extend(np.mean(mfcc_delta, axis=1)) # 20
            
            # 2. Spectral Flatness (Search for Robotic Tones)
            # AI speech often has unnaturally flat or repetitive spectral regions
            flatness = librosa.feature.spectral_flatness(y=y)
            features.append(np.mean(flatness)) # 1
            features.append(np.std(flatness))  # 1
            
            # 3. Spectral Contrast (Harmonics & Filter Analysis)
            # Captures the valley-to-peak ratio in different frequency bands
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend(np.mean(contrast, axis=1)) # 7
            
            # 4. Spectral Centroid & Rolloff
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroids)) # 1
            features.append(np.std(spectral_centroids))  # 1
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff)) # 1
            features.append(np.std(spectral_rolloff))  # 1
            
            # 5. Zero Crossing Rate (Algorithmic Echoes/Artifacts)
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr)) # 1
            features.append(np.std(zcr))  # 1
            
            # 6. Chroma Features (Harmonic Content)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1)) # 12
            features.extend(np.std(chroma, axis=1))  # 12
            
            # 7. RMS Energy (Volume Stability)
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms)) # 1
            features.append(np.std(rms))  # 1
            
            # 8. Mel Spectrogram (Advanced Speech Analysis)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12)
            features.extend(np.mean(mel, axis=1)) # 12
            features.extend(np.std(mel, axis=1))  # 12
            
            # 9. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo)) # 1
            
            # Total to reach 128 (padding if necessary to keep consistent sizing)
            # Currently added: 20+20+20 + 2 + 7 + 2 + 2 + 2 + 12+12 + 2 + 12+12 + 1 = 126
            # Adding spectral bandwidth to reach 128
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(bandwidth))
            features.append(np.std(bandwidth))
            
            return np.array(features).reshape(1, -1)
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
