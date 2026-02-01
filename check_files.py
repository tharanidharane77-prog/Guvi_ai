import librosa
import os

files = ['sample.mp3', 'test2.mp3', 'test3.mp3', 'test4.mp3']

for f in files:
    if os.path.exists(f):
        print(f"Checking {f}...")
        try:
            y, sr = librosa.load(f, sr=None)
            print(f"  Success! Duration: {librosa.get_duration(y=y, sr=sr):.2f}s")
        except Exception as e:
            print(f"  FAILED to load {f}: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"{f} does not exist.")
