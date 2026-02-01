from pydub import AudioSegment
import os

try:
    audio = AudioSegment.from_file("test4.mp3")
    print("Success loading with pydub!")
    audio.export("test4_converted.wav", format="wav")
    print("Exported to test4_converted.wav")
except Exception as e:
    print(f"Failed loading with pydub: {str(e)}")
