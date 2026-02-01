import requests
import base64
import json
import os

import sys

def test_detection(audio_path='sample.mp3'):
    url = "http://127.0.0.1:5000/detect"
    api_key = "guvi_ai_voice_secret_key"
    
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found. Please place an audio file named '{audio_path}' in the folder.")
        return

    print(f"Reading {audio_path}...")
    with open(audio_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')

    payload = {
        "audio": encoded_string
    }
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    print(f"Sending request for {audio_path} to API...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            print("Successfully received response:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Connection error: {str(e)}")

if __name__ == "__main__":
    target_file = 'sample.mp3'
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    
    test_detection(target_file)
