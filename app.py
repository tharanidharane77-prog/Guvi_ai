import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from utils.audio_processor import AudioProcessor
from utils.model_handler import ModelHandler

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
processor = AudioProcessor()
model_handler = ModelHandler()

# Configuration
API_KEY = os.getenv('API_KEY', 'guvi_ai_voice_secret_key')

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        "name": "AI Voice Detection API",
        "status": "online",
        "endpoints": {
            "health": "/health (GET)",
            "detect": "/detect (POST)"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Check API health and status"""
    model_loaded = model_handler.model is not None
    scaler_loaded = model_handler.scaler is not None
    
    return jsonify({
        "status": "success",
        "message": "AI Voice Detection API is running",
        "version": "1.0.0",
        "checks": {
            "model_loaded": model_loaded,
            "scaler_loaded": scaler_loaded
        }
    }), 200

@app.route('/detect', methods=['POST'])
def detect_voice():
    """Main endpoint to detect AI vs Human voice"""
    # 1. Authentication
    auth_header = request.headers.get('X-API-KEY')
    if not auth_header or auth_header != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Validation
    data = request.get_json()
    if not data or 'audio' not in data:
        return jsonify({"error": "Missing audio data in base64 format"}), 400
    
    base64_audio = data['audio']
    paths_to_cleanup = []

    try:
        # 3. Audio Processing (Decode & Convert)
        wav_path, mp3_path = processor.base64_to_wav(base64_audio)
        paths_to_cleanup.extend([wav_path, mp3_path])

        # 4. Feature Extraction
        features = processor.extract_features(wav_path)

        # 5. Inference
        result, error = model_handler.predict(features)
        
        if error:
            return jsonify({"error": error}), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Internal process error: {str(e)}"}), 500
    
    finally:
        # 6. Cleanup
        processor.cleanup(paths_to_cleanup)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('temp_audio', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
