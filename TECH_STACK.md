# AI Voice Detection API - Complete Tech Stack Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Core Technologies](#core-technologies)
3. [Backend Stack](#backend-stack)
4. [Audio Processing Stack](#audio-processing-stack)
5. [Machine Learning Stack](#machine-learning-stack)
6. [Deployment Stack](#deployment-stack)
7. [Development Tools](#development-tools)
8. [System Architecture](#system-architecture)
9. [Feature Extraction Pipeline](#feature-extraction-pipeline)
10. [Dependencies Breakdown](#dependencies-breakdown)
11. [Infrastructure](#infrastructure)
12. [Security Stack](#security-stack)
13. [Performance Optimization](#performance-optimization)
14. [Monitoring & Logging](#monitoring--logging)
15. [Version Information](#version-information)

---

## Overview

This document provides a comprehensive overview of all technologies, libraries, frameworks, and tools used in the AI Voice Detection API project. The system is designed to detect AI-generated voices vs human voices across 5 languages using advanced audio processing and machine learning.

**Project Type**: REST API for Audio Classification  
**Primary Language**: Python 3.10+  
**Architecture**: Microservice, Containerized  
**Deployment**: Multi-cloud ready  

---

## Core Technologies

### Programming Language
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Primary programming language for backend, ML, and audio processing |

**Why Python?**
- Rich ecosystem for ML and audio processing
- Excellent libraries (librosa, scikit-learn)
- Easy deployment and scaling
- Strong community support

---

## Backend Stack

### Web Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| **Flask** | 3.0.0 | Lightweight web framework for REST API |
| **Flask-CORS** | 4.0.0 | Cross-Origin Resource Sharing support |

**Flask Features Used**:
- Route decorators (`@app.route`)
- JSON request/response handling
- Error handlers (`@app.errorhandler`)
- Request validation
- Header-based authentication

### Production Server
| Technology | Version | Purpose |
|------------|---------|---------|
| **Gunicorn** | 21.2.0 | Production WSGI HTTP server |

**Gunicorn Configuration**:
```bash
--bind 0.0.0.0:$PORT
--workers 4
--timeout 120
--worker-class sync
```

**Benefits**:
- Multi-worker process management
- Graceful worker restarts
- Request timeout handling
- Production-grade stability
- Load balancing across workers

---

## Audio Processing Stack

### Primary Audio Libraries

#### 1. librosa
| Technology | Version | Purpose |
|------------|---------|---------|
| **librosa** | 0.10.1 | Advanced audio analysis and feature extraction |

**librosa Features Used**:
```python
# Core Functions
librosa.load()                          # Load audio files
librosa.feature.mfcc()                  # MFCC extraction
librosa.feature.spectral_centroid()     # Spectral analysis
librosa.feature.spectral_rolloff()      # Frequency rolloff
librosa.feature.zero_crossing_rate()    # ZCR calculation
librosa.feature.chroma_stft()           # Chroma features
librosa.beat.beat_track()               # Tempo detection
librosa.feature.rms()                   # RMS energy
```

**Why librosa?**
- Industry standard for audio ML
- Comprehensive feature extraction
- Optimized for music/speech analysis
- Excellent documentation

#### 2. pydub
| Technology | Version | Purpose |
|------------|---------|---------|
| **pydub** | 0.25.1 | Audio format conversion and manipulation |

**pydub Features Used**:
```python
AudioSegment.from_mp3()    # Load MP3 files
.export()                  # Convert to WAV
```

**Why pydub?**
- Simple API for format conversion
- Works seamlessly with FFmpeg
- Handles various audio formats
- Minimal code required

#### 3. soundfile
| Technology | Version | Purpose |
|------------|---------|---------|
| **soundfile** | 0.12.1 | Audio file I/O operations |

**Purpose**:
- Low-level audio file reading/writing
- Support for multiple formats
- Required dependency for librosa

### System Audio Dependencies

#### FFmpeg
| Technology | Type | Purpose |
|------------|------|---------|
| **FFmpeg** | System Binary | Universal audio/video codec library |

**Capabilities**:
- MP3 decoding
- Audio format conversion
- Codec support (MP3, WAV, AAC, etc.)
- Audio stream processing
- Audio filtering

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Docker
RUN apt-get install -y ffmpeg
```

#### libsndfile1
| Technology | Type | Purpose |
|------------|------|---------|
| **libsndfile1** | System Library | C library for audio file I/O |

**Purpose**:
- Backend for soundfile Python package
- Supports WAV, AIFF, AU, and other formats

---

## Machine Learning Stack

### ML Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| **scikit-learn** | 1.3.2 | Machine learning algorithms and preprocessing |

**scikit-learn Components Used**:

1. **Classifiers**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn.neural_network import MLPClassifier
   ```

2. **Preprocessing**:
   ```python
   from sklearn.preprocessing import StandardScaler
   ```

3. **Model Persistence**:
   ```python
   from sklearn.externals import joblib  # Via joblib
   ```

**Classifier Details**:

#### Random Forest (Primary Model)
```python
RandomForestClassifier(
    n_estimators=200,      # 200 decision trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=2,    # Min samples per leaf
    random_state=42        # Reproducibility
)
```

#### Gradient Boosting (Alternative)
```python
GradientBoostingClassifier(
    n_estimators=150,      # 150 boosting stages
    learning_rate=0.1,     # Learning rate
    max_depth=10,          # Max tree depth
    random_state=42
)
```

#### Neural Network (Alternative)
```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',                  # ReLU activation
    solver='adam',                      # Adam optimizer
    max_iter=500,                       # Max iterations
    random_state=42
)
```

### Numerical Computing
| Technology | Version | Purpose |
|------------|---------|---------|
| **NumPy** | 1.24.3 | Array operations and mathematical functions |

**NumPy Usage**:
```python
# Feature manipulation
np.mean()          # Calculate means
np.std()           # Calculate standard deviations
np.clip()          # Clip values to range
np.random.normal() # Add noise/randomness
np.array()         # Array creation
```

### Model Serialization
| Technology | Version | Purpose |
|------------|---------|---------|
| **joblib** | 1.3.2 | Efficient model save/load operations |

**Usage**:
```python
# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')
```

**Why joblib?**
- Efficient for large NumPy arrays
- Better than pickle for ML models
- Optimized compression

---

## Deployment Stack

### Containerization

#### Docker
| Technology | Type | Purpose |
|------------|------|---------|
| **Docker** | Container Platform | Application containerization |

**Dockerfile Configuration**:
```dockerfile
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY . .
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", 
     "--workers", "4", "--timeout", "120", "app:app"]
```

**Base Image**: `python:3.10-slim`
- Minimal Debian-based Python image
- Smaller size (~150MB vs 1GB)
- Faster builds and deployments

#### Docker Compose
| Technology | Version | Purpose |
|------------|---------|---------|
| **Docker Compose** | 2.0+ | Multi-container orchestration |

**Configuration**:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - API_KEY=${API_KEY}
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Supported Cloud Platforms

#### 1. Heroku
**Stack**: Container (Docker) or Buildpack  
**Features**:
- Auto-scaling
- Add-ons ecosystem
- Easy deployment via Git
- Built-in monitoring

**Required Files**:
- `Procfile`: Process type definitions
- `runtime.txt`: Python version
- `Aptfile`: System dependencies

#### 2. AWS EC2
**Stack**: Virtual Machine  
**Features**:
- Full control over environment
- Custom instance types
- VPC networking
- Auto Scaling Groups

**Deployment**: Docker or native Python

#### 3. Google Cloud Run
**Stack**: Serverless Containers  
**Features**:
- Auto-scaling (0 to N)
- Pay per request
- Fully managed
- Container-based

**Deployment**: Container Registry + Cloud Run

#### 4. DigitalOcean
**Stack**: Droplets (VMs)  
**Features**:
- Simple pricing
- SSD storage
- Easy firewall rules
- One-click apps

**Deployment**: Docker on Droplet

#### 5. Railway.app
**Stack**: Modern PaaS  
**Features**:
- Auto-deploy from GitHub
- Built-in CI/CD
- Environment variables
- Simple pricing

#### 6. Render.com
**Stack**: Cloud Platform  
**Features**:
- Auto-deploy from Git
- Native Docker support
- Free SSL
- DDoS protection

---

## Development Tools

### Version Control
| Technology | Purpose |
|------------|---------|
| **Git** | Source code version control |
| **.gitignore** | Exclude files from version control |

### Package Management
| Technology | Purpose |
|------------|---------|
| **pip** | Python package installer |
| **virtualenv** | Isolated Python environments |
| **requirements.txt** | Dependency specification |

### Configuration Management
| Technology | Version | Purpose |
|------------|---------|---------|
| **python-dotenv** | 1.0.0 | Load environment variables from .env files |

**Usage**:
```python
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
```

### Testing Tools
| Technology | Version | Purpose |
|------------|---------|---------|
| **requests** | 2.31.0 | HTTP library for API testing |

**Test Script Features**:
- Automated API testing
- Base64 encoding
- Error handling
- Retry logic
- Response validation

---

## System Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CLIENT LAYER                        │
│  (Browser, Mobile App, API Consumer, Test Scripts)  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/HTTPS Request
                     │ (Base64 MP3 + API Key)
                     ▼
┌─────────────────────────────────────────────────────┐
│               LOAD BALANCER (Optional)               │
│        (Cloud LB, Nginx, HAProxy, etc.)             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              GUNICORN WSGI SERVER                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Worker 1 │ │ Worker 2 │ │ Worker N │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│        Multi-process, Load Balanced                  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              FLASK APPLICATION LAYER                 │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │         API Gateway & Routing               │   │
│  │  • Route handling (/detect, /health)       │   │
│  │  • Request parsing                          │   │
│  │  • Response formatting                      │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │         Authentication Layer                │   │
│  │  • API key validation                       │   │
│  │  • Bearer token parsing                     │   │
│  │  • Authorization checks                     │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │         Validation Layer                    │   │
│  │  • Input validation                         │   │
│  │  • Language verification                    │   │
│  │  • Base64 format check                      │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           AUDIO PROCESSING LAYER                     │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │      Base64 Decoder (Python base64)         │   │
│  │  • Decode Base64 string to binary          │   │
│  │  • Error handling for invalid encoding     │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Temporary File Manager (tempfile)        │   │
│  │  • Create temp MP3 file                     │   │
│  │  • Manage file lifecycle                    │   │
│  │  • Cleanup after processing                 │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │   Format Converter (pydub + FFmpeg)         │   │
│  │  • MP3 → WAV conversion                     │   │
│  │  • Sample rate normalization                │   │
│  │  • Channel handling                         │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Audio Loader (librosa)                   │   │
│  │  • Load WAV file                            │   │
│  │  • Resample audio                           │   │
│  │  • Normalize amplitude                      │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Feature Extractor (librosa)              │   │
│  │  • MFCC (20 coefficients)                   │   │
│  │  • Spectral Centroid                        │   │
│  │  • Spectral Rolloff                         │   │
│  │  • Zero Crossing Rate                       │   │
│  │  • Chroma Features                          │   │
│  │  • Tempo Detection                          │   │
│  │  • RMS Energy                               │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          MACHINE LEARNING LAYER                      │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │    Feature Preprocessing (NumPy)            │   │
│  │  • Flatten feature arrays                   │   │
│  │  • Normalize features                       │   │
│  │  • Handle missing values                    │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Feature Scaling (StandardScaler)         │   │
│  │  • Scale to zero mean, unit variance        │   │
│  │  • Apply saved scaler transform             │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Model Inference (scikit-learn)           │   │
│  │  • Random Forest prediction                 │   │
│  │  • Probability estimation                   │   │
│  │  • Confidence calculation                   │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Classification Logic                     │   │
│  │  • Threshold application (0.5)              │   │
│  │  • AI_GENERATED vs HUMAN                    │   │
│  │  • Confidence score (0.0-1.0)               │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              RESPONSE LAYER                          │
│  ┌─────────────────────────────────────────────┐   │
│  │    JSON Formatter                           │   │
│  │  • Format classification result             │   │
│  │  • Round confidence to 2 decimals           │   │
│  │  • Add metadata (if needed)                 │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Error Handler                            │   │
│  │  • Catch exceptions                         │   │
│  │  • Format error messages                    │   │
│  │  • Set appropriate HTTP status              │   │
│  └─────────────────────────────────────────────┘   │
│                     │                                 │
│                     ▼                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │    Logger                                   │   │
│  │  • Log requests                             │   │
│  │  • Log predictions                          │   │
│  │  • Log errors                               │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 CLIENT RESPONSE                      │
│  {                                                    │
│    "classification": "AI_GENERATED" | "HUMAN",       │
│    "confidence": 0.87                                │
│  }                                                    │
└─────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Audio Input (Base64 MP3)
    │
    ├─→ [1] Base64 Decode
    │       └─→ Binary MP3 Data
    │
    ├─→ [2] Save to Temp File
    │       └─→ /tmp/audio_xyz.mp3
    │
    ├─→ [3] FFmpeg Convert
    │       └─→ /tmp/audio_xyz.wav
    │
    ├─→ [4] Librosa Load
    │       └─→ NumPy Array (waveform)
    │       └─→ Sample Rate (Hz)
    │
    ├─→ [5] Feature Extraction
    │       ├─→ MFCC: [20 coefficients]
    │       ├─→ Spectral Centroid: [mean, std]
    │       ├─→ Spectral Rolloff: [mean, std]
    │       ├─→ Zero Crossing Rate: [mean, std]
    │       ├─→ Chroma: [12 features]
    │       ├─→ Tempo: [single value]
    │       └─→ RMS Energy: [mean, std]
    │
    ├─→ [6] Feature Vector
    │       └─→ [40+ features flattened]
    │
    ├─→ [7] Scaling
    │       └─→ StandardScaler transform
    │
    ├─→ [8] Model Prediction
    │       ├─→ Random Forest forward pass
    │       └─→ Probability: [P(human), P(ai)]
    │
    ├─→ [9] Classification
    │       ├─→ If P(ai) > 0.5: "AI_GENERATED"
    │       └─→ Else: "HUMAN"
    │
    └─→ [10] Response
            └─→ {"classification": "...", "confidence": 0.XX}
```

---

## Feature Extraction Pipeline

### Detailed Audio Features

#### 1. MFCC (Mel-Frequency Cepstral Coefficients)
```python
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
mfcc_mean = np.mean(mfccs, axis=1)  # 20 values
mfcc_std = np.std(mfccs, axis=1)    # 20 values
```
**Purpose**: Capture spectral envelope, phonetic content  
**Output**: 40 features (20 means + 20 stds)  
**Why**: Distinguishes voice characteristics

#### 2. Spectral Centroid
```python
spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
centroid_mean = np.mean(spectral_centroids)
centroid_std = np.std(spectral_centroids)
```
**Purpose**: Measure "brightness" of sound  
**Output**: 2 features (mean, std)  
**Why**: AI voices often have more consistent brightness

#### 3. Spectral Rolloff
```python
spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
rolloff_mean = np.mean(spectral_rolloff)
rolloff_std = np.std(spectral_rolloff)
```
**Purpose**: Frequency below which 85% of energy is contained  
**Output**: 2 features  
**Why**: Indicates voice fullness and naturalness

#### 4. Zero Crossing Rate
```python
zcr = librosa.feature.zero_crossing_rate(audio)
zcr_mean = np.mean(zcr)
zcr_std = np.std(zcr)
```
**Purpose**: Rate at which signal changes sign  
**Output**: 2 features  
**Why**: Measures noisiness vs tonality

#### 5. Chroma Features
```python
chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
chroma_mean = np.mean(chroma, axis=1)  # 12 values
chroma_std = np.std(chroma, axis=1)    # 12 values
```
**Purpose**: Pitch class representation  
**Output**: 24 features (12 means + 12 stds)  
**Why**: Captures harmonic content

#### 6. Tempo
```python
tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
```
**Purpose**: Estimated tempo in BPM  
**Output**: 1 feature  
**Why**: Speaking rate can indicate AI generation

#### 7. RMS Energy
```python
rms = librosa.feature.rms(y=audio)
rms_mean = np.mean(rms)
rms_std = np.std(rms)
```
**Purpose**: Root mean square energy  
**Output**: 2 features  
**Why**: AI voices often have more consistent energy

### Total Feature Count
```
MFCC:               40 features (20 mean + 20 std)
Spectral Centroid:   2 features (mean + std)
Spectral Rolloff:    2 features (mean + std)
Zero Crossing Rate:  2 features (mean + std)
Chroma:            24 features (12 mean + 12 std)
Tempo:              1 feature
RMS Energy:         2 features (mean + std)
─────────────────────────────────────────────
TOTAL:             73 features
```

---

## Dependencies Breakdown

### Python Package Dependencies

```plaintext
requirements.txt
├── flask==3.0.0
│   ├── Werkzeug (WSGI utility)
│   ├── Jinja2 (template engine)
│   ├── click (CLI)
│   └── itsdangerous (signing)
│
├── flask-cors==4.0.0
│   └── flask (peer dependency)
│
├── numpy==1.24.3
│   └── (C extensions for speed)
│
├── librosa==0.10.1
│   ├── numpy
│   ├── scipy
│   ├── scikit-learn
│   ├── joblib
│   ├── decorator
│   ├── audioread
│   ├── soundfile
│   ├── pooch
│   ├── soxr
│   ├── typing-extensions
│   ├── lazy-loader
│   └── msgpack
│
├── soundfile==0.12.1
│   ├── cffi (C Foreign Function Interface)
│   └── (requires libsndfile1 system library)
│
├── pydub==0.25.1
│   └── (requires ffmpeg system binary)
│
├── scikit-learn==1.3.2
│   ├── numpy
│   ├── scipy
│   ├── joblib
│   └── threadpoolctl
│
├── joblib==1.3.2
│   └── (minimal dependencies)
│
├── gunicorn==21.2.0
│   └── packaging
│
├── python-dotenv==1.0.0
│   └── (no dependencies)
│
└── requests==2.31.0
    ├── charset-normalizer
    ├── idna
    ├── urllib3
    └── certifi
```

### System Dependencies

```plaintext
System Packages (Ubuntu/Debian)
├── ffmpeg
│   ├── libavcodec (codecs)
│   ├── libavformat (formats)
│   ├── libavutil (utilities)
│   └── libswresample (resampling)
│
├── libsndfile1
│   ├── libogg
│   ├── libvorbis
│   └── libflac
│
├── gcc (C compiler)
│   └── (for compiling C extensions)
│
└── g++ (C++ compiler)
    └── (for C++ dependencies)
```

---

## Infrastructure

### Container Infrastructure

```
Docker Container
├── Base: python:3.10-slim (Debian-based)
│   ├── OS: Debian 11 (Bullseye)
│   ├── Python: 3.10.x
│   └── Size: ~150MB
│
├── System Layer
│   ├── ffmpeg (~50MB)
│   ├── libsndfile1 (~2MB)
│   ├── gcc/g++ (~100MB)
│   └── Other dependencies
│
├── Python Layer
│   ├── pip packages (~500MB)
│   └── Compiled extensions
│
└── Application Layer
    ├── app.py
    ├── model.py
    ├── models/ (directory)
    └── Configuration files

Total Container Size: ~800MB - 1GB
```

### Network Architecture

```
Internet
    │
    ▼
[Cloud Load Balancer]
    │
    ├─→ [Container Instance 1] :5000
    ├─→ [Container Instance 2] :5000
    └─→ [Container Instance N] :5000
         │
         ├─→ [Gunicorn Master Process]
         │    ├─→ Worker 1 (handles requests)
         │    ├─→ Worker 2 (handles requests)
         │    ├─→ Worker 3 (handles requests)
         │    └─→ Worker 4 (handles requests)
         │
         └─→ [Shared Storage]
              └─→ /app/models/ (ML models)
```

### Storage Architecture

```
Persistent Storage
├── /app/models/
│   ├── voice_detection_model.pkl (~50MB)
│   └── scaler.pkl (~1MB)
│
Temporary Storage
└── /tmp/
    ├── audio_*.mp3 (cleaned up after use)
    └── audio_*.wav (cleaned up after use)
```

---

## Security Stack

### Authentication
- **Method**: API Key (Bearer Token)
- **Header**: `Authorization: Bearer <key>` or `X-API-Key: <key>`
- **Storage**: Environment variables (`.env`)
- **Validation**: Request middleware

### Input Validation
```python
# Request validation
- Content-Type: application/json
- Required fields: audio (base64 string)
- Optional fields: language (enum)
- Max payload size: ~10MB (configurable)
```

### CORS Configuration
```python
CORS(app)  # Allows cross-origin requests
# Can be configured for specific origins
```

### Environment Variables
```bash
# Sensitive data stored in .env (not committed)
API_KEY=<secure-random-string>
FLASK_ENV=production
PORT=5000
```

### Error Handling
- Sanitized error messages
- No stack traces in production
- Logged errors for debugging
- Generic user-facing errors

---

## Performance Optimization

### Multi-Processing
```python
# Gunicorn workers
--workers 4

# Formula: (2 x CPU cores) + 1
# Example: 2 cores = 5 workers
```

### Caching Opportunities
- Feature extraction results
- Model predictions for identical audio
- Scaler transformations

### Optimization Techniques
1. **Lazy Loading**: Models loaded once at startup
2. **Worker Pooling**: Gunicorn manages worker processes
3. **Efficient I/O**: Temporary file cleanup
4. **NumPy Operations**: Vectorized computations
5. **Compressed Models**: joblib with compression

### Resource Usage
```
Per Request:
├── Memory: ~100-200MB (peak during feature extraction)
├── CPU: ~1-2 seconds (feature extraction + inference)
└── Disk: ~5-10MB (temporary audio files)

Per Container:
├── Memory: ~500MB-1GB (base + workers)
├── CPU: 0.5-2 cores (depending on load)
└── Disk: ~1GB (container + models)
```

---

## Monitoring & Logging

### Logging Framework
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Log Levels
- **INFO**: Request received, prediction made
- **WARNING**: Invalid input, authentication failure
- **ERROR**: Processing errors, model failures
- **DEBUG**: Detailed execution flow (development only)

### Health Monitoring
```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model.is_loaded()
    }
```

### Metrics to Monitor
- Request rate (requests/second)
- Response time (ms)
- Error rate (%)
- CPU usage (%)
- Memory usage (MB)
- Model inference time (ms)
- Feature extraction time (ms)

### Logging Best Practices
```python
# Log important events
logger.info(f"Prediction: {classification} (confidence: {confidence:.2f})")

# Log errors with context
logger.error(f"Error processing audio: {str(e)}", exc_info=True)

# Log authentication failures
logger.warning(f"Invalid API key attempt from {request.remote_addr}")
```

---

## Version Information

### Complete Version Matrix

| Component | Version | Release Date | Status |
|-----------|---------|--------------|--------|
| **Python** | 3.10+ | Oct 2021 | Stable |
| **Flask** | 3.0.0 | Sep 2023 | Stable |
| **Flask-CORS** | 4.0.0 | Jul 2023 | Stable |
| **NumPy** | 1.24.3 | Apr 2023 | Stable |
| **librosa** | 0.10.1 | Sep 2023 | Stable |
| **soundfile** | 0.12.1 | Sep 2022 | Stable |
| **pydub** | 0.25.1 | Mar 2021 | Stable |
| **scikit-learn** | 1.3.2 | Oct 2023 | Stable |
| **joblib** | 1.3.2 | Aug 2023 | Stable |
| **Gunicorn** | 21.2.0 | Jul 2023 | Stable |
| **python-dotenv** | 1.0.0 | May 2023 | Stable |
| **requests** | 2.31.0 | May 2023 | Stable |
| **Docker** | 20.10+ | - | Recommended |
| **Docker Compose** | 2.0+ | - | Recommended |
| **FFmpeg** | 4.0+ | - | Required |

### Compatibility Matrix

| Python Version | Supported |
|----------------|-----------|
| 3.10.x | ✅ Recommended |
| 3.11.x | ✅ Compatible |
| 3.12.x | ✅ Compatible |
| 3.9.x | ⚠️ May work |
| 3.8.x | ❌ Not tested |

| OS | Docker | Native |
|----|--------|--------|
| Linux (Ubuntu 20.04+) | ✅ | ✅ |
| Linux (Debian 11+) | ✅ | ✅ |
| macOS (11+) | ✅ | ✅ |
| Windows 10/11 | ✅ | ⚠️ |

---

## Technology Choices Rationale

### Why Flask?
- ✅ Lightweight and simple
- ✅ Minimal boilerplate
- ✅ Easy to deploy
- ✅ Large ecosystem
- ✅ Perfect for microservices

### Why librosa?
- ✅ Industry standard
- ✅ Comprehensive features
- ✅ Well-documented
- ✅ Active development
- ✅ Optimized for audio ML

### Why scikit-learn?
- ✅ Production-ready
- ✅ Easy to use
- ✅ Wide algorithm support
- ✅ Good documentation
- ✅ Excellent for traditional ML

### Why Docker?
- ✅ Consistent environments
- ✅ Easy deployment
- ✅ Portable
- ✅ Scalable
- ✅ Industry standard

### Why Gunicorn?
- ✅ Production-grade
- ✅ Multi-worker support
- ✅ Stable and reliable
- ✅ Easy configuration
- ✅ Good performance

---

## Alternative Technologies Considered

### Alternatives We Could Use

#### Web Frameworks
- **FastAPI**: Async, modern, but overkill for this use case
- **Django**: Too heavy for simple API
- **Tornado**: Async, but unnecessary complexity

#### Audio Processing
- **torchaudio**: PyTorch-based, requires GPU for best performance
- **Essentia**: More comprehensive, but larger footprint
- **pyAudioAnalysis**: Simpler, but less feature-rich

#### ML Frameworks
- **TensorFlow**: Overkill, requires more resources
- **PyTorch**: Better for deep learning, not needed here
- **XGBoost**: Good alternative, similar performance

#### Deployment
- **Kubernetes**: Overkill for single service
- **Serverless (Lambda)**: Cold start issues with large models
- **Apache/Nginx**: More complex setup

---

## Upgrade Path

### Future Enhancements

#### Short-term (1-3 months)
- [ ] Add Redis caching
- [ ] Implement rate limiting
- [ ] Add request queuing
- [ ] Database for logging

#### Medium-term (3-6 months)
- [ ] Deep learning model (PyTorch/TensorFlow)
- [ ] Real-time WebSocket support
- [ ] Batch processing endpoint
- [ ] Model versioning system

#### Long-term (6-12 months)
- [ ] Kubernetes deployment
- [ ] Microservices architecture
- [ ] A/B testing framework
- [ ] Auto-scaling based on load

---

## Summary

### Tech Stack at a Glance

```
┌─────────────────────────────────────┐
│     PRODUCTION ENVIRONMENT          │
├─────────────────────────────────────┤
│ Platform: Docker Container          │
│ Server: Gunicorn (4 workers)       │
│ Framework: Flask 3.0.0              │
│ Language: Python 3.10+              │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      AUDIO PROCESSING               │
├─────────────────────────────────────┤
│ Analysis: librosa 0.10.1            │
│ Conversion: pydub 0.25.1            │
│ I/O: soundfile 0.12.1               │
│ Codec: FFmpeg 4.0+                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      MACHINE LEARNING               │
├─────────────────────────────────────┤
│ ML: scikit-learn 1.3.2              │
│ Computing: NumPy 1.24.3             │
│ Persistence: joblib 1.3.2           │
│ Model: Random Forest (200 trees)    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│        DEPLOYMENT                   │
├─────────────────────────────────────┤
│ Container: Docker 20.10+            │
│ Orchestration: Docker Compose 2.0+  │
│ Clouds: Heroku, AWS, GCP, DO        │
└─────────────────────────────────────┘
```

---

## Conclusion

This tech stack is designed for:
- ✅ **Simplicity**: Easy to understand and maintain
- ✅ **Reliability**: Production-tested components
- ✅ **Scalability**: Can handle increasing load
- ✅ **Performance**: Optimized for audio processing
- ✅ **Portability**: Works on any platform
- ✅ **Maintainability**: Clear architecture and documentation

**Total Lines of Code**: ~2,000+  
**Total Documentation**: ~2,600+ lines  
**Dependencies**: 11 Python packages + 2 system packages  
**Deployment Options**: 6+ platforms  
**Container Size**: ~800MB - 1GB  

---

**Document Version**: 1.0.0  
**Last Updated**: February 2025  
**Status**: Production Ready ✅
