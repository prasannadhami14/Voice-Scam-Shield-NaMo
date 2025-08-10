# 🎭 Voice Scam Shield - Setup Guide

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Navigate to the project directory
cd Voice-Scam-Shield-NaMo

# Run the automated startup script
python start_system.py
```

This will:

- Install all required dependencies
- Start the backend server
- Start the frontend application
- Open the system in your browser

### Option 2: Manual Setup

#### 1. Install Dependencies

```bash
# Navigate to the project directory
cd Voice-Scam-Shield-NaMo

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Start Backend Server

```bash
# Navigate to backend directory
cd backend

# Start the FastAPI server
python main.py
```

The backend will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

#### 3. Start Frontend Application

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Start the Streamlit application
streamlit run app.py
```

The frontend will be available at: http://localhost:8501

## 🔧 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Speech        │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   Processing    │
│                 │    │                 │    │                 │
│ • Live Call     │    │ • Audio Routes  │    │ • Transcription │
│ • File Upload   │    │ • Alert Routes  │    │ • Anti-spoofing │
│ • Real-time UI  │    │ • WebSocket     │    │ • Scam Detection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Features

### 🎯 Core Capabilities

- **Real-time Audio Processing**: Live call monitoring with WebRTC
- **Synthetic Voice Detection**: AI-powered detection of fake voices
- **Multi-language Support**: English, Spanish, French, Nepali, Hindi, Sanskrit
- **Scam Keyword Detection**: Identifies suspicious language patterns
- **Live Alerts**: Real-time notifications and warnings
- **File Analysis**: Upload and analyze pre-recorded audio files

### 🔍 Detection Methods

1. **Audio Feature Extraction**: MFCC, spectral, and statistical features
2. **Machine Learning Model**: Random Forest classifier for synthetic voice detection
3. **Keyword Analysis**: Language-specific scam keyword detection
4. **Real-time Transcription**: Whisper-based speech recognition
5. **Risk Assessment**: Multi-factor risk scoring system

## 🌐 API Endpoints

### Backend API (http://localhost:8000)

- `GET /` - System information
- `GET /health` - Health check
- `POST /start_call` - Start call session
- `POST /end_call` - End call session
- `POST /process_file` - Process uploaded audio file

### Audio Routes (/audio)

- `GET /audio/health` - Audio system health
- `POST /audio/process` - Process audio data
- `WebSocket /audio/stream` - Real-time audio streaming

### Alert Routes (/alert)

- `POST /alert/send` - Send alert
- `POST /alert/synthetic-voice` - Synthetic voice alert
- `GET /alert/summary` - Alert system summary
- `GET /alert/health` - Alert system health

## 📁 Project Structure

```
Voice-Scam-Shield-NaMo/
├── backend/                 # FastAPI backend server
│   ├── main.py             # Main application entry point
│   └── routes/             # API route definitions
│       ├── audio.py        # Audio processing routes
│       └── alert.py        # Alert system routes
├── frontend/               # Streamlit frontend application
│   └── app.py             # Main frontend application
├── speech/                 # Speech processing core
│   ├── anti_spoofing/     # Anti-spoofing system
│   ├── config.py          # Configuration settings
│   └── speech_processing.py # Core speech processing
├── requirements.txt        # Python dependencies
├── start_system.py        # Automated startup script
└── SETUP.md              # This setup guide
```

## 🛠️ Requirements

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **OS**: macOS, Linux, or Windows

### Python Dependencies

- **Backend**: FastAPI, Uvicorn, WebSockets
- **Frontend**: Streamlit, Streamlit-WebRTC
- **Audio**: Librosa, SoundDevice, SoundFile
- **ML**: Scikit-learn, Torch, TorchAudio
- **Speech**: Faster-Whisper, OpenAI-Whisper
- **Processing**: NumPy, Pandas, Pydub

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Check what's using the ports
lsof -i :8000  # Backend port
lsof -i :8501  # Frontend port

# Kill processes if needed
kill -9 <PID>
```

#### 2. Import Errors

```bash
# Ensure you're in the correct directory
cd Voice-Scam-Shield-NaMo

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 3. Audio Device Issues

```bash
# Check audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set default device if needed
export AUDIODEVICE="<device_name>"
```

#### 4. Model Loading Errors

```bash
# Check if model files exist
ls -la speech/anti_spoofing/*.pkl

# Verify file permissions
chmod 644 speech/anti_spoofing/*.pkl
```

### Getting Help

1. **Check Logs**: Look at terminal output for error messages
2. **Verify Dependencies**: Ensure all packages are installed correctly
3. **Check Ports**: Verify ports 8000 and 8501 are available
4. **File Permissions**: Ensure model files are readable
5. **Python Version**: Verify you're using Python 3.8+

## 🎯 Next Steps

After successful setup:

1. **Test the System**: Upload an audio file to verify functionality
2. **Configure Alerts**: Set up ElevenLabs API for TTS alerts (optional)
3. **Customize Keywords**: Modify scam keywords in `speech/keywords/`
4. **Deploy**: Consider deploying to production servers
5. **Monitor**: Check system performance and accuracy

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:8000/docs
3. Check the system logs for detailed error information
4. Verify all components are running and healthy

---

**🎭 Voice Scam Shield is now ready to protect against AI-generated voice scams!**
