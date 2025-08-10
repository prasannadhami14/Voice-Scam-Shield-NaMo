# Anti-Spoofing & Synthetic Voice Detection System

This module provides a comprehensive solution for detecting synthetic voices and generating alerts in the Voice Scam Shield system. It's designed as an MVP alternative to pre-trained models like AASIST.

## 🚀 Features

### Core Components

1. **Synthetic Voice Detector** (`synthetic_voice_detector.py`)
   - Extracts 16+ audio features for synthetic voice detection
   - Uses Random Forest classifier for robust detection
   - Trains from scratch without requiring pre-trained models
   - Handles multiple audio formats (WAV, MP3, FLAC)

2. **Alert System** (`alert_system.py`)
   - Integrates with ElevenLabs TTS for audio alerts
   - Multi-language support (English, Spanish, French, Nepali, Hindi, Sanskrit)
   - Real-time alert generation and audio playback
   - Queue-based alert processing

3. **Integrated Detection** (`integrated_detection.py`)
   - Combines scam detection with synthetic voice detection
   - Real-time audio monitoring and processing
   - Comprehensive risk assessment
   - Session history and analytics

4. **Training Tools** (`train_synthetic_detector.py`)
   - Automated training data setup
   - Synthetic sample generation
   - Model training and validation
   - Easy testing and evaluation

## 🛠️ Installation

### Prerequisites

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libsndfile1-dev portaudio19-dev

# Install system dependencies (macOS)
brew install portaudio libsndfile

# Install system dependencies (Windows)
# Use conda or install pre-compiled binaries
```

### Python Dependencies

```bash
# Install required packages
pip3 install -r requirements.txt

# Or install individually
pip3 install numpy librosa scikit-learn joblib soundfile sounddevice
pip3 install requests elevenlabs pydub webrtcvad
```

### ElevenLabs Setup (Optional)

1. Sign up at [ElevenLabs](https://elevenlabs.io/)
2. Get your API key from the dashboard
3. Set environment variable:
   ```bash
   export ELEVENLABS_API_KEY="your_api_key_here"
   ```

## 🎯 Quick Start

### 1. Setup Training Data

```bash
# Create training data structure and generate synthetic samples
python3 train_synthetic_detector.py --setup
```

This creates:
- `training_data/real/` - Add your real voice samples here
- `training_data/synthetic/` - Contains generated synthetic samples

### 2. Add Real Voice Samples

Place real human voice recordings in `training_data/real/`:
- Supported formats: WAV, MP3, FLAC
- Recommended: 3-10 second clips
- Various speakers and speech patterns

### 3. Train the Model

```bash
# Train with default settings
python3 train_synthetic_detector.py --train

# Custom training parameters
python3 train_synthetic_detector.py --train --test-size 0.3 --model my_model.pkl
```

### 4. Test the System

```bash
# Test with an audio file
python3 integrated_detection.py --synthetic-model synthetic_voice_detector.pkl --test-audio test.wav

# Start real-time monitoring
python3 integrated_detection.py --synthetic-model synthetic_voice_detector.pkl --monitor 60
```

## 🔧 Configuration

### Model Parameters

The synthetic voice detector uses these default parameters:
- **Classifier**: Random Forest (100 trees, max depth 10)
- **Features**: 16+ audio characteristics
- **Test split**: 20% for validation
- **Audio format**: 16kHz, mono

### Feature Extraction

The system extracts these audio features:

1. **Spectral Features**
   - MFCCs (13 coefficients)
   - Spectral centroid, rolloff, bandwidth
   - Spectral contrast, flatness

2. **Pitch Features**
   - Fundamental frequency
   - Pitch variation

3. **Temporal Features**
   - Zero crossing rate
   - RMS energy
   - Temporal consistency

4. **Advanced Features**
   - Harmonic-percussive separation
   - Spectral slope analysis
   - Formant-like structure detection

### Alert Configuration

```python
# Customize alert messages
alert_messages = {
    'scam': {
        'en': "Your custom scam message",
        'es': "Tu mensaje personalizado",
        # ... other languages
    }
}

# Customize voice IDs
voice_ids = {
    'en': 'your_voice_id',
    'es': 'your_spanish_voice_id',
    # ... other languages
}
```

## 📊 Performance & Accuracy

### Expected Results

- **Training Time**: 2-5 minutes (depending on dataset size)
- **Inference Time**: <100ms per audio chunk
- **Accuracy**: 70-85% (MVP baseline)
- **False Positive Rate**: <15%
- **False Negative Rate**: <20%

### Improving Accuracy

1. **Increase Training Data**
   ```bash
   # Generate more synthetic samples
   python3 train_synthetic_detector.py --generate-more 50
   ```

2. **Feature Engineering**
   - Modify `extract_features()` in `synthetic_voice_detector.py`
   - Add domain-specific features
   - Adjust feature scaling

3. **Model Tuning**
   - Experiment with different classifiers
   - Adjust Random Forest parameters
   - Try ensemble methods

## 🔍 Usage Examples

### Basic Detection

```python
from synthetic_voice_detector import SyntheticVoiceDetector

# Load trained model
detector = SyntheticVoiceDetector('synthetic_voice_detector.pkl')

# Detect synthetic voice
result = detector.predict('audio_file.wav')
print(f"Is synthetic: {result['is_synthetic']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Alert Generation

```python
from alert_system import AlertSystem

# Initialize alert system
alert_system = AlertSystem(elevenlabs_api_key="your_key")

# Create alert
alert = alert_system.create_alert('scam', 'en')
print(f"Alert message: {alert['message']}")
print(f"Audio path: {alert['audio_path']}")
```

### Real-time Monitoring

```python
from integrated_detection import IntegratedVoiceScamDetector

# Initialize integrated system
detector = IntegratedVoiceScamDetector(
    synthetic_model_path='synthetic_voice_detector.pkl',
    elevenlabs_api_key='your_key'
)

# Start monitoring
detector.start_real_time_monitoring(duration=60)
```

## 🧪 Testing

### Unit Tests

```bash
# Run basic tests
python3 -m pytest test_synthetic_detector.py -v

# Run with coverage
python3 -m pytest --cov=synthetic_voice_detector test_synthetic_detector.py
```

### Integration Tests

```bash
# Test complete pipeline
python3 integrated_detection.py --test-audio sample.wav --save-history

# Test real-time monitoring
python3 integrated_detection.py --monitor 10
```

### Performance Testing

```bash
# Benchmark feature extraction
python3 -c "
import time
from synthetic_voice_detector import SyntheticVoiceDetector
detector = SyntheticVoiceDetector('model.pkl')
start = time.time()
result = detector.predict('test.wav')
print(f'Processing time: {time.time() - start:.3f}s')
"
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip3 install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Audio Processing Errors**
   ```bash
   # Install system audio libraries
   sudo apt-get install libsndfile1-dev portaudio19-dev
   
   # Check audio file format
   file audio_file.wav
   ```

3. **Model Training Issues**
   ```bash
   # Check training data
   ls -la training_data/real/
   ls -la training_data/synthetic/
   
   # Verify file formats
   file training_data/real/*.wav
   ```

4. **ElevenLabs API Issues**
   ```bash
   # Check API key
   echo $ELEVENLABS_API_KEY
   
   # Test API connection
   curl -H "xi-api-key: $ELEVENLABS_API_KEY" \
        "https://api.elevenlabs.io/v1/voices"
   ```

### Performance Optimization

1. **Reduce Audio Chunk Size**
   ```python
   # Smaller chunks for faster processing
   chunk_size = 0.2  # 200ms instead of 300ms
   ```

2. **Feature Selection**
   ```python
   # Use only essential features
   essential_features = ['mfcc', 'pitch', 'spectral_centroid']
   ```

3. **Model Optimization**
   ```python
   # Lighter model for faster inference
   model = RandomForestClassifier(n_estimators=50, max_depth=5)
   ```

## 🔮 Future Enhancements

### Planned Features

1. **Deep Learning Models**
   - CNN-based feature extraction
   - Transformer-based classification
   - Pre-trained audio models

2. **Advanced Detection**
   - Voice cloning detection
   - Deepfake audio identification
   - Real-time adaptation

3. **Enhanced Alerts**
   - Custom TTS voices
   - Multi-channel alerts
   - Alert prioritization

### Research Areas

1. **Feature Engineering**
   - Psychoacoustic features
   - Phase-based analysis
   - Temporal dynamics

2. **Model Architecture**
   - Ensemble methods
   - Online learning
   - Transfer learning

## 📚 References

- **Audio Feature Extraction**: [Librosa Documentation](https://librosa.org/)
- **Machine Learning**: [Scikit-learn User Guide](https://scikit-learn.org/)
- **TTS API**: [ElevenLabs Documentation](https://elevenlabs.io/docs)
- **Audio Processing**: [SoundFile Documentation](https://pysoundfile.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the Voice Scam Shield system and follows the same licensing terms.

---

**Note**: This is an MVP implementation designed for the 24-hour hackathon. For production use, consider using established pre-trained models and additional validation techniques.
