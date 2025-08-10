# Voice-Scam-Shield-NaMo

A real-time multilingual voice scam detection system using AI-powered speech recognition and keyword analysis.

## Features

- **Real-time Audio Processing**: Live microphone input with configurable chunk sizes and overlap
- **Multilingual Support**: English, Spanish, French, Hindi, Nepali, and Sanskrit
- **AI-Powered Transcription**: Uses OpenAI Whisper or Faster Whisper for accurate speech-to-text
- **Smart Scam Detection**: Keyword-based analysis with risk scoring and contextual awareness
- **Batch Testing**: Automated testing on audio datasets with ground truth comparison
- **Audio Generation Tools**: Generate test audio files in multiple languages

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Voice-Scam-Shield-NaMo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies (macOS):
```bash
brew install portaudio
```

## Usage

### Real-time Scam Detection

Run the live audio transcription and scam detection:

```bash
cd speech
python speech_processing.py
```

### Batch Testing

Test the system on audio datasets:

```bash
cd speech/scripts
python batch_testing_english.py
python batch_testing_spanish.py
python batch_testing_french.py
python batch_testing_nepali.py
python batch_testing_hi.py
```

### Audio Generation

Generate test audio files in different languages:

```bash
cd speech/tools
python generate_audio_en.py
python generate_audio_es.py
python generate_audio_fr.py
python generate_audio_ne.py
```

### Debug Audio Files

Analyze audio file properties:

```bash
cd speech/tools
python debug_audio.py
```

## Configuration

Edit `speech/config.py` to customize:

- Sample rate and audio parameters
- Model selection (Whisper vs Faster Whisper)
- Language preferences
- Risk thresholds
- Processing parameters

## Project Structure

```
speech/
├── config.py                 # Configuration settings
├── speech_processing.py      # Core transcription and detection logic
├── datasets/                 # Audio test datasets by language
├── keywords/                 # Scam keyword files by language
├── ground_truth/            # Ground truth labels for testing
├── results/                 # Test results output
├── scripts/                 # Batch testing scripts
└── tools/                   # Audio generation and debugging tools
```

## Supported Languages

- **English (en)**: Latin script with comprehensive scam keyword database
- **Spanish (es)**: Latin script with Spanish-specific scam patterns
- **French (fr)**: Latin script with French-specific scam patterns
- **Hindi (hi)**: Devanagari script with Hindi scam keywords
- **Nepali (ne)**: Devanagari script with Nepali-specific patterns
- **Sanskrit (sa)**: Devanagari script support

## Performance

- **Latency**: Typically <2 seconds for audio processing
- **Accuracy**: High accuracy on clear audio with proper keywords
- **Memory**: Efficient streaming audio processing
- **CPU/GPU**: Supports both CPU and GPU acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- OpenAI Whisper for speech recognition
- Faster Whisper for optimized inference
- Community contributors for multilingual keyword databases