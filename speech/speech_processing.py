#!/usr/bin/env python3
"""
Speech processing module for Voice Scam Shield.
Handles audio transcription and basic speech analysis.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import soundfile as sf
    import librosa
    from faster_whisper import WhisperModel
    print("✅ Audio processing libraries imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Some audio libraries not available: {e}")

class SpeechProcessor:
    """
    Speech processor for audio transcription and analysis.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the speech processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.supported_languages = ['en', 'es', 'fr', 'hi', 'ne']
        
        # Initialize Whisper model
        try:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print(f"✅ Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Whisper model: {e}")
            print("Speech recognition will be limited")
    
    async def transcribe_audio(self, audio_path: str, language: str = 'en') -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Expected language code
            
        Returns:
            Dictionary with transcription results
        """
        try:
            if not os.path.exists(audio_path):
                return {
                    'error': f'Audio file not found: {audio_path}',
                    'text': '',
                    'confidence': 0.0,
                    'language': language,
                    'status': 'error'
                }
            
            if not self.model:
                return {
                    'error': 'Whisper model not available',
                    'text': '',
                    'confidence': 0.0,
                    'language': language,
                    'status': 'error'
                }
            
            print(f"🎤 Transcribing audio: {audio_path}")
            
            # Transcribe audio
            segments, info = self.model.transcribe(
                audio_path,
                language=language if language in self.supported_languages else None,
                beam_size=5
            )
            
            # Collect transcription text
            transcription_text = ""
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                transcription_text += segment.text + " "
                total_confidence += segment.avg_logprob
                segment_count += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0
            # Convert log probability to confidence (0-1 scale)
            confidence = min(1.0, max(0.0, (avg_confidence + 1.0) / 2.0))
            
            # Clean up text
            transcription_text = transcription_text.strip()
            
            # Detect actual language if not specified
            detected_language = info.language if info.language else language
            
            result = {
                'text': transcription_text,
                'confidence': confidence,
                'language': detected_language,
                'segments': segment_count,
                'status': 'success',
                'file_path': audio_path
            }
            
            print(f"✅ Transcription complete: {len(transcription_text)} characters")
            print(f"   Language: {detected_language}")
            print(f"   Confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return {
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'language': language,
                'status': 'error',
                'file_path': audio_path
            }
    
    def detect_language(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect the language of an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with language detection results
        """
        try:
            if not self.model:
                return {
                    'error': 'Whisper model not available',
                    'language': 'unknown',
                    'confidence': 0.0
                }
            
            # Use Whisper to detect language
            segments, info = self.model.transcribe(audio_path, language=None)
            
            return {
                'language': info.language,
                'confidence': info.language_probability,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'language': 'unknown',
                'confidence': 0.0
            }
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        Get basic information about an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get basic statistics
            amplitude_mean = float(np.mean(np.abs(y)))
            amplitude_std = float(np.std(np.abs(y)))
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[1],
                'samples': len(y),
                'amplitude_mean': amplitude_mean,
                'amplitude_std': amplitude_std,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def is_valid_audio(self, audio_path: str) -> bool:
        """
        Check if a file is a valid audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if valid audio, False otherwise
        """
        try:
            # Try to load with librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) > 0 and sr > 0
        except:
            return False
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_size': self.model_size,
            'model_loaded': self.model is not None,
            'supported_languages': self.supported_languages
        }


class RealTimeTranscriber:
    """
    Minimal real-time transcriber for MVP streaming route.
    Converts numpy audio chunks into a temporary WAV, transcribes,
    and computes a simple risk score based on scam keywords.
    """

    def __init__(self, config):
        self.config = config
        self.speech_processor = SpeechProcessor(model_size=getattr(config, 'model_name', 'base'))
        # Load keyword sets
        try:
            from .config import scam_keywords
            self.scam_keywords = scam_keywords
        except Exception:
            self.scam_keywords = {lang: set() for lang in ['en', 'es', 'fr', 'hi', 'ne']}

    def _score_keywords(self, text: str, language: str) -> dict:
        if not text:
            return {"keywords_found": [], "keyword_score": 0.0, "is_suspicious": False}
        kws = self.scam_keywords.get(language, set())
        text_lower = text.lower()
        found = [kw for kw in kws if kw in text_lower]
        total = max(1, len(kws))
        score = len(found) / total
        return {
            "keywords_found": found,
            "keyword_score": score,
            "is_suspicious": score > 0.06  # ~top 6% of list
        }

    def process_audio_chunk(self, audio_data: 'np.ndarray', language: str = 'en') -> dict:
        """Synchronous wrapper for non-async contexts. Avoid in asyncio loops."""
        try:
            # If already inside an event loop, raise to prevent deadlock
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError("process_audio_chunk called inside running event loop; use process_audio_chunk_async")
            except RuntimeError:
                # No running loop -> safe to create one
                pass

            return asyncio.run(self.process_audio_chunk_async(audio_data, language))
        except Exception as e:
            return {"risk": 0, "label": "Error", "rationale": str(e)}

    async def process_audio_chunk_async(self, audio_data: 'np.ndarray', language: str = 'en') -> dict:
        try:
            import soundfile as sf
            import numpy as np
            import tempfile
            # Ensure mono float32
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
            audio_data = audio_data.astype(np.float32, copy=False)

            # Write temp wav with configured sample rate
            sr = getattr(self.config, 'sample_rate', 16000)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, sr)
                temp_path = tmp.name

            # Transcribe
            trans = await self.speech_processor.transcribe_audio(temp_path, language)

            # Keyword analysis
            kw = self._score_keywords(trans.get('text', ''), trans.get('language', language))

            # Simple risk
            risk = int(min(100, max(0, kw.get('keyword_score', 0.0) * 100)))
            label = 'Scam' if risk >= 60 else 'Safe'
            rationale = 'Keywords suggest potential scam' if kw.get('is_suspicious') else 'No suspicious keywords detected'

            return {
                "risk": risk,
                "label": label,
                "rationale": rationale,
                "transcription": trans,
                "keywords": kw
            }
        except Exception as e:
            return {"risk": 0, "label": "Error", "rationale": str(e)}


async def main():
    """Test the speech processor."""
    processor = SpeechProcessor()
    
    # Test with a sample audio file if available
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"Testing with: {test_file}")
        
        # Get audio info
        info = processor.get_audio_info(test_file)
        print(f"Audio info: {info}")
        
        # Detect language
        lang_result = processor.detect_language(test_file)
        print(f"Language detection: {lang_result}")
        
        # Transcribe
        trans_result = await processor.transcribe_audio(test_file, 'en')
        print(f"Transcription: {trans_result}")
    else:
        print("No test file found. Create a test_audio.wav file to test the processor.")
        print("Model info:", processor.get_model_info())


if __name__ == "__main__":
    asyncio.run(main())