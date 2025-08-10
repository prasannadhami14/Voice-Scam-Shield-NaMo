#!/usr/bin/env python3
"""
Integrated detection system for Voice Scam Shield.
Combines speech recognition, keyword detection, and synthetic voice detection.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our simplified synthetic voice detector
from .simple_synthetic_detector import SimpleSyntheticDetector

# Import speech processing
from speech.speech_processing import SpeechProcessor

class IntegratedDetectionSystem:
    """
    Integrated detection system that combines multiple detection methods.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the integrated detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.speech_processor = SpeechProcessor()
        self.synthetic_detector = SimpleSyntheticDetector()
        
        # Load keyword databases
        self.keywords = self._load_keywords()
        
        print("✅ Integrated detection system initialized")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'languages': ['en', 'es', 'fr', 'hi', 'ne'],
            'confidence_threshold': 0.7,
            'synthetic_threshold': 0.4,
            'keyword_threshold': 0.6
        }
    
    def _load_keywords(self) -> Dict[str, List[str]]:
        """Load keyword databases for different languages."""
        keywords = {}
        # Prefer keywords inside speech/ if present; otherwise fallback to repo-level keywords/
        base_dir = os.path.dirname(__file__)
        primary_dir = os.path.abspath(os.path.join(base_dir, '..', 'keywords'))
        fallback_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'keywords'))
        
        for lang in self.config['languages']:
            # Try primary location first
            keyword_file = os.path.join(primary_dir, f'scam_keywords_{lang}.json')
            if not os.path.exists(keyword_file):
                # Try fallback location at repo root
                alt = os.path.join(fallback_dir, f'scam_keywords_{lang}.json')
                if os.path.exists(alt):
                    keyword_file = alt
            
            if os.path.exists(keyword_file):
                try:
                    with open(keyword_file, 'r', encoding='utf-8') as f:
                        keywords[lang] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load keywords for {lang}: {e}")
                    keywords[lang] = []
            else:
                print(f"Warning: Keyword file not found for {lang} in either '{primary_dir}' or '{fallback_dir}'")
                keywords[lang] = []
        
        return keywords
    
    async def analyze_audio(self, audio_path: str, language: str = 'en') -> Dict[str, Any]:
        """
        Analyze audio file for scam detection.
        
        Args:
            audio_path: Path to audio file
            language: Language code (en, es, fr, hi, ne)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            print(f"🔍 Analyzing audio: {audio_path}")
            
            # 1. Speech recognition
            print("📝 Performing speech recognition...")
            transcription_result = await self.speech_processor.transcribe_audio(audio_path, language)
            
            # 2. Keyword detection
            print("🔑 Detecting scam keywords...")
            keyword_result = self._detect_keywords(transcription_result.get('text', ''), language)
            
            # 3. Synthetic voice detection
            print("🎭 Detecting synthetic voice...")
            synthetic_result = self.synthetic_detector.detect_synthetic_voice(audio_path)
            
            # 4. Combine results
            combined_result = self._combine_results(
                transcription_result, 
                keyword_result, 
                synthetic_result
            )
            
            print("✅ Analysis complete")
            return combined_result
            
        except Exception as e:
            print(f"❌ Error in audio analysis: {e}")
            return {
                'error': str(e),
                'is_scam': False,
                'confidence': 0.0,
                'status': 'error'
            }
    
    def _detect_keywords(self, text: str, language: str) -> Dict[str, Any]:
        """Detect scam keywords in transcribed text."""
        if not text or language not in self.keywords:
            return {
                'keywords_found': [],
                'keyword_score': 0.0,
                'is_suspicious': False
            }
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.keywords[language]:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # Calculate keyword score
        total_keywords = len(self.keywords[language])
        keyword_score = len(found_keywords) / total_keywords if total_keywords > 0 else 0.0
        
        is_suspicious = keyword_score > self.config['keyword_threshold']
        
        return {
            'keywords_found': found_keywords,
            'keyword_score': keyword_score,
            'is_suspicious': is_suspicious,
            'total_keywords_checked': total_keywords
        }
    
    def _combine_results(self, transcription: Dict, keywords: Dict, synthetic: Dict) -> Dict[str, Any]:
        """Combine all detection results into a final assessment."""
        # Calculate overall scam probability
        scam_indicators = 0
        total_indicators = 0
        rationale_parts = []
        
        # 1. Keyword detection
        if keywords.get('is_suspicious', False):
            scam_indicators += 1
            rationale_parts.append(f"Suspicious keywords detected: {', '.join(keywords.get('keywords_found', []))}")
        total_indicators += 1
        
        # 2. Synthetic voice detection
        if synthetic.get('is_synthetic', False):
            scam_indicators += 1
            rationale_parts.append(f"Synthetic voice detected: {synthetic.get('rationale', '')}")
        total_indicators += 1
        
        # 3. Transcription quality (if transcription failed, it might be suspicious)
        if not transcription.get('text') or transcription.get('confidence', 0) < 0.5:
            scam_indicators += 0.5
            rationale_parts.append("Poor transcription quality")
        total_indicators += 1
        
        # Calculate overall confidence
        overall_confidence = scam_indicators / total_indicators if total_indicators > 0 else 0.0
        
        # Determine if it's a scam
        is_scam = overall_confidence > self.config['confidence_threshold']
        
        # Generate final rationale
        if rationale_parts:
            final_rationale = f"Scam indicators: {', '.join(rationale_parts)}"
        else:
            final_rationale = "No scam indicators detected"
        
        return {
            'is_scam': is_scam,
            'confidence': overall_confidence,
            'rationale': final_rationale,
            'status': 'success',
            'details': {
                'transcription': transcription,
                'keywords': keywords,
                'synthetic_voice': synthetic,
                'scam_indicators': {
                    'count': scam_indicators,
                    'total': total_indicators
                }
            },
            'recommendation': self._generate_recommendation(overall_confidence, is_scam)
        }
    
    def _generate_recommendation(self, confidence: float, is_scam: bool) -> str:
        """Generate recommendation based on detection results."""
        if is_scam:
            if confidence > 0.8:
                return "HIGH RISK: Strong evidence of scam detected. Do not proceed."
            elif confidence > 0.6:
                return "MEDIUM RISK: Suspicious activity detected. Exercise extreme caution."
            else:
                return "LOW RISK: Some suspicious indicators. Proceed with caution."
        else:
            if confidence < 0.2:
                return "LOW RISK: No suspicious indicators detected."
            else:
                return "MINIMAL RISK: Few suspicious indicators. Proceed normally."
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the detection system."""
        return {
            'status': 'operational',
            'components': {
                'speech_processor': 'initialized',
                'synthetic_detector': 'initialized',
                'keyword_detector': 'initialized'
            },
            'languages_supported': list(self.keywords.keys()),
            'config': self.config
        }


async def main():
    """Test the integrated detection system."""
    # Initialize the system
    detector = IntegratedDetectionSystem()
    
    # Test with a sample audio file if available
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"Testing with: {test_file}")
        result = await detector.analyze_audio(test_file, 'en')
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("No test file found. Create a test_audio.wav file to test the system.")
        print("System status:", detector.get_system_status())


if __name__ == "__main__":
    asyncio.run(main())
