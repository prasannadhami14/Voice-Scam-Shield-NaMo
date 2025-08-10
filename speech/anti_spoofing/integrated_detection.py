#!/usr/bin/env python3
"""
Integrated detection system that combines scam detection with synthetic voice detection.
This provides a comprehensive solution for the Voice Scam Shield MVP.
"""

import os
import sys
import time
import json
import numpy as np
import soundfile as sf
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from speech_processing import RealTimeTranscriber
    from speech.config import StreamingConfig
    from synthetic_voice_detector import SyntheticVoiceDetector
    from alert_system import AlertSystem
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

class IntegratedVoiceScamDetector:
    """
    Integrated system that combines:
    1. Real-time audio transcription
    2. Scam keyword detection
    3. Synthetic voice detection
    4. Alert generation and TTS
    """
    
    def __init__(self, config: StreamingConfig = None, 
                 synthetic_model_path: str = None,
                 elevenlabs_api_key: str = None):
        """
        Initialize the integrated voice scam detector.
        
        Args:
            config: StreamingConfig object for speech processing
            synthetic_model_path: Path to trained synthetic voice detection model
            elevenlabs_api_key: API key for ElevenLabs TTS
        """
        # Use default config if none provided
        if config is None:
            config = StreamingConfig()
        
        # Initialize components
        self.transcriber = RealTimeTranscriber(config)
        self.synthetic_detector = SyntheticVoiceDetector(synthetic_model_path) if synthetic_model_path else None
        self.alert_system = AlertSystem(elevenlabs_api_key, synthetic_model_path) if elevenlabs_api_key else None
        
        # Detection history
        self.detection_history = []
        
        print("✅ Integrated Voice Scam Detector initialized successfully!")
        if self.synthetic_detector:
            print("✅ Synthetic voice detection enabled")
        if self.alert_system:
            print("✅ Alert system enabled")
    
    def process_audio_chunk(self, audio_data: np.ndarray, 
                           chunk_duration: float = 3.0) -> Dict[str, any]:
        """
        Process an audio chunk for comprehensive detection.
        
        Args:
            audio_data: Audio data as numpy array
            chunk_duration: Duration of audio chunk in seconds
            
        Returns:
            Comprehensive detection results
        """
        start_time = time.time()
        
        # Save audio chunk temporarily
        temp_audio_path = f"temp_chunk_{int(time.time())}.wav"
        sf.write(temp_audio_path, audio_data, 16000)
        
        try:
            results = {
                'timestamp': time.time(),
                'chunk_duration': chunk_duration,
                'processing_time': 0,
                'transcription': None,
                'scam_detection': None,
                'synthetic_detection': None,
                'integrated_risk': 0,
                'alerts': []
            }
            
            # 1. Audio transcription
            print("🔄 Transcribing audio...")
            try:
                transcription_result = self.transcriber.transcribe_audio_file(temp_audio_path)
                results['transcription'] = transcription_result
                results['current_language'] = transcription_result.get('language', 'en')
                self.current_language = results['current_language']
                print(f"✓ Transcription: {transcription_result.get('text', '')[:100]}...")
            except Exception as e:
                print(f"⚠ Transcription error: {e}")
                results['transcription'] = {'error': str(e)}
            
            # 2. Scam detection
            print("🔄 Detecting scams...")
            try:
                if results['transcription'] and 'text' in results['transcription']:
                    scam_result = self.transcriber.detect_scam_keywords(
                        results['transcription']['text'],
                        results['current_language']
                    )
                    results['scam_detection'] = scam_result
                    results['current_risk_score'] = scam_result.get('risk', 0)
                    self.current_risk_score = results['current_risk_score']
                    print(f"✓ Scam risk: {scam_result.get('risk', 0)}%")
                else:
                    results['scam_detection'] = {'error': 'No transcription available'}
            except Exception as e:
                print(f"⚠ Scam detection error: {e}")
                results['scam_detection'] = {'error': str(e)}
            
            # 3. Synthetic voice detection
            print("🔄 Detecting synthetic voice...")
            try:
                if self.synthetic_detector:
                    synthetic_result = self.synthetic_detector.predict(temp_audio_path)
                    results['synthetic_detection'] = synthetic_result
                    
                    if synthetic_result.get('is_synthetic', False):
                        print(f"⚠ Synthetic voice detected! Confidence: {synthetic_result.get('confidence', 0):.2f}")
                        # Boost risk score for synthetic voice
                        results['current_risk_score'] = min(100, results['current_risk_score'] + 20)
                    else:
                        print("✓ Natural voice detected")
                else:
                    results['synthetic_detection'] = {'error': 'Synthetic voice detector not available'}
            except Exception as e:
                print(f"⚠ Synthetic voice detection error: {e}")
                results['synthetic_detection'] = {'error': str(e)}
            
            # 4. Calculate integrated risk score
            results['integrated_risk'] = self._calculate_integrated_risk(results)
            
            # 5. Generate alerts
            print("🔄 Generating alerts...")
            try:
                if self.alert_system:
                    alert_result = self.alert_system.process_audio_chunk(
                        audio_data,
                        {'risk': results['integrated_risk']},
                        results['current_language']
                    )
                    results['alerts'] = alert_result
                    print("✓ Alerts generated")
                else:
                    results['alerts'] = {'error': 'Alert system not available'}
            except Exception as e:
                print(f"⚠ Alert generation error: {e}")
                results['alerts'] = {'error': str(e)}
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            
            # Store in history
            self.detection_history.append(results)
            
            # Print summary
            self._print_detection_summary(results)
            
            return results
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    def _calculate_integrated_risk(self, results: Dict) -> int:
        """Calculate integrated risk score combining all detection methods."""
        base_risk = results.get('current_risk_score', 0)
        
        # Boost risk for synthetic voice
        synthetic_boost = 0
        if results.get('synthetic_detection', {}).get('is_synthetic', False):
            confidence = results['synthetic_detection'].get('confidence', 0)
            synthetic_boost = int(confidence * 30)  # Up to 30 points boost
        
        # Boost risk for transcription errors (potential evasion)
        transcription_boost = 0
        if results.get('transcription', {}).get('error'):
            transcription_boost = 15
        
        # Calculate final risk
        final_risk = min(100, base_risk + synthetic_boost + transcription_boost)
        
        return final_risk
    
    def _print_detection_summary(self, results: Dict):
        """Print a summary of detection results."""
        print("\n" + "="*60)
        print("🔍 DETECTION SUMMARY")
        print("="*60)
        
        # Transcription
        if results.get('transcription', {}).get('text'):
            text = results['transcription']['text'][:100]
            if len(results['transcription']['text']) > 100:
                text += "..."
            print(f"📝 Transcription: {text}")
        
        # Language
        print(f"🌍 Language: {results.get('current_language', 'Unknown')}")
        
        # Risk scores
        print(f"⚠️  Scam Risk: {results.get('current_risk_score', 0)}%")
        print(f"🎯 Integrated Risk: {results.get('integrated_risk', 0)}%")
        
        # Synthetic voice
        if results.get('synthetic_detection', {}).get('is_synthetic'):
            confidence = results['synthetic_detection'].get('confidence', 0)
            print(f"🤖 Synthetic Voice: DETECTED (Confidence: {confidence:.2f})")
        else:
            print("👤 Synthetic Voice: Not detected")
        
        # Processing time
        print(f"⏱️  Processing Time: {results.get('processing_time', 0):.2f}s")
        
        # Risk level
        risk = results.get('integrated_risk', 0)
        if risk >= 80:
            print("🚨 RISK LEVEL: HIGH - IMMEDIATE ACTION REQUIRED")
        elif risk >= 50:
            print("⚠️  RISK LEVEL: MEDIUM - CAUTION ADVISED")
        else:
            print("✅ RISK LEVEL: LOW - NO IMMEDIATE THREAT")
        
        print("="*60 + "\n")
    
    def start_real_time_monitoring(self, duration: int = 60):
        """
        Start real-time monitoring for a specified duration.
        
        Args:
            duration: Monitoring duration in seconds
        """
        print(f"🚀 Starting real-time monitoring for {duration} seconds...")
        print("Speak into your microphone to test the system...")
        
        # Initialize audio stream
        import sounddevice as sd
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            
            # Process audio chunk
            audio_data = indata[:, 0]  # Take first channel
            self.process_audio_chunk(audio_data, frames / 16000)
        
        try:
            with sd.InputStream(callback=audio_callback, 
                              channels=1, 
                              samplerate=16000, 
                              blocksize=4800):  # 0.3 second chunks
                
                print("🎤 Microphone active. Press Ctrl+C to stop...")
                time.sleep(duration)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        
        print("📊 Monitoring session completed")
        self._print_session_summary()
    
    def _print_session_summary(self):
        """Print a summary of the monitoring session."""
        if not self.detection_history:
            print("No detection results to summarize")
            return
        
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        
        total_chunks = len(self.detection_history)
        avg_processing_time = np.mean([r.get('processing_time', 0) for r in self.detection_history])
        max_risk = max([r.get('integrated_risk', 0) for r in self.detection_history])
        synthetic_detections = sum([1 for r in self.detection_history 
                                  if r.get('synthetic_detection', {}).get('is_synthetic', False)])
        
        print(f"Total audio chunks processed: {total_chunks}")
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Maximum risk detected: {max_risk}%")
        print(f"Synthetic voice detections: {synthetic_detections}")
        
        # Risk distribution
        high_risk = sum([1 for r in self.detection_history if r.get('integrated_risk', 0) >= 80])
        medium_risk = sum([1 for r in self.detection_history if 50 <= r.get('integrated_risk', 0) < 80])
        low_risk = sum([1 for r in self.detection_history if r.get('integrated_risk', 0) < 50])
        
        print(f"Risk distribution: High={high_risk}, Medium={medium_risk}, Low={low_risk}")
        
        print("="*60)
    
    def save_detection_history(self, filepath: str = None):
        """Save detection history to a JSON file."""
        if not filepath:
            timestamp = int(time.time())
            filepath = f"detection_history_{timestamp}.json"
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Deep convert numpy types
            def deep_convert(obj):
                if isinstance(obj, dict):
                    return {k: deep_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            converted_history = deep_convert(self.detection_history)
            
            with open(filepath, 'w') as f:
                json.dump(converted_history, f, indent=2, default=str)
            
            print(f"✓ Detection history saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving detection history: {e}")


def main():
    """Main function for testing the integrated system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Voice Scam Detection System')
    parser.add_argument('--config', type=str, help='Path to speech processing config')
    parser.add_argument('--synthetic-model', type=str, help='Path to synthetic voice detector model')
    parser.add_argument('--elevenlabs-key', type=str, help='ElevenLabs API key')
    parser.add_argument('--monitor', type=int, metavar='SECONDS', default=30,
                       help='Start real-time monitoring for specified seconds')
    parser.add_argument('--test-audio', type=str, help='Test with an audio file')
    parser.add_argument('--save-history', action='store_true', help='Save detection history')
    
    args = parser.parse_args()
    
    # Initialize integrated detector
    print("🚀 Initializing Integrated Voice Scam Detector...")
    
    detector = IntegratedVoiceScamDetector(
        config_path=args.config,
        synthetic_model_path=args.synthetic_model,
        elevenlabs_api_key=args.elevenlabs_key
    )
    
    if args.test_audio:
        # Test with audio file
        if os.path.exists(args.test_audio):
            print(f"🧪 Testing with audio file: {args.test_audio}")
            audio_data, sr = sf.read(args.test_audio)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Take first channel
            detector.process_audio_chunk(audio_data, len(audio_data) / sr)
        else:
            print(f"❌ Audio file not found: {args.test_audio}")
    
    if args.monitor:
        # Start real-time monitoring
        detector.start_real_time_monitoring(args.monitor)
    
    if args.save_history:
        # Save detection history
        detector.save_detection_history()
    
    print("✅ Integrated detection system completed")


if __name__ == "__main__":
    main()
