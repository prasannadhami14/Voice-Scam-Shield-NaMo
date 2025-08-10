import os
import time
import json
import requests
from typing import Dict, List, Optional
from .synthetic_voice_detector import SyntheticVoiceDetector
import threading
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np

class AlertSystem:
    """
    Alert system for Voice Scam Shield that integrates synthetic voice detection
    and ElevenLabs TTS for audio alerts.
    """
    
    def __init__(self, elevenlabs_api_key: str = None, model_path: str = None):
        """
        Initialize the alert system.
        
        Args:
            elevenlabs_api_key: API key for ElevenLabs TTS
            model_path: Path to trained synthetic voice detector model
        """
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv('ELEVENLABS_API_KEY')
        self.synthetic_detector = None
        self.alert_queue = queue.Queue()
        self.is_running = False
        
        # Load synthetic voice detector if available
        if model_path and os.path.exists(model_path):
            try:
                self.synthetic_detector = SyntheticVoiceDetector(model_path)
                print("Synthetic voice detector loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load synthetic voice detector: {e}")
        
        # Alert messages for different risk levels
        self.alert_messages = {
            'scam': {
                'en': "Warning! This call appears to be a scam. Please hang up immediately.",
                'es': "¡Advertencia! Esta llamada parece ser una estafa. Cuelgue inmediatamente.",
                'fr': "Attention ! Cet appel semble être une arnaque. Raccrochez immédiatement.",
                'ne': "सावधान! यो कल एउटा ठगी जस्तो देखिन्छ। तुरुन्तै कल बन्द गर्नुहोस्।",
                'hi': "सावधान! यह कॉल एक घोटाला लग रहा है। तुरंत कॉल बंद करें।",
                'sa': "सावधान्यम्! एषः कालः कपटः प्रतीयते। तत्कालं कालं विरमयतु।"
            },
            'suspicious': {
                'en': "Caution: This call seems suspicious. Stay alert.",
                'es': "Precaución: Esta llamada parece sospechosa. Manténgase alerta.",
                'fr': "Prudence : Cet appel semble suspect. Restez vigilant.",
                'ne': "सावधान: यो कल संदिग्ध जस्तो देखिन्छ। सतर्क रहनुहोस्।",
                'hi': "सावधान: यह कॉल संदिग्ध लग रहा है। सतर्क रहें।",
                'sa': "सावधान्यम्: एषः कालः संदिग्धः प्रतीयते। सावधानः भवतु।"
            },
            'synthetic': {
                'en': "Alert: Synthetic voice detected. This may be an AI-generated scam.",
                'es': "Alerta: Voz sintética detectada. Esto puede ser una estafa generada por IA.",
                'fr': "Alerte : Voix synthétique détectée. Cela peut être une arnaque générée par IA.",
                'ne': "सावधान: कृत्रिम आवाज पत्ता लग्यो। यो AI द्वारा सिर्जना गरिएको ठगी हुन सक्छ।",
                'hi': "सावधान: कृत्रिम आवाज पाई गई। यह AI द्वारा जनरेट किया गया घोटाला हो सकता है।",
                'sa': "सावधान्यम्: कृत्रिमस्वरः आसादितः। एतत् AI-जनितं कपटं भवितुम् अर्हति।"
            }
        }
        
        # ElevenLabs voice IDs for different languages
        self.voice_ids = {
            'en': '21m00Tcm4TlvDq8ikWAM',  # Rachel (English)
            'es': 'ErXwobaYiN019PkySvjV',  # Antoni (Spanish)
            'fr': 'VR6AewLTigWG4xSOukaG',  # Josh (French)
            'ne': '21m00Tcm4TlvDq8ikWAM',  # Use English voice for Nepali
            'hi': '21m00Tcm4TlvDq8ikWAM',  # Use English voice for Hindi
            'sa': '21m00Tcm4TlvDq8ikWAM'   # Use English voice for Sanskrit
        }
    
    def generate_tts_alert(self, message: str, language: str = 'en', 
                          output_path: str = None) -> Optional[str]:
        """
        Generate TTS alert using ElevenLabs API.
        
        Args:
            message: Alert message to convert to speech
            language: Language code for voice selection
            output_path: Path to save audio file (optional)
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.elevenlabs_api_key:
            print("Warning: ElevenLabs API key not provided. Cannot generate TTS alerts.")
            return None
        
        try:
            # Select appropriate voice
            voice_id = self.voice_ids.get(language, self.voice_ids['en'])
            
            # Prepare API request
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            data = {
                "text": message,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            # Make API request
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save audio file
                if not output_path:
                    timestamp = int(time.time())
                    output_path = f"alerts/alert_{timestamp}.mp3"
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                print(f"TTS alert generated: {output_path}")
                return output_path
            else:
                print(f"TTS generation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error generating TTS alert: {e}")
            return None
    
    def detect_synthetic_voice(self, audio_path: str) -> Dict[str, any]:
        """
        Detect synthetic voice in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detection results
        """
        if not self.synthetic_detector:
            return {
                'is_synthetic': False,
                'confidence': 0.0,
                'error': 'Synthetic voice detector not available'
            }
        
        try:
            result = self.synthetic_detector.predict(audio_path)
            return result
        except Exception as e:
            return {
                'is_synthetic': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def create_alert(self, risk_level: str, language: str = 'en', 
                    synthetic_detection: Dict = None) -> Dict[str, any]:
        """
        Create a comprehensive alert based on risk level and synthetic voice detection.
        
        Args:
            risk_level: Risk level ('scam', 'suspicious', 'safe')
            language: Language for alert message
            synthetic_detection: Results from synthetic voice detection
            
        Returns:
            Alert information
        """
        alert_info = {
            'timestamp': time.time(),
            'risk_level': risk_level,
            'language': language,
            'message': None,
            'audio_path': None,
            'synthetic_detection': synthetic_detection
        }
        
        # Generate alert message
        if risk_level in ['scam', 'suspicious']:
            message = self.alert_messages[risk_level].get(language, 
                                                        self.alert_messages[risk_level]['en'])
        else:
            message = "Call appears safe. No immediate threat detected."
        
        alert_info['message'] = message
        
        # Generate TTS alert
        if risk_level in ['scam', 'suspicious']:
            audio_path = self.generate_tts_alert(message, language)
            alert_info['audio_path'] = audio_path
        
        # Add synthetic voice alert if detected
        if synthetic_detection and synthetic_detection.get('is_synthetic', False):
            synthetic_message = self.alert_messages['synthetic'].get(language, 
                                                                   self.alert_messages['synthetic']['en'])
            alert_info['message'] += f" {synthetic_message}"
            
            # Generate additional TTS for synthetic voice alert
            if self.elevenlabs_api_key:
                synthetic_audio = self.generate_tts_alert(synthetic_message, language)
                if synthetic_audio:
                    alert_info['synthetic_audio_path'] = synthetic_audio
        
        return alert_info
    
    def play_alert_audio(self, audio_path: str):
        """
        Play alert audio through speakers.
        
        Args:
            audio_path: Path to audio file
        """
        try:
            # Load and play audio
            data, samplerate = sf.read(audio_path)
            
            # Normalize audio
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Play audio
            sd.play(data, samplerate)
            sd.wait()  # Wait for audio to finish
            
            print(f"Alert audio played: {audio_path}")
            
        except Exception as e:
            print(f"Error playing alert audio: {e}")
    
    def process_audio_chunk(self, audio_data: np.ndarray, risk_assessment: Dict, 
                           language: str = 'en') -> Dict[str, any]:
        """
        Process audio chunk for synthetic voice detection and generate alerts.
        
        Args:
            audio_data: Audio data as numpy array
            risk_assessment: Risk assessment from scam detection
            language: Detected language
            
        Returns:
            Processing results including alerts
        """
        # Save audio chunk temporarily for analysis
        temp_audio_path = f"temp_audio_{int(time.time())}.wav"
        sf.write(temp_audio_path, audio_data, 16000)
        
        try:
            # Detect synthetic voice
            synthetic_result = self.detect_synthetic_voice(temp_audio_path)
            
            # Determine risk level
            risk_level = 'safe'
            if risk_assessment.get('risk', 0) >= 80:
                risk_level = 'scam'
            elif risk_assessment.get('risk', 0) >= 50:
                risk_level = 'suspicious'
            
            # Create alert
            alert = self.create_alert(risk_level, language, synthetic_result)
            
            # Play audio alert if needed
            if alert.get('audio_path'):
                # Play in background thread to avoid blocking
                threading.Thread(target=self.play_alert_audio, 
                               args=(alert['audio_path'],)).start()
            
            if alert.get('synthetic_audio_path'):
                threading.Thread(target=self.play_alert_audio, 
                               args=(alert['synthetic_audio_path'],)).start()
            
            return {
                'alert': alert,
                'synthetic_detection': synthetic_result,
                'processing_time': time.time()
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    def start_alert_monitoring(self):
        """Start the alert monitoring system."""
        self.is_running = True
        print("Alert monitoring system started")
        
        # Start alert processing thread
        def process_alerts():
            while self.is_running:
                try:
                    # Process alerts from queue
                    if not self.alert_queue.empty():
                        alert_data = self.alert_queue.get(timeout=1)
                        self._process_queued_alert(alert_data)
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Error in alert processing: {e}")
                    time.sleep(1)
        
        threading.Thread(target=process_alerts, daemon=True).start()
    
    def stop_alert_monitoring(self):
        """Stop the alert monitoring system."""
        self.is_running = False
        print("Alert monitoring system stopped")
    
    def _process_queued_alert(self, alert_data: Dict):
        """Process an alert from the queue."""
        try:
            # Generate and play alert
            alert = self.create_alert(
                alert_data['risk_level'],
                alert_data.get('language', 'en'),
                alert_data.get('synthetic_detection')
            )
            
            if alert.get('audio_path'):
                self.play_alert_audio(alert['audio_path'])
            
        except Exception as e:
            print(f"Error processing queued alert: {e}")
    
    def add_alert_to_queue(self, risk_level: str, language: str = 'en', 
                          synthetic_detection: Dict = None):
        """Add an alert to the processing queue."""
        alert_data = {
            'risk_level': risk_level,
            'language': language,
            'synthetic_detection': synthetic_detection,
            'timestamp': time.time()
        }
        self.alert_queue.put(alert_data)


def test_alert_system():
    """Test the alert system functionality."""
    print("Testing Alert System...")
    
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Test TTS generation (if API key available)
    if alert_system.elevenlabs_api_key:
        print("Testing TTS alert generation...")
        audio_path = alert_system.generate_tts_alert(
            "This is a test alert for synthetic voice detection.",
            'en'
        )
        if audio_path:
            print(f"TTS test successful: {audio_path}")
            # Play the test alert
            alert_system.play_alert_audio(audio_path)
    else:
        print("No ElevenLabs API key found. Skipping TTS tests.")
    
    # Test alert creation
    print("Testing alert creation...")
    test_alert = alert_system.create_alert('scam', 'en')
    print(f"Test alert created: {test_alert}")
    
    # Test synthetic voice detection (if model available)
    if alert_system.synthetic_detector:
        print("Testing synthetic voice detection...")
        # This would require a test audio file
        print("Synthetic voice detector is available")
    else:
        print("No synthetic voice detector model found.")
    
    print("Alert system test completed!")


if __name__ == "__main__":
    test_alert_system()
