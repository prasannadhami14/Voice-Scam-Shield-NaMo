#!/usr/bin/env python3
"""
Simplified synthetic voice detector for MVP.
This version uses basic audio features and doesn't require TensorFlow.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleSyntheticDetector:
    """
    Simplified synthetic voice detector using basic audio features.
    This is a rule-based approach for the MVP.
    """
    
    def __init__(self):
        """Initialize the simple synthetic voice detector."""
        self.is_initialized = True
        print("✅ Simple synthetic voice detector initialized")
    
    def extract_basic_features(self, audio_path: str) -> Dict[str, float]:
        """Extract basic audio features for synthetic voice detection."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            features = {}
            
            # 1. Spectral features
            try:
                # Spectral centroid (brightness)
                spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
                features['spectral_centroid_std'] = float(np.std(spec_centroid))
                
                # Spectral rolloff
                spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                features['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
                
                # Spectral bandwidth
                spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                features['spectral_bandwidth_mean'] = float(np.mean(spec_bandwidth))
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)
                features['zero_crossing_rate_mean'] = float(np.mean(zcr))
                
                # RMS energy
                rms = librosa.feature.rms(y=y)
                features['rms_mean'] = float(np.mean(rms))
                features['rms_std'] = float(np.std(rms))
                
            except Exception as e:
                print(f"Error extracting spectral features: {e}")
                features.update({
                    'spectral_centroid_mean': 0.0,
                    'spectral_centroid_std': 0.0,
                    'spectral_rolloff_mean': 0.0,
                    'spectral_bandwidth_mean': 0.0,
                    'zero_crossing_rate_mean': 0.0,
                    'rms_mean': 0.0,
                    'rms_std': 0.0
                })
            
            # 2. Pitch features
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                if np.any(magnitudes > 0.1):
                    pitch_values = pitches[magnitudes > 0.1]
                    features['pitch_mean'] = float(np.mean(pitch_values))
                    features['pitch_std'] = float(np.std(pitch_values))
                    features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                else:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
                    features['pitch_range'] = 0.0
            except Exception as e:
                print(f"Error extracting pitch features: {e}")
                features.update({
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_range': 0.0
                })
            
            # 3. MFCC features
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
                features['mfcc_1_mean'] = float(np.mean(mfccs[0]))
                features['mfcc_2_mean'] = float(np.mean(mfccs[1]))
                features['mfcc_3_mean'] = float(np.mean(mfccs[2]))
                features['mfcc_4_mean'] = float(np.mean(mfccs[3]))
                features['mfcc_5_mean'] = float(np.mean(mfccs[4]))
            except Exception as e:
                print(f"Error extracting MFCC features: {e}")
                features.update({
                    'mfcc_1_mean': 0.0,
                    'mfcc_2_mean': 0.0,
                    'mfcc_3_mean': 0.0,
                    'mfcc_4_mean': 0.0,
                    'mfcc_5_mean': 0.0
                })
            
            # 4. Basic statistical features
            try:
                features['amplitude_mean'] = float(np.mean(np.abs(y)))
                features['amplitude_std'] = float(np.std(np.abs(y)))
                features['amplitude_max'] = float(np.max(np.abs(y)))
                features['amplitude_min'] = float(np.min(np.abs(y)))
            except Exception as e:
                print(f"Error extracting amplitude features: {e}")
                features.update({
                    'amplitude_mean': 0.0,
                    'amplitude_std': 0.0,
                    'amplitude_max': 0.0,
                    'amplitude_min': 0.0
                })
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return {}
    
    def detect_synthetic_voice(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect synthetic voice using rule-based approach.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detection results
        """
        try:
            features = self.extract_basic_features(audio_path)
            
            if not features:
                return {
                    'is_synthetic': False,
                    'confidence': 0.0,
                    'rationale': 'Feature extraction failed',
                    'features': {}
                }
            
            # Rule-based synthetic voice detection
            synthetic_indicators = 0
            total_indicators = 0
            rationale_parts = []
            
            # 1. Check for unnaturally consistent pitch
            if features.get('pitch_std', 0) < 50 and features.get('pitch_mean', 0) > 0:
                synthetic_indicators += 1
                rationale_parts.append("Unnaturally consistent pitch")
            total_indicators += 1
            
            # 2. Check for unnaturally smooth spectral features
            if features.get('spectral_centroid_std', 0) < 100:
                synthetic_indicators += 1
                rationale_parts.append("Unnaturally smooth spectral features")
            total_indicators += 1
            
            # 3. Check for artificial amplitude consistency
            if features.get('amplitude_std', 0) < 0.01:
                synthetic_indicators += 1
                rationale_parts.append("Unnaturally consistent amplitude")
            total_indicators += 1
            
            # 4. Check for artificial MFCC patterns
            mfcc_variance = np.var([features.get(f'mfcc_{i}_mean', 0) for i in range(1, 6)])
            if mfcc_variance < 10:
                synthetic_indicators += 1
                rationale_parts.append("Artificial MFCC patterns")
            total_indicators += 1
            
            # 5. Check for unnaturally low zero crossing rate
            if features.get('zero_crossing_rate_mean', 0) < 0.01:
                synthetic_indicators += 1
                rationale_parts.append("Unnaturally low zero crossing rate")
            total_indicators += 1
            
            # Calculate confidence based on indicators
            confidence = synthetic_indicators / total_indicators if total_indicators > 0 else 0.0
            
            # Determine if synthetic
            is_synthetic = confidence > 0.4  # Threshold for synthetic detection
            
            # Generate rationale
            if rationale_parts:
                rationale = f"Synthetic indicators: {', '.join(rationale_parts)}"
            else:
                rationale = "No synthetic indicators detected"
            
            return {
                'is_synthetic': is_synthetic,
                'confidence': confidence,
                'rationale': rationale,
                'features': features,
                'indicators': {
                    'synthetic_count': synthetic_indicators,
                    'total_count': total_indicators
                }
            }
            
        except Exception as e:
            print(f"Error in synthetic voice detection: {e}")
            return {
                'is_synthetic': False,
                'confidence': 0.0,
                'rationale': f'Detection error: {str(e)}',
                'features': {},
                'error': str(e)
            }
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict whether an audio file contains synthetic voice.
        This is a compatibility method for the existing system.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        return self.detect_synthetic_voice(audio_path)


def main():
    """Test the simple synthetic voice detector."""
    detector = SimpleSyntheticDetector()
    
    # Test with a sample audio file if available
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"Testing with: {test_file}")
        result = detector.detect_synthetic_voice(test_file)
        print(f"Result: {result}")
    else:
        print("No test file found. Create a test_audio.wav file to test the detector.")


if __name__ == "__main__":
    main()
