import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SyntheticVoiceDetector:
    """
    Basic synthetic voice detection model using audio features.
    This is a simplified MVP version that doesn't require pre-trained models.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the synthetic voice detector.
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """
        Extract features from audio file for synthetic voice detection.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Feature vector of fixed length (100 features)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr)
            
            features = []
            
            # 1. MFCC features (26 features)
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfccs, axis=1)
                mfcc_std = np.std(mfccs, axis=1)
                features.extend(mfcc_mean.tolist())
                features.extend(mfcc_std.tolist())
            except:
                features.extend([0.0] * 26)
            
            # 2. Spectral features (20 features)
            try:
                # Spectral centroid
                spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                features.append(float(np.mean(spec_centroid)))
                features.append(float(np.std(spec_centroid)))
                
                # Spectral rolloff
                spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                features.append(float(np.mean(spec_rolloff)))
                features.append(float(np.std(spec_rolloff)))
                
                # Spectral bandwidth
                spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                features.append(float(np.mean(spec_bandwidth)))
                features.append(float(np.std(spec_bandwidth)))
                
                # Spectral contrast
                spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features.append(float(np.mean(spec_contrast)))
                features.append(float(np.std(spec_contrast)))
                
                # Spectral flatness
                spec_flatness = librosa.feature.spectral_flatness(y=y)
                features.append(float(np.mean(spec_flatness)))
                features.append(float(np.std(spec_flatness)))
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)
                features.append(float(np.mean(zcr)))
                features.append(float(np.std(zcr)))
                
                # RMS energy
                rms = librosa.feature.rms(y=y)
                features.append(float(np.mean(rms)))
                features.append(float(np.std(rms)))
                
                # Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                features.append(float(np.mean(mel_spec)))
                features.append(float(np.std(mel_spec)))
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features.append(float(np.mean(chroma)))
                features.append(float(np.std(chroma)))
            except:
                features.extend([0.0] * 20)
            
            # 3. Pitch features (4 features)
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                if np.any(magnitudes > 0.1):
                    pitch_mean = float(np.mean(pitches[magnitudes > 0.1]))
                    pitch_std = float(np.std(pitches[magnitudes > 0.1]))
                else:
                    pitch_mean, pitch_std = 0.0, 0.0
                features.extend([pitch_mean, pitch_std])
                
                # Pitch range
                pitch_range = float(np.max(pitches) - np.min(pitches)) if len(pitches) > 0 else 0.0
                features.append(pitch_range)
                
                # Pitch stability (inverse of variance)
                pitch_stability = 1.0 / (pitch_std + 1e-10) if pitch_std > 0 else 0.0
                features.append(float(pitch_stability))
            except:
                features.extend([0.0] * 4)
            
            # 4. Tempo and rhythm features (6 features)
            try:
                # Tempo
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo) if not np.isnan(tempo) else 0.0)
                
                # Beat strength
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                features.append(float(np.mean(onset_env)))
                
                # Harmonic-percussive separation
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                harmonic_ratio = float(np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_harmonic)) + np.sum(np.abs(y_percussive)) + 1e-10))
                features.append(harmonic_ratio)
                
                # Spectral slope
                freqs = np.fft.fftfreq(len(y))
                spectrum = np.abs(np.fft.fft(y))
                if len(freqs[freqs > 0]) > 0:
                    slope = float(np.polyfit(freqs[freqs > 0], np.log(spectrum[freqs > 0] + 1e-10), 1)[0])
                else:
                    slope = 0.0
                features.append(slope if not np.isnan(slope) else 0.0)
                
                # Spectral kurtosis and skewness
                spec_kurtosis = float(np.mean((spectrum - np.mean(spectrum))**4) / (np.std(spectrum)**4 + 1e-10))
                spec_skewness = float(np.mean((spectrum - np.mean(spectrum))**3) / (np.std(spectrum)**3 + 1e-10))
                features.extend([spec_kurtosis if not np.isnan(spec_kurtosis) else 0.0, 
                               spec_skewness if not np.isnan(spec_skewness) else 0.0])
            except:
                features.extend([0.0] * 6)
            
            # 5. Temporal features (8 features)
            try:
                # Frame-based energy analysis
                frame_length = int(0.025 * sr)  # 25ms frames
                hop_length = int(0.010 * sr)    # 10ms hop
                
                frame_energy = []
                for i in range(0, len(y) - frame_length, hop_length):
                    frame = y[i:i + frame_length]
                    frame_energy.append(float(np.sum(frame**2)))
                
                if frame_energy:
                    # Temporal consistency
                    temporal_consistency = float(np.std(frame_energy) / (np.mean(frame_energy) + 1e-10))
                    features.append(temporal_consistency if not np.isnan(temporal_consistency) else 0.0)
                    
                    # Energy dynamics
                    energy_range = float(np.max(frame_energy) - np.min(frame_energy))
                    energy_stability = float(1.0 / (np.std(frame_energy) + 1e-10))
                    features.extend([energy_range, energy_stability])
                else:
                    features.extend([0.0, 0.0, 0.0])
                
                # Formant-like features
                freqs = np.fft.fftfreq(len(y))
                spectrum = np.abs(np.fft.fft(y))
                formant_peaks = []
                for i in range(1, len(spectrum) - 1):
                    if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                        if spectrum[i] > np.mean(spectrum) + 2 * np.std(spectrum):
                            formant_peaks.append(i)
                
                formant_count = float(len(formant_peaks))
                features.append(formant_count)
                
                if len(formant_peaks) > 1:
                    formant_spacing = float(np.mean(np.diff(formant_peaks)))
                    features.append(formant_spacing if not np.isnan(formant_spacing) else 0.0)
                else:
                    features.append(0.0)
                
                # Additional temporal features
                features.extend([0.0, 0.0])  # Placeholder for future features
            except:
                features.extend([0.0] * 8)
            
            # 6. Additional features to reach 100 (36 features)
            try:
                # Simple statistical features
                features.append(float(np.mean(y)))
                features.append(float(np.std(y)))
                features.append(float(np.max(y)))
                features.append(float(np.min(y)))
                features.append(float(np.percentile(y, 25)))
                features.append(float(np.percentile(y, 75)))
                
                # Frequency domain features
                freqs = np.fft.fftfreq(len(y))
                spectrum = np.abs(np.fft.fft(y))
                features.append(float(np.mean(spectrum)))
                features.append(float(np.std(spectrum)))
                features.append(float(np.max(spectrum)))
                
                # Fill remaining features with zeros
                remaining_features = 100 - len(features)
                features.extend([0.0] * remaining_features)
            except:
                remaining_features = 100 - len(features)
                features.extend([0.0] * remaining_features)
            
            # Ensure exactly 100 features
            if len(features) != 100:
                if len(features) < 100:
                    features.extend([0.0] * (100 - len(features)))
                else:
                    features = features[:100]
            
            # Convert to numpy array and ensure all values are float64
            features = np.array(features, dtype=np.float64)
            
            # Handle any remaining NaN or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zero features if extraction fails
            return np.zeros(100, dtype=np.float64)
    
    def train(self, real_audio_dir: str, synthetic_audio_dir: str, 
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the synthetic voice detection model.
        
        Args:
            real_audio_dir: Directory containing real voice samples
            synthetic_audio_dir: Directory containing synthetic voice samples
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        print("Extracting features from real voice samples...")
        real_features = []
        for file in os.listdir(real_audio_dir):
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(real_audio_dir, file)
                features = self.extract_features(file_path)
                real_features.append(features)
        
        print("Extracting features from synthetic voice samples...")
        synthetic_features = []
        for file in os.listdir(synthetic_audio_dir):
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(synthetic_audio_dir, file)
                features = self.extract_features(file_path)
                synthetic_features.append(features)
        
        if not real_features or not synthetic_features:
            raise ValueError("No audio files found in one or both directories")
        
        # Prepare data
        X = np.vstack([real_features, synthetic_features])
        y = np.array([0] * len(real_features) + [1] * len(synthetic_features))  # 0=real, 1=synthetic
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed!")
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Synthetic']))
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'n_real_samples': len(real_features),
            'n_synthetic_samples': len(synthetic_features),
            'n_features': X.shape[1]
        }
    
    def predict(self, audio_path: str) -> Dict[str, any]:
        """
        Predict whether an audio file contains synthetic voice.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features(audio_path)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Get feature importance for the top features
        feature_importance = self.model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 features
        
        result = {
            'is_synthetic': bool(prediction),
            'confidence': float(probability[prediction]),
            'real_probability': float(probability[0]),
            'synthetic_probability': float(probability[1]),
            'top_features': {
                f'feature_{i}': float(feature_importance[i]) 
                for i in top_features_idx
            }
        }
        
        return result
    
    def save_model(self, model_path: str):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model and scaler."""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data.get('feature_names', [])
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        return {f'feature_{i}': float(importance[i]) for i in range(len(importance))}


def create_synthetic_voice_dataset():
    """
    Create a basic synthetic voice dataset for training.
    This generates synthetic-like audio using basic audio processing techniques.
    """
    import soundfile as sf
    
    # Create output directories
    os.makedirs('training_data/real', exist_ok=True)
    os.makedirs('training_data/synthetic', exist_ok=True)
    
    print("Creating synthetic voice dataset...")
    
    # Generate synthetic-like audio (simplified approach)
    sr = 16000
    duration = 3  # 3 seconds
    
    for i in range(20):  # Generate 20 synthetic samples
        # Create synthetic-like audio with artificial characteristics
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Add artificial harmonics and noise
        signal = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # Fundamental frequency
            0.3 * np.sin(2 * np.pi * 880 * t) +  # First harmonic
            0.2 * np.sin(2 * np.pi * 1320 * t) + # Second harmonic
            0.1 * np.random.normal(0, 1, len(t))  # Artificial noise
        )
        
        # Add artificial formant-like structure
        formant_freqs = [500, 1500, 2500]
        for freq in formant_freqs:
            signal += 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Save synthetic sample
        filename = f'training_data/synthetic/synthetic_{i:02d}.wav'
        sf.write(filename, signal, sr)
    
    print("Synthetic dataset created in training_data/synthetic/")
    print("Please add real voice samples to training_data/real/ for training")


if __name__ == "__main__":
    # Example usage
    detector = SyntheticVoiceDetector()
    
    # Check if training data exists
    if not os.path.exists('training_data/real') or not os.path.exists('training_data/synthetic'):
        print("Training data not found. Creating synthetic dataset...")
        create_synthetic_voice_dataset()
        print("\nPlease add real voice samples to training_data/real/ and run again.")
    else:
        # Train the model
        print("Training synthetic voice detection model...")
        metrics = detector.train('training_data/real', 'training_data/synthetic')
        
        # Save the model
        detector.save_model('synthetic_voice_detector.pkl')
        
        # Test on a sample
        if os.path.exists('training_data/synthetic/synthetic_00.wav'):
            result = detector.predict('training_data/synthetic/synthetic_00.wav')
            print(f"\nTest prediction: {result}")
