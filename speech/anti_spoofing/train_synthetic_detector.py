#!/usr/bin/env python3
"""
Training script for the synthetic voice detection model.
This script helps you train a basic model from scratch for your MVP.
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from synthetic_voice_detector import SyntheticVoiceDetector, create_synthetic_voice_dataset

def create_training_data():
    """Create training data structure and synthetic samples."""
    print("Setting up training data...")
    
    # Create directories
    os.makedirs('training_data/real', exist_ok=True)
    os.makedirs('training_data/synthetic', exist_ok=True)
    
    # Generate synthetic samples
    print("Generating synthetic voice samples...")
    create_synthetic_voice_dataset()
    
    print("\nTraining data setup complete!")
    print("Next steps:")
    print("1. Add real voice samples to 'training_data/real/' directory")
    print("2. Run: python3 train_synthetic_detector.py --train")
    print("\nReal voice samples can be:")
    print("- Recordings of human speech (WAV, MP3, FLAC)")
    print("- Downloaded from free speech datasets")
    print("- Your own voice recordings")

def train_model(real_dir: str = 'training_data/real', 
                synthetic_dir: str = 'training_data/synthetic',
                test_size: float = 0.2,
                save_path: str = 'synthetic_voice_detector.pkl'):
    """Train the synthetic voice detection model."""
    
    # Check if training data exists
    if not os.path.exists(real_dir):
        print(f"Error: Real voice directory '{real_dir}' not found.")
        print("Please add real voice samples first.")
        return False
    
    if not os.path.exists(synthetic_dir):
        print(f"Error: Synthetic voice directory '{synthetic_dir}' not found.")
        print("Please run with --setup first.")
        return False
    
    # Count audio files
    real_files = [f for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    synthetic_files = [f for f in os.listdir(synthetic_dir) 
                      if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    if not real_files:
        print("Error: No real voice samples found.")
        print(f"Please add audio files to '{real_dir}'")
        return False
    
    if not synthetic_files:
        print("Error: No synthetic voice samples found.")
        print(f"Please run with --setup first.")
        return False
    
    print(f"Found {len(real_files)} real voice samples")
    print(f"Found {len(synthetic_files)} synthetic voice samples")
    
    # Initialize detector
    print("\nInitializing synthetic voice detector...")
    detector = SyntheticVoiceDetector()
    
    # Train model
    print("\nStarting training...")
    try:
        metrics = detector.train(real_dir, synthetic_dir, test_size=test_size)
        
        print(f"\nTraining completed successfully!")
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Real samples: {metrics['n_real_samples']}")
        print(f"Synthetic samples: {metrics['n_synthetic_samples']}")
        print(f"Features used: {metrics['n_features']}")
        
        # Save model
        detector.save_model(save_path)
        print(f"\nModel saved to: {save_path}")
        
        # Test the model
        print("\nTesting model on synthetic sample...")
        test_file = os.path.join(synthetic_dir, synthetic_files[0])
        result = detector.predict(test_file)
        print(f"Test prediction: {result}")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def test_model(model_path: str, test_audio: str):
    """Test a trained model on an audio file."""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return False
    
    if not os.path.exists(test_audio):
        print(f"Error: Test audio file '{test_audio}' not found.")
        return False
    
    try:
        # Load model
        detector = SyntheticVoiceDetector(model_path)
        
        # Make prediction
        result = detector.predict(test_audio)
        
        print(f"\nPrediction results for {test_audio}:")
        print(f"Is synthetic: {result['is_synthetic']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Real probability: {result['real_probability']:.4f}")
        print(f"Synthetic probability: {result['synthetic_probability']:.4f}")
        
        # Show top features
        print("\nTop contributing features:")
        for feature, importance in result['top_features'].items():
            print(f"  {feature}: {importance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Testing failed: {e}")
        return False

def generate_additional_synthetic_samples(count: int = 10):
    """Generate additional synthetic samples for training."""
    print(f"Generating {count} additional synthetic samples...")
    
    os.makedirs('training_data/synthetic', exist_ok=True)
    
    sr = 16000
    duration = 3  # 3 seconds
    
    for i in range(count):
        # Create more varied synthetic-like audio
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Random fundamental frequency
        base_freq = np.random.uniform(200, 800)
        
        # Create synthetic-like signal with artificial characteristics
        signal = (
            0.4 * np.sin(2 * np.pi * base_freq * t) +
            0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.2 * np.sin(2 * np.pi * base_freq * 3 * t) +
            0.1 * np.random.normal(0, 1, len(t))
        )
        
        # Add artificial formants
        formant_freqs = np.random.uniform(300, 3000, 3)
        for freq in formant_freqs:
            signal += 0.15 * np.sin(2 * np.pi * freq * t)
        
        # Add artificial noise bursts
        noise_positions = np.random.choice(len(t), size=5, replace=False)
        for pos in noise_positions:
            start = max(0, pos - 100)
            end = min(len(t), pos + 100)
            signal[start:end] += 0.3 * np.random.normal(0, 1, end - start)
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Save
        filename = f'training_data/synthetic/additional_synthetic_{i:02d}.wav'
        sf.write(filename, signal, sr)
    
    print(f"Generated {count} additional synthetic samples")

def main():
    parser = argparse.ArgumentParser(description='Train synthetic voice detection model')
    parser.add_argument('--setup', action='store_true', 
                       help='Setup training data structure and generate synthetic samples')
    parser.add_argument('--train', action='store_true',
                       help='Train the model with existing training data')
    parser.add_argument('--test', type=str, metavar='AUDIO_FILE',
                       help='Test trained model on an audio file')
    parser.add_argument('--model', type=str, default='synthetic_voice_detector.pkl',
                       help='Path to save/load model (default: synthetic_voice_detector.pkl)')
    parser.add_argument('--real-dir', type=str, default='training_data/real',
                       help='Directory containing real voice samples')
    parser.add_argument('--synthetic-dir', type=str, default='training_data/synthetic',
                       help='Directory containing synthetic voice samples')
    parser.add_argument('--generate-more', type=int, metavar='COUNT',
                       help='Generate additional synthetic samples')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    if not any([args.setup, args.train, args.test, args.generate_more]):
        parser.print_help()
        return
    
    if args.setup:
        create_training_data()
    
    if args.generate_more:
        generate_additional_synthetic_samples(args.generate_more)
    
    if args.train:
        success = train_model(
            real_dir=args.real_dir,
            synthetic_dir=args.synthetic_dir,
            test_size=args.test_size,
            save_path=args.model
        )
        if not success:
            sys.exit(1)
    
    if args.test:
        success = test_model(args.model, args.test)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
