import soundfile as sf
import numpy as np
from glob import glob
import os
import argparse
import json
import sys

def debug_audio_files(language="hi", dataset_path=None):
    """
    Debug and analyze audio files for the specified language.
    
    Args:
        language: Language code (en, es, fr, hi, ne, sa)
        dataset_path: Custom path to audio dataset
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Default dataset paths for each language
    default_paths = {
        "en": "datasets/en/test_audio_english",
        "es": "datasets/es/test_audio_spanish", 
        "fr": "datasets/fr/test_audio_french",
        "hi": "datasets/hi/test_audio_converted",
        "ne": "datasets/ne/test_audio_nepali",
        "sa": "datasets/hi/test_audio_converted"  # Sanskrit uses Hindi dataset
    }
    
    if dataset_path:
        audio_dir = dataset_path
    else:
        audio_dir = os.path.join(base_dir, default_paths.get(language, default_paths["en"]))
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
    
    # Find all WAV files
    audio_files = glob(os.path.join(audio_dir, "*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    print(f"Language: {language}")
    print("-" * 60)
    
    if not audio_files:
        print("No WAV files found!")
        return
    
    # Analyze each audio file
    total_duration = 0
    total_samples = 0
    sample_rates = set()
    
    for wav_file in audio_files:
        try:
            audio, sr = sf.read(wav_file)
            length_sec = len(audio) / sr
            
            # Calculate audio statistics
            max_amp = np.max(audio)
            min_amp = np.min(audio)
            mean_amp = np.mean(np.abs(audio))
            rms_amp = np.sqrt(np.mean(audio**2))
            
            # Check for potential issues
            issues = []
            if sr != 16000:
                issues.append(f"Sample rate {sr}Hz (expected 16kHz)")
            if length_sec < 0.5:
                issues.append("Very short audio (<0.5s)")
            if length_sec > 30:
                issues.append("Very long audio (>30s)")
            if max_amp > 0.95:
                issues.append("Audio may be clipping")
            if rms_amp < 0.01:
                issues.append("Very quiet audio")
            
            # Print file analysis
            filename = os.path.basename(wav_file)
            print(f"File: {filename}")
            print(f"  Duration: {length_sec:.2f}s")
            print(f"  Sample rate: {sr} Hz")
            print(f"  Channels: {audio.ndim}")
            print(f"  Max amplitude: {max_amp:.4f}")
            print(f"  Min amplitude: {min_amp:.4f}")
            print(f"  Mean amplitude: {mean_amp:.4f}")
            print(f"  RMS amplitude: {rms_amp:.4f}")
            
            if issues:
                print(f"  Issues: {', '.join(issues)}")
            
            print()
            
            # Accumulate statistics
            total_duration += length_sec
            total_samples += len(audio)
            sample_rates.add(sr)
            
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            print()
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total files processed: {len(audio_files)}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average duration per file: {total_duration/len(audio_files):.2f} seconds")
    print(f"Total samples: {total_samples:,}")
    print(f"Sample rates found: {sorted(sample_rates)}")
    
    # Check for consistency issues
    if len(sample_rates) > 1:
        print("⚠️  Warning: Multiple sample rates detected!")
    
    if total_duration == 0:
        print("⚠️  Warning: No valid audio files processed!")

def main():
    parser = argparse.ArgumentParser(description="Debug and analyze audio files")
    parser.add_argument(
        "--language", "-l",
        choices=["en", "es", "fr", "hi", "ne", "sa"],
        default="hi",
        help="Language code (default: hi)"
    )
    parser.add_argument(
        "--dataset", "-d",
        help="Custom path to audio dataset directory"
    )
    
    args = parser.parse_args()
    
    debug_audio_files(language=args.language, dataset_path=args.dataset)

if __name__ == "__main__":
    # If no arguments provided, default to Hindi (backward compatibility)
    if len(sys.argv) == 1:
        debug_audio_files("hi")
    else:
        main()