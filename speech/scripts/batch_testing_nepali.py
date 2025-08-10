import json
import os
import sys
import time
import soundfile as sf
from glob import glob
import numpy as np

# Ensure project root (speech/) is importable when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import StreamingConfig
from speech_processing import RealTimeTranscriber

def run_batch_test(audio_files_pattern: str, ground_truth_file: str, output_file: str = "results/ne/test_results_nepali.json"):
    """
    Run batch testing on audio files, compare with ground truth, and calculate accuracy/latency for Nepali.

    Args:
        audio_files_pattern: Glob pattern for WAV files (e.g., "datasets/ne/test_audio_nepali/*.wav").
        ground_truth_file: Path to ground truth JSON file.
        output_file: Path to save test results JSON.
    """
    config = StreamingConfig()
    transcriber = RealTimeTranscriber(config)
    base_dir = os.path.dirname(os.path.dirname(__file__))

    gt_path = os.path.join(base_dir, ground_truth_file)
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = []
    pattern_abs = os.path.join(base_dir, audio_files_pattern)
    candidate_files = sorted(glob(pattern_abs))
    filtered_files = []
    for wav_file in candidate_files:
        rel = os.path.relpath(wav_file, base_dir)
        rel_key = rel.replace("datasets/ne/", "")
        if rel_key in ground_truth:
            filtered_files.append(wav_file)

    for wav_file in filtered_files:
        try:
            audio, sr = sf.read(wav_file, dtype="float32")
            if sr != config.sample_rate:
                raise ValueError(f"Audio {wav_file} must be {config.sample_rate}Hz")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            chunk_start = time.time()
            result = transcriber.process_audio_chunk(audio)
            result["processing_time"] = time.time() - chunk_start
            rel_cur = os.path.relpath(wav_file, base_dir).replace("datasets/ne/", "")
            result["file"] = rel_cur
            results.append(result)
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            results.append({
                "file": os.path.relpath(wav_file, base_dir),
                "transcript": "",
                "risk": 0,
                "label": "Safe",
                "rationale": f"Error processing audio: {str(e)}",
                "language": "unknown",
                "processing_time": 0.0
            })

    out_path = os.path.join(base_dir, output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    correct = 0
    total = len(results)
    for result in results:
        file = result["file"]
        predicted = result["label"]
        actual = ground_truth.get(file, {}).get("label", "Unknown")
        language = result["language"]
        processing_time = result["processing_time"]
        if actual == "Unknown":
            print(f"Warning: No ground truth for {file}")
            total -= 1
            continue
        if actual == "Non-Scam":
            actual = "Safe"
        is_correct = predicted == actual
        if is_correct:
            correct += 1
        advice = result.get("advice", "")
        print(f"File: {file}, Language: {language}, Actual: {actual}, Predicted: {predicted}, Advice: {advice}, "
              f"Correct: {is_correct}, Processing Time: {processing_time:.2f}s")

    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_processing_time = sum(r["processing_time"] for r in results) / len(results) if results else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"Average processing time: {avg_processing_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    run_batch_test(
        audio_files_pattern="datasets/ne/test_audio_nepali/*.wav",
        ground_truth_file="ground_truth/ground_truth_nepali.json",
        output_file="results/ne/test_results_nepali.json",
    )
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
