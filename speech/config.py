from dataclasses import dataclass
from typing import Dict, Set
import json
import os

def load_keywords_from_json(filepath: str) -> Set[str]:
    """Load keywords from a JSON file using a path relative to this file."""
    base_dir = os.path.dirname(__file__)
    abs_path = os.path.join(base_dir, filepath)
    with open(abs_path, "r", encoding="utf-8") as f:
        keywords = json.load(f)
    # Filter out generic keywords to reduce false positives
    generic_keywords = {
        "hello", "please", "form", "now", "open", "only",
        "hola", "por favor", "formulario", "ahora", "abrir", "solo",
        "bonjour", "s'il vous plaît", "formulaire", "maintenant", "ouvrir", "seulement",
        "नमस्ते", "कृपया", "फारम", "अब", "खोल्नु", "मात्र",
        "नमस्ते", "कृपया", "प्रपत्र", "अब", "खोलना", "केवल",
        "नमस्ति", "कृपया", "प्रपत्रम्", "अद्य", "उद्घाटति", "केवलम्"
    }
    return {kw.lower() for kw in keywords if kw.lower() not in generic_keywords}

# Load keywords for each language
scam_keywords: Dict[str, Set[str]] = {
    "en": load_keywords_from_json("scam_keywords_en.json"),
    "es": load_keywords_from_json("scam_keywords_es.json"),
    "fr": load_keywords_from_json("scam_keywords_fr.json"),
    "ne": load_keywords_from_json("scam_keywords_ne.json"),
    "hi": load_keywords_from_json("scam_keywords_hi.json"),
    "sa": load_keywords_from_json("scam_keywords_sa.json")
}

@dataclass
class StreamingConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Reduce chunk size for latency while keeping overlap for context
    chunk_seconds: float = 1.5
    overlap_seconds: float = 0.5
    # Multilingual model name for faster-whisper (ctranslate2)
    model_name: str = "base"
    # Device/precision for faster-whisper; "metal" preferred on macOS, fallback auto/cpu
    device: str = "auto"  # one of: "metal", "cpu", "auto"
    compute_type: str = "int8"  # e.g., "int8", "int8_float16", "float16", "float32"
    use_fp16: bool = False
    duration_limit_seconds: float = 0.0