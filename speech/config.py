from dataclasses import dataclass
from typing import Dict, Set
import json
import os

def load_keywords_from_json(filepath: str) -> Set[str]:
    """Load keywords from JSON, trying speech/keywords then repo-root keywords/."""
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "keywords", filepath),
        os.path.join(os.path.dirname(base_dir), "keywords", filepath),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), "keywords", filepath),
    ]
    keywords = []
    for path in candidates:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    keywords = json.load(f)
                    break
        except Exception:
            continue
    # Filter out generic keywords to reduce false positives
    generic_keywords = {
        "hello", "please", "form", "now", "open", "only",
        "hola", "por favor", "formulario", "ahora", "abrir", "solo",
        "bonjour", "s'il vous plaît", "formulaire", "maintenant", "ouvrir", "seulement",
        "नमस्ते", "कृपया", "फारम", "अब", "खोल्नु", "मात्र",
        "नमस्ते", "कृपया", "प्रपत्र", "अब", "खोलना", "केवल",
        "नमस्ति", "कृपया", "प्रपत्रम्", "अद्य", "उद्घाटति", "केवलम्"
    }
    return {kw.lower() for kw in keywords if isinstance(kw, str) and kw.lower() not in generic_keywords}

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
    model_name: str = "tiny"
    # Device/precision for faster-whisper; "metal" preferred on macOS, fallback auto/cpu
    device: str = "auto"  # one of: "metal", "cpu", "auto"
    compute_type: str = "int8"  # e.g., "int8", "int8_float16", "float16", "float32"
    use_fp16: bool = False
    beam_size: int = 5
    duration_limit_seconds: float = 0.0
    # If set, force transcription to this language code (e.g., "hi", "en")
    force_language: str | None = None
    # Restrict output to these languages; if detection yields other languages, we remap/re-run
    allowed_languages: tuple[str, ...] = ("en", "ne", "hi", "es", "sa", "fr")
    # Prefer Nepali over Hindi when strong Devanagari evidence
    prefer_nepali_over_hindi: bool = True
    # When True, automatically lock language based on script detection (e.g., Devanagari → "hi")
    auto_language_lock: bool = True
    # Threshold (0..1) of script ratio to trigger auto lock for a language (e.g., Devanagari for Hindi)
    language_lock_script_threshold: float = 0.35