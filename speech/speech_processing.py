import queue
import threading
import time
import sys
import unicodedata
import numpy as np
import sounddevice as sd
import re
from typing import Set, Dict, List, Tuple
from config import StreamingConfig, scam_keywords

try:
    from faster_whisper import WhisperModel  # type: ignore
    _FASTER_WHISPER_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _FASTER_WHISPER_AVAILABLE = False
    import whisper  # fallback

class RealTimeTranscriber:
    """Real-time audio transcription and scam detection with multilingual support."""

    def __init__(self, config: StreamingConfig, scam_keywords_dict: Dict[str, Set[str]] = None):
        """
        Initialize transcriber with configuration and scam keywords.

        Args:
            config: StreamingConfig object with sample rate, model name, etc.
            scam_keywords_dict: Dict of language-specific keywords (default: scam_keywords from config).
        """
        self.config = config
        self.scam_keywords = scam_keywords_dict or scam_keywords
        # Configure model: prefer faster-whisper for latency
        if _FASTER_WHISPER_AVAILABLE:
            device = None if self.config.device == "auto" else self.config.device
            device = device or "auto"
            self.model = WhisperModel(
                self.config.model_name,
                device=device,
                compute_type=self.config.compute_type,
            )
            self._use_faster = True
        else:
            self.model = whisper.load_model(self.config.model_name)
            self._use_faster = False
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stop_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self.chunk_size_samples: int = int(self.config.chunk_seconds * self.config.sample_rate)
        self.overlap_size_samples: int = int(self.config.overlap_seconds * self.config.sample_rate)
        # Use 40 ms blocks for smoother buffering
        self.block_size: int = int(0.04 * self.config.sample_rate)

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove diacritics for Latin scripts to improve matching (es/fr)."""
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Unicode-aware tokenization; keeps Devanagari tokens as-is."""
        return re.findall(r"\w+", text, flags=re.UNICODE)

    @staticmethod
    def _script_ratios(text: str) -> Dict[str, float]:
        """Approximate fraction of characters by script family (latin, greek, devanagari)."""
        if not text:
            return {"latin": 0.0, "greek": 0.0, "devanagari": 0.0}
        total_letters = 0
        counts = {"latin": 0, "greek": 0, "devanagari": 0}
        for ch in text:
            code = ord(ch)
            # Basic Latin + Latin-1 Supplement + Latin Extended-A/B
            if (0x0041 <= code <= 0x024F) or (0x1E00 <= code <= 0x1EFF):
                counts["latin"] += 1
                total_letters += 1
            # Greek and Coptic
            elif 0x0370 <= code <= 0x03FF:
                counts["greek"] += 1
                total_letters += 1
            # Devanagari
            elif 0x0900 <= code <= 0x097F:
                counts["devanagari"] += 1
                total_letters += 1
        if total_letters == 0:
            return {k: 0.0 for k in counts}
        return {k: counts[k] / total_letters for k in counts}

    def _maybe_lock_language(self, provisional_text: str, detected_language: str) -> str:
        """Optionally lock language to 'hi' if Devanagari dominates and no forced language is set."""
        if getattr(self.config, "force_language", None):
            return self.config.force_language  # type: ignore
        if not getattr(self.config, "auto_language_lock", True):
            return detected_language
        ratios = self._script_ratios(provisional_text)
        threshold = getattr(self.config, "language_lock_script_threshold", 0.35)
        if ratios.get("devanagari", 0.0) >= float(threshold):
            return "hi"
        return detected_language

    @staticmethod
    def _has_negation(tokens: List[str], keyword_index: int, language: str, window: int = 6) -> bool:
        negations: Dict[str, Set[str]] = {
            "en": {"no", "not", "don't", "do", "never", "avoid"},
            "es": {"no", "nunca"},
            "fr": {"pas", "ne", "jamais"},
            "ne": {"न", "नहिँ", "नहीं", "नहि"},
            "hi": {"नहीं", "मत", "ना", "कभी"},
            "sa": {"मा", "न"},
        }
        negs = negations.get(language, negations["en"])
        start = max(0, keyword_index - window)
        context = set(tokens[start:keyword_index])
        return any(tok in context for tok in negs)

    def detect_scam_keywords(self, text: str, language: str) -> Dict[str, any]:
        """
        Detect scam keywords in text using regex with word boundaries and language-specific keywords.

        Args:
            text: Transcribed text to analyze.
            language: Detected language code (e.g., 'en', 'es', 'fr', 'ne', 'hi', 'sa').

        Returns:
            Dict with risk (0/50/80), label (Safe/Suspicious/Scam), and rationale.
        """
        if not text:
            return {"risk": 0, "label": "Safe", "rationale": "No scam keywords detected"}
        
        lowered = text.lower()
        # Accent-insensitive for Latin scripts (es, fr). Keep others intact.
        if language in {"es", "fr"}:
            lowered_norm = self._strip_accents(lowered)
        else:
            lowered_norm = lowered
        matches: List[str] = []
        lang_keywords = self.scam_keywords.get(language, self.scam_keywords["en"])  # Fallback to English
        for kw in lang_keywords:
            kw_cmp = self._strip_accents(kw) if language in {"es", "fr"} else kw
            # Use unicode word boundaries via tokenization check
            if re.search(r'\b' + re.escape(kw_cmp) + r'\b', lowered_norm, flags=re.UNICODE):
                matches.append(kw)
        
        # Remove redundant keywords
        unique_matches = []
        for kw in sorted(matches, key=len, reverse=True):
            if not any(kw in other and kw != other for other in unique_matches):
                unique_matches.append(kw)
        
        num_matches = len(unique_matches)
        
        # Language-specific high-risk combinations
        high_risk_combinations = {
            "en": [
                {"urgent", "payment", "code"},
                {"bank", "account", "verify"},
                {"money", "transfer", "urgent"},
                {"scam", "money"},
                {"fraud", "payment"}
            ],
            "es": [
                {"urgente", "pago", "código"},
                {"banco", "cuenta", "verificar"},
                {"dinero", "transferencia", "urgente"},
                {"estafa", "dinero"},
                {"fraude", "pago"}
            ],
            "fr": [
                {"urgent", "paiement", "code"},
                {"banque", "compte", "vérifier"},
                {"argent", "transfert", "urgent"},
                {"arnaque", "argent"},
                {"fraude", "paiement"}
            ],
            "ne": [
                {"आकस्मिक", "भुक्तानी", "कोड"},
                {"बैंक", "खाता", "प्रमाणीकरण"},
                {"पैसा", "स्थानान्तरण", "आकस्मिक"},
                {"ठगी", "पैसा"},
                {"धोखाधडी", "भुक्तानी"}
            ],
            "hi": [
                {"तत्काल", "भुगतान", "कोड"},
                {"बैंक", "खाता", "सत्यापन"},
                {"पैसा", "हस्तांतरण", "तत्काल"},
                {"घोटाला", "पैसा"},
                {"धोखाधड़ी", "भुगतान"}
            ],
            "sa": [
                {"शीघ्रम्", "संनादति", "सङ्केतः"},
                {"विपणिः", "खाता", "प्रमाणीकरणम्"},
                {"धनम्", "सङ्क्रमणम्", "शीघ्रम्"},
                {"वञ्चना", "धनम्"},
                {"कपटम्", "संनादति"}
            ]
        }
        # Proximity-based check: all words of combo must appear within a small window
        tokens = self._tokenize(lowered_norm)
        token_to_indices: Dict[str, List[int]] = {}
        for idx, tok in enumerate(tokens):
            token_to_indices.setdefault(tok, []).append(idx)

        def combo_within_window(combo: Set[str], window: int = 7) -> Tuple[bool, int]:
            # Return (found, min_span)
            indices_lists = []
            for term in combo:
                term_cmp = self._strip_accents(term) if language in {"es", "fr"} else term
                indices = token_to_indices.get(term_cmp, [])
                if not indices:
                    return False, 0
                indices_lists.append(indices)
            # brute-force small sets
            all_positions = [
                (i, positions)
                for i, positions in enumerate(indices_lists)
            ]
            # Flatten simple: take min/max among one index per term
            min_span = 1_000_000
            found = False
            # For small combos (<=3), brute force combinations
            from itertools import product
            for choice in product(*indices_lists):
                span = max(choice) - min(choice)
                if span <= window:
                    min_span = min(min_span, span)
                    found = True
            return found, (min_span if found else 0)

        for combo in high_risk_combinations.get(language, high_risk_combinations["en"]):
            found, _ = combo_within_window(combo)
            if found:
                # Negation handling: if negation appears before any term, downgrade to Suspicious
                for term in combo:
                    term_cmp = self._strip_accents(term) if language in {"es", "fr"} else term
                    for idx in token_to_indices.get(term_cmp, []):
                        if self._has_negation(tokens, idx, language):
                            return {"risk": 50, "label": "Suspicious", "rationale": f"Keywords with negation context: {', '.join(combo)}"}
                return {"risk": 80, "label": "Scam", "rationale": f"High-risk keyword proximity detected: {', '.join(combo)}"}

        if num_matches == 0:
            return {"risk": 0, "label": "Safe", "rationale": "No scam keywords detected"}
        elif num_matches <= 2:
            # Check negation context around any matched term
            for term in unique_matches:
                term_cmp = self._strip_accents(term) if language in {"es", "fr"} else term
                for idx in token_to_indices.get(term_cmp, []):
                    if self._has_negation(tokens, idx, language):
                        return {"risk": 0, "label": "Safe", "rationale": f"Negated advisory detected: {term}"}
            return {"risk": 50, "label": "Suspicious", "rationale": f"Keywords detected: {', '.join(unique_matches)}"}
        else:
            return {"risk": 80, "label": "Scam", "rationale": f"Multiple keywords detected: {', '.join(unique_matches)}"}

    def _audio_callback(self, indata, frames, ctime, status):
        """Callback for sounddevice to handle incoming audio."""
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        # indata is an ndarray (frames, channels) with dtype int16
        if indata.ndim == 2 and indata.shape[1] > 1:
            audio_i16 = indata.mean(axis=1).astype(np.int16)
        else:
            audio_i16 = indata.reshape(-1).astype(np.int16)
        audio = (audio_i16.astype(np.float32)) / 32768.0
        self.audio_queue.put(audio)

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            blocksize=self.block_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture."""
        self.stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def process_audio_chunk(self, audio_data: np.ndarray) -> Dict[str, any]:
        """
        Process an audio chunk: transcribe, detect language, and detect scam patterns.

        Args:
            audio_data: NumPy array of float32 audio samples at 16kHz.

        Returns:
            Dict with transcript, risk, label, rationale, and language.
        """
        try:
            preferred_language = getattr(self.config, "force_language", None)
            # First pass transcription (may be forced if user set it)
            if self._use_faster:
                segments, info = self.model.transcribe(
                    audio_data,
                    language=preferred_language,
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=True,
                )
                text = (" ".join(seg.text for seg in segments)).strip()
                language = (preferred_language or info.language or "en")
            else:
                result = self.model.transcribe(
                    audio_data,
                    language=preferred_language,
                    fp16=self.config.use_fp16,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    verbose=False,
                )
                text = (result.get("text") or "").strip()
                language = preferred_language or result.get("language", "en")

            # Optional re-transcribe if Hindi script detected but language isn't 'hi'
            locked_language = self._maybe_lock_language(text, language)
            if locked_language != language and not preferred_language:
                if self._use_faster:
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=locked_language,
                        beam_size=1,
                        temperature=0.0,
                        vad_filter=True,
                    )
                    text = (" ".join(seg.text for seg in segments)).strip()
                    language = locked_language
                else:
                    result = self.model.transcribe(
                        audio_data,
                        language=locked_language,
                        fp16=self.config.use_fp16,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        verbose=False,
                    )
                    text = (result.get("text") or "").strip()
                    language = locked_language
            scam_result = self.detect_scam_keywords(text, language)
            return {
                "transcript": text,
                "risk": scam_result["risk"],
                "label": scam_result["label"],
                "rationale": scam_result["rationale"],
                "language": language
            }
        except Exception as e:
            return {
                "transcript": "",
                "risk": 0,
                "label": "Safe",
                "rationale": f"Error processing audio: {str(e)}",
                "language": "unknown"
            }

    def run(self) -> None:
        """Run live audio transcription and scam detection until stopped."""
        print(
            f"Listening at {self.config.sample_rate/1000:.0f}kHz — chunk {self.config.chunk_seconds:.1f}s, "
            f"overlap {self.config.overlap_seconds:.1f}s\nPress Ctrl+C to stop...",
            flush=True,
        )

        rolling_buffer: List[np.ndarray] = []
        collected_samples = 0
        start_time = time.time()

        try:
            while not self.stop_event.is_set():
                if self.config.duration_limit_seconds > 0 and (time.time() - start_time) >= self.config.duration_limit_seconds:
                    break

                try:
                    block = self.audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                rolling_buffer.append(block)
                collected_samples += block.shape[0]

                if collected_samples >= self.chunk_size_samples:
                    audio_chunk = np.concatenate(rolling_buffer, axis=0)
                    audio_chunk = audio_chunk[-self.chunk_size_samples:]

                    if self.overlap_size_samples > 0:
                        overlap = audio_chunk[-self.overlap_size_samples:]
                        rolling_buffer = [overlap]
                        collected_samples = overlap.shape[0]
                    else:
                        rolling_buffer = []
                        collected_samples = 0

                    chunk_start = time.time()
                    result = self.process_audio_chunk(audio_chunk)
                    processing_time = time.time() - chunk_start
                    if processing_time > 2.0:
                        print(f"[WARNING] Latency exceeded: {processing_time:.2f} seconds")
                    print(f"Processing time: {processing_time:.2f} seconds")
                    print(result, flush=True)
                    if result.get("risk", 0) >= 80:
                        print("ALERT: High-risk scam indicators detected.", flush=True)

        finally:
            self.stop()

if __name__ == "__main__":
    import signal
    from config import StreamingConfig

    config = StreamingConfig()
    transcriber = RealTimeTranscriber(config)

    def handle_sigint(_sig, _frame):
        transcriber.stop()
    signal.signal(signal.SIGINT, handle_sigint)

    transcriber.start()
    transcriber.run()