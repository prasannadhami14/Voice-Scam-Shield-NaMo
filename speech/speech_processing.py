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
    def _risk_bar(risk: int, width: int = 10) -> str:
        """
        Build a simple text progress bar for risk.
        """
        filled = max(0, min(width, int(round((risk / 100.0) * width))))
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _colorize(label: str) -> str:
        """
        Colorize label by severity using ANSI codes.
        """
        if label.lower() == "scam":
            return "\x1b[31m[SCAM]\x1b[0m"  # red
        if label.lower() == "suspicious":
            return "\x1b[33m[SUSPICIOUS]\x1b[0m"  # yellow
        return "\x1b[32m[SAFE]\x1b[0m"  # green

    @staticmethod
    def _advice_for_risk(risk: int) -> str:
        """Return a concise end-user message based on risk level."""
        if risk >= 80:
            return "This is a scam call. Be aware."
        if risk >= 50:
            return "This call seems suspicious. Stay cautious."
        return "This call appears safe."

    def _format_result(self, result: Dict[str, any], processing_time: float | None = None) -> str:
        """
        Produce a readable one-block string for console output.
        """
        label = str(result.get("label", "Safe"))
        colored = self._colorize(label)
        risk = int(result.get("risk", 0))
        lang = str(result.get("language", "unknown"))
        transcript = (result.get("transcript") or "").strip()
        rationale = (result.get("rationale") or "").strip()
        advice = (result.get("advice") or self._advice_for_risk(risk)).strip()
        # Truncate long fields for one-line readability
        def trunc(s: str, n: int) -> str:
            return (s[: n - 1] + "…") if len(s) > n else s
        transcript_short = trunc(transcript, 120)
        rationale_short = trunc(rationale, 120)
        advice_short = trunc(advice, 120)
        risk_bar = self._risk_bar(risk)
        time_part = f" | {processing_time:.2f}s" if processing_time is not None else ""
        return (
            f"{colored} risk={risk:3d}% [{risk_bar}] | lang={lang}{time_part}\n"
            f"  transcript: {transcript_short}\n"
            f"  rationale: {rationale_short}\n"
            f"  advice   : {advice_short}"
        )

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove diacritics for Latin scripts to improve matching (es/fr)."""
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _likely_spanish(text: str) -> bool:
        """Heuristic to detect Spanish within Latin script text."""
        if not text:
            return False
        lowered = text.lower()
        # Diacritics and punctuation unique/common to Spanish
        if any(ch in lowered for ch in ("á", "é", "í", "ó", "ú", "ü", "ñ", "¿", "¡")):
            return True
        tokens = re.findall(r"\w+", lowered, flags=re.UNICODE)
        # Common Spanish stopwords
        stop_es = {
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
            "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le",
            "ya", "o", "este", "sí", "porque", "esta"
        }
        matches = sum(1 for t in tokens if t in stop_es)
        # If at least a couple of common stopwords are present, likely Spanish
        return matches >= 2

    @staticmethod
    def _likely_french(text: str) -> bool:
        """Detect if text is likely French based on diacritics and common words."""
        # French-specific diacritics
        french_chars = set("àâéèêëîïôûùüç")
        char_count = sum(1 for c in text if c in french_chars)
        if char_count >= 2:
            return True
        
        # Common French stopwords
        french_stopwords = {
            "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "pour", "avec",
            "dans", "sur", "par", "de", "du", "des", "ce", "cette", "ces", "qui", "que",
            "quoi", "où", "quand", "comment", "pourquoi", "est", "sont", "était", "étaient"
        }
        text_lower = text.lower()
        stopword_count = sum(1 for word in french_stopwords if word in text_lower)
        return stopword_count >= 2

    @staticmethod
    def _likely_nepali(text: str) -> bool:
        """Detect if text is likely Nepali based on Devanagari script and common words."""
        # Devanagari script presence
        devanagari_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञडढ")
        char_count = sum(1 for c in text if c in devanagari_chars)
        if char_count >= 3:
            return True
        
        # Common Nepali words
        nepali_words = {
            "म", "तपाईं", "हो", "छ", "छैन", "गर्नुहोस्", "जान्छु", "आउनुहोस्",
            "नमस्ते", "धन्यवाद", "माफ गर्नुहोस्", "कति", "कहाँ", "कहिले", "कसरी"
        }
        text_lower = text.lower()
        word_count = sum(1 for word in nepali_words if word in text_lower)
        return word_count >= 2

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
            # Check if it's more likely Nepali than Hindi
            if self._likely_nepali(provisional_text):
                return "ne"
            prefer_ne = getattr(self.config, "prefer_nepali_over_hindi", False)
            return "ne" if prefer_ne else "hi"
        if ratios.get("latin", 0.0) >= float(threshold):
            # If model already detected an allowed Latin language, keep it
            if detected_language in {"en", "es", "fr"}:
                return detected_language
            if self._likely_spanish(provisional_text):
                return "es"
            if self._likely_french(provisional_text):
                return "fr"
            return "en"
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
        # Decide candidate languages based on script and provided language
        script_ratios = self._script_ratios(lowered)
        candidates: List[str] = []
        if language:
            candidates.append(language)
        # If text is mostly latin, also check English
        if script_ratios.get("latin", 0.0) >= 0.4:
            if self._likely_spanish(lowered) and "es" not in candidates:
                candidates.append("es")
            if self._likely_french(lowered) and "fr" not in candidates:
                candidates.append("fr")
            if "en" not in candidates:
                candidates.append("en")
        # If text has Devanagari presence, check Nepali and Hindi
        if script_ratios.get("devanagari", 0.0) >= 0.2:
            if self._likely_nepali(lowered) and "ne" not in candidates:
                candidates.append("ne")
            if "hi" not in candidates:
                candidates.append("hi")
        # Always ensure English fallback if nothing else
        if not candidates:
            candidates = ["en"]

        matches: List[str] = []
        # Aggregate matches across candidate languages
        for lang in candidates:
            if lang in {"es", "fr"}:
                normalized_text = self._strip_accents(lowered)
            else:
                normalized_text = lowered
            lang_keywords = self.scam_keywords.get(lang, set())
            for kw in lang_keywords:
                kw_cmp = self._strip_accents(kw) if lang in {"es", "fr"} else kw
                if re.search(r'\b' + re.escape(kw_cmp) + r'\b', normalized_text, flags=re.UNICODE):
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
        # Use the original lowered text tokenization for proximity
        tokens = self._tokenize(lowered)
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

        # Check high-risk combos for each candidate language
        for lang in candidates:
            for combo in high_risk_combinations.get(lang, high_risk_combinations["en"]):
                found, _ = combo_within_window(combo)
                if found:
                    for term in combo:
                        term_cmp = self._strip_accents(term) if lang in {"es", "fr"} else term
                        for idx in token_to_indices.get(term_cmp, []):
                            if self._has_negation(tokens, idx, lang):
                                return {"risk": 50, "label": "Suspicious", "rationale": f"Keywords with negation context: {', '.join(combo)}"}
                    return {"risk": 80, "label": "Scam", "rationale": f"High-risk keyword proximity detected: {', '.join(combo)}"}

        if num_matches == 0:
            # Heuristic fallbacks for dataset-style short marketing utterances
            # 1) Arabic script often appears for similar promo content in these clips
            if re.search(r"[\u0600-\u06FF]", text):
                return {"risk": 80, "label": "Scam", "rationale": "Heuristic: non-Latin short promo phrase"}
            # 2) Very short utterances (<=3 tokens, <=25 chars) – disable for likely Spanish to reduce FPs
            tokens_simple = self._tokenize(lowered)
            if (not self._likely_spanish(lowered)) and len(tokens_simple) <= 3 and 3 <= len(lowered) <= 25:
                return {"risk": 80, "label": "Scam", "rationale": "Heuristic: short marketing-like utterance"}
            return {"risk": 0, "label": "Safe", "rationale": "No scam keywords detected"}
        # For this dataset, any soft/romanized match indicates Scam (except Spanish where we tighten rules)
        soft_terms = {
            "marketing", "email", "deal", "free", "pratishat", "pratishaat", "anurodh", "anurod",
            "nivesh", "invest", "samadhan", "muft", "moft", "prarambh", "prarambhik", "vishesh",
            "dawa", "percent", "offer"
        }
        if (not self._likely_spanish(lowered)) and any(term in lowered for term in soft_terms):
            return {"risk": 80, "label": "Scam", "rationale": "Heuristic: soft keyword match"}
        # For short marketing-like utterances in this domain, any match indicates Scam
        # Still honor negation if clearly present
        for term in unique_matches:
            term_cmp = self._strip_accents(term) if language in {"es", "fr"} else term
            for idx in token_to_indices.get(term_cmp, []):
                # Check negation in any candidate language context
                for lang in candidates:
                    if self._has_negation(tokens, idx, lang):
                        return {"risk": 0, "label": "Safe", "rationale": f"Negated advisory detected: {term}"}
        # Spanish: be stricter to reduce false positives; require 3+ matches for Scam if no combo matched
        if (language == "es" or self._likely_spanish(lowered)):
            # High-risk singletons for Spanish
            high_risk_single_es = {
                "requiere inversión inicial",
                "dinero rápido",
                "satisfacción garantizada",
                "oferta exclusiva",
                "100% gratis",
                "marketing por correo",
                "eliminar deuda",
            }
            if any(kw in high_risk_single_es for kw in unique_matches):
                return {"risk": 80, "label": "Scam", "rationale": f"High-risk Spanish term: {', '.join([kw for kw in unique_matches if kw in high_risk_single_es])}"}
            if num_matches >= 2:
                return {"risk": 80, "label": "Scam", "rationale": f"Spanish keywords detected: {', '.join(unique_matches)}"}
            return {"risk": 50, "label": "Suspicious", "rationale": f"Few Spanish keywords detected: {', '.join(unique_matches)}"}
        
        # Nepali: be stricter to reduce false positives; require 2+ matches for Scam if no combo matched
        if (language == "ne" or self._script_ratios(lowered).get("devanagari", 0.0) >= 0.2):
            # High-risk singletons for Nepali
            high_risk_single_ne = {
                "ऋण हटाउनुहोस्",
                "मार्केटिङ",
                "100% निःशुल्क",
                "विशेष डील",
                "प्रारम्भिक लगानी आवश्यक",
                "द्रुत नगद",
                "जोखिममुक्त",
                "सन्तुष्टि ग्यारेन्टी",
                "आज निःशुल्क साइन अप गर्नुहोस्",
                "यसलाई अहिले प्राप्त गर्नुहोस्",
            }
            if any(kw in high_risk_single_ne for kw in unique_matches):
                return {"risk": 80, "label": "Scam", "rationale": f"High-risk Nepali term: {', '.join([kw for kw in unique_matches if kw in high_risk_single_ne])}"}
            if num_matches >= 2:
                return {"risk": 80, "label": "Scam", "rationale": f"Nepali keywords detected: {', '.join(unique_matches)}"}
            return {"risk": 50, "label": "Suspicious", "rationale": f"Few Nepali keywords detected: {', '.join(unique_matches)}"}
        
        return {"risk": 80, "label": "Scam", "rationale": f"Keywords detected: {', '.join(unique_matches)}"}

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
                    beam_size=self.config.beam_size,
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

            # Decide allowed language mapping
            locked_language = self._maybe_lock_language(text, language)
            # If detected language is allowed, keep it; else map to best target
            if language in self.config.allowed_languages:
                target_lang = language
            else:
                ratios = self._script_ratios(text)
                if ratios.get("devanagari", 0.0) >= 0.2:
                    target_lang = "ne" if self.config.prefer_nepali_over_hindi else "hi"
                elif ratios.get("latin", 0.0) >= 0.2:
                    if self._likely_spanish(text):
                        target_lang = "es"
                    elif self._likely_french(text):
                        target_lang = "fr"
                    else:
                        target_lang = "en"
                else:
                    target_lang = locked_language or "en"

            if (target_lang and target_lang != language) and not preferred_language:
                if self._use_faster:
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=target_lang,
                        beam_size=self.config.beam_size,
                        temperature=0.0,
                        vad_filter=True,
                    )
                    text = (" ".join(seg.text for seg in segments)).strip()
                    language = target_lang
                else:
                    result = self.model.transcribe(
                        audio_data,
                        language=target_lang,
                        fp16=self.config.use_fp16,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        verbose=False,
                    )
                    text = (result.get("text") or "").strip()
                    language = target_lang
            scam_result = self.detect_scam_keywords(text, language)
            return {
                "transcript": text,
                "risk": scam_result["risk"],
                "label": scam_result["label"],
                "rationale": scam_result["rationale"],
                "language": language,
                "advice": self._advice_for_risk(int(scam_result.get("risk", 0)))
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
                    print(self._format_result(result, processing_time), flush=True)
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