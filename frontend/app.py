import streamlit as st
import pandas as pd
import time
import requests  # Backend integration
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, AudioProcessorBase
import av
import mimetypes
import asyncio
import threading
import queue
import numpy as np
import websockets
import io
import soundfile as sf
import json
from urllib.parse import urlparse, urlunparse

# Placeholder for backend URL (uncomment and set when backend is available)
BACKEND_URL = "http://127.0.0.1:8000"

# Styles for keyword highlighting
st.markdown(
    """
<style>
.kw-chip {
  display: inline-block;
  padding: 0.2rem 0.5rem;
  margin: 0.2rem 0.35rem 0 0;
  background-color: #ffe8e8;
  border: 1px solid #ff4d4f;
  color: #a8071a;
  border-radius: 12px;
  font-size: 0.85rem;
}
.kw-highlight {
  background-color: #fff3cd;
  border-bottom: 2px solid #ff4d4f;
  padding: 0 2px;
}
</style>
""",
    unsafe_allow_html=True,
)

LANG_TO_CODE = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Nepali": "ne",
    # Map Sanskrit to Hindi for transcription support
    "Sanskrit": "hi",
}

def highlight_keywords(text: str, found_keywords) -> str:
    import re
    if not text:
        return ""
    kws = [kw for kw in (found_keywords or []) if isinstance(kw, str) and kw.strip()]
    if not kws:
        return text
    # Longest first to avoid partial overlaps
    kws = sorted(set(kws), key=len, reverse=True)
    pattern = re.compile(r"(" + "|".join(re.escape(k) for k in kws) + r")", re.IGNORECASE)
    return pattern.sub(lambda m: f'<span class="kw-highlight">{m.group(0)}</span>', text)

def analyze_audio_with_backend(file_name: str, file_bytes: bytes, language_code: str):
    try:
        guessed_mime, _ = mimetypes.guess_type(file_name)
        content_type = guessed_mime if guessed_mime and guessed_mime.startswith("audio/") else "audio/wav"
        files = {
            "file": (file_name, file_bytes, content_type),
        }
        data = {"language": language_code}
        resp = requests.post(f"{BACKEND_URL}/analyze-audio", files=files, data=data, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Backend error {resp.status_code}", "detail": resp.text}
    except Exception as e:
        return {"error": str(e)}


def _compute_ws_url(http_url: str) -> str:
    """Compute the WebSocket URL from the configured HTTP backend URL."""
    parsed = urlparse(http_url)
    if parsed.scheme not in ("http", "https"):
        # Assume user already provided ws/wss URL
        return http_url.rstrip("/") + "/audio/stream"
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_parsed = parsed._replace(scheme=ws_scheme)
    base_ws = urlunparse(ws_parsed).rstrip("/")
    return f"{base_ws}/audio/stream"


def _start_ws_sender_once():
    """Start a background WebSocket client to send audio chunks to backend and receive results."""
    if st.session_state.get("ws_thread_started"):
        return

    st.session_state.setdefault("ws_queue", queue.Queue(maxsize=50))
    st.session_state.setdefault("results_queue", queue.Queue(maxsize=50))
    st.session_state.setdefault("live_result", None)

    ws_url = _compute_ws_url(BACKEND_URL)
    ws_queue = st.session_state["ws_queue"]
    results_queue = st.session_state["results_queue"]

    def _run():
        async def _ws_loop():
            while True:
                try:
                    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
                        async def _recv_task():
                            while True:
                                try:
                                    msg = await ws.recv()
                                except Exception:
                                    break
                                # Backend sends JSON text frames
                                try:
                                    data = json.loads(msg) if isinstance(msg, (str, bytes)) else msg
                                    # push to results queue for main thread to consume
                                    try:
                                        results_queue.put_nowait(data)
                                    except queue.Full:
                                        # drop if overflowing
                                        pass
                                except Exception:
                                    pass

                        async def _send_task():
                            while True:
                                chunk = ws_queue.get()
                                if chunk is None:
                                    try:
                                        await ws.close()
                                    except Exception:
                                        pass
                                    return
                                try:
                                    await ws.send(chunk)
                                except Exception:
                                    # Put back the chunk if needed or drop
                                    pass

                        await asyncio.gather(_recv_task(), _send_task())
                except Exception:
                    # Retry connection after short delay
                    time.sleep(1.0)
                    continue

        asyncio.run(_ws_loop())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    st.session_state["ws_thread_started"] = True


# Global reference to avoid calling Streamlit APIs from audio callback thread
_STREAM_STATE = {"ws_queue": None}

def _set_global_stream_state_from_session():
    try:
        _STREAM_STATE["ws_queue"] = st.session_state.get("ws_queue")
    except Exception:
        _STREAM_STATE["ws_queue"] = None

# Language selection for multilingual support
language = st.selectbox("Select Language", ["English", "Spanish", "French", "Hindi", "Nepali", "Sanskrit"])
translations = {
    "English": {
        "title": "Voice Scam Shield",
        "record_audio": "Record Audio",
        "upload_file": "Upload Audio File for Analysis",
        "start_recording": "Start Recording",
        "stop_recording": "Stop Recording",
        "summary": "Recording Summary",
        "flagged_segments": "Flagged Segments",
        "no_segments": "No flagged segments.",
        "download_report": "Download Report",
        "processing_file": "Processing file",
        "choose_file": "Choose an audio file"
    },
    "Spanish": {
        "title": "Escudo de Estafas de Voz",
        "record_audio": "Grabar Audio",
        "upload_file": "Subir Archivo de Audio para Análisis",
        "start_recording": "Iniciar Grabación",
        "stop_recording": "Detener Grabación",
        "summary": "Resumen de la Grabación",
        "flagged_segments": "Segmentos Marcados",
        "no_segments": "No hay segmentos marcados.",
        "download_report": "Descargar Informe",
        "processing_file": "Procesando archivo",
        "choose_file": "Elige un archivo de audio"
    },
    "French": {
        "title": "Bouclier Antifraude Vocale",
        "record_audio": "Enregistrer l'Audio",
        "upload_file": "Télécharger un Fichier Audio pour Analyse",
        "start_recording": "Démarrer l'Enregistrement",
        "stop_recording": "Arrêter l'Enregistrement",
        "summary": "Résumé de l'Enregistrement",
        "flagged_segments": "Segments Signalés",
        "no_segments": "Aucun segment signalé.",
        "download_report": "Télécharger le Rapport",
        "processing_file": "Traitement du fichier",
        "choose_file": "Choisissez un fichier audio"
    },
    "Hindi": {
        "title": "वॉयस स्कैम शील्ड",
        "record_audio": "ऑडियो रिकॉर्ड करें",
        "upload_file": "विश्लेषण के लिए ऑडियो फ़ाइल अपलोड करें",
        "start_recording": "रिकॉर्डिंग शुरू करें",
        "stop_recording": "रिकॉर्डिंग रोकें",
        "summary": "रिकॉर्डिंग सारांश",
        "flagged_segments": "चिह्नित खंड",
        "no_segments": "कोई चिह्नित खंड नहीं।",
        "download_report": "रिपोर्ट डाउनलोड करें",
        "processing_file": "फ़ाइल प्रसंस्करण कर रहा है",
        "choose_file": "एक ऑडियो फ़ाइल चुनें"
    },
    "Nepali": {
        "title": "भ्वाइस स्क्याम शिल्ड",
        "record_audio": "अडियो रेकर्ड गर्नुहोस्",
        "upload_file": "विश्लेषणको लागि अडियो फाइल अपलोड गर्नुहोस्",
        "start_recording": "रिकर्डिङ सुरु गर्नुहोस्",
        "stop_recording": "रिकर्डिङ रोक्नुहोस्",
        "summary": "रिकर्डिङ सारांश",
        "flagged_segments": "फ्ल्याग गरिएका सेगमेन्टहरू",
        "no_segments": "कुनै फ्ल्याग गरिएका सेगमेन्टहरू छैनन्।",
        "download_report": "रिपोर्ट डाउनलोड गर्नुहोस्",
        "processing_file": "फाइल प्रशोधन गर्दै",
        "choose_file": "एउटा अडियो फाइल छान्नुहोस्"
    },
    "Sanskrit": {
        "title": "स्वर घोटक रक्षक",
        "record_audio": "शब्द रेकॉर्ड कुरु",
        "upload_file": "विश्लेषणाय ऑडियो फाइल अपलोड कुरु",
        "start_recording": "रिकॉर्डिंग आरम्भ कुरु",
        "stop_recording": "रिकॉर्डिंग स्थगित कुरु",
        "summary": "रिकॉर्डिंग सारांश",
        "flagged_segments": "चिह्नित खण्ड",
        "no_segments": "न चिह्नित खण्ड।",
        "download_report": "रिपोर्ट डाउनलोड कुरु",
        "processing_file": "फाइल प्रसंस्करण",
        "choose_file": "एकं ऑडियो फाइल वृणु"
    }
}

# Apply selected language
t = translations[language]

# Localized title
st.title(t["title"])

# Mode selection
option = st.selectbox("Choose Mode", (t["record_audio"], t["upload_file"]))

if option == t["record_audio"]:
    st.header(t["record_audio"])

    # Initialize session state
    if "recording_active" not in st.session_state:
        st.session_state["recording_active"] = False
        st.session_state["current_data"] = None
        st.session_state["call_segments"] = []
        st.session_state["start_time"] = None
        st.session_state["show_report"] = False
        st.session_state["detected_keywords"] = set()

    # WebRTC configuration for audio recording
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # Audio processor to capture outgoing mic audio in SENDONLY mode
    class WSForwardingAudioProcessor(AudioProcessorBase):
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            try:
                pcm = frame.to_ndarray()
                if pcm.ndim == 2:
                    mono = pcm.mean(axis=0)
                else:
                    mono = pcm
                if mono.dtype.kind in ("f",):
                    mono = np.clip(mono, -1.0, 1.0)
                    mono = (mono * 32767.0).astype(np.int16)
                elif mono.dtype != np.int16:
                    mono = mono.astype(np.int16)

                buf = io.BytesIO()
                sr = getattr(frame, "sample_rate", 16000) or 16000
                sf.write(buf, mono, samplerate=int(sr), subtype="PCM_16", format="WAV")
                data = buf.getvalue()
                q = _STREAM_STATE.get("ws_queue")
                if q is not None and not q.full():
                    try:
                        q.put_nowait(data)
                    except Exception:
                        pass
            except Exception:
                pass
            return frame

    # Start/stop recording
    if not st.session_state["recording_active"] and not st.session_state["show_report"]:
        if st.button(t["start_recording"]):
            st.session_state["recording_active"] = True
            st.session_state["call_segments"] = []
            st.session_state["detected_keywords"] = set()
            st.session_state["live_transcript"] = ""
            st.session_state["live_keywords"] = set()
            st.session_state["live_keyword_score"] = 0.0
            st.session_state["start_time"] = time.time()
            st.session_state["show_report"] = False
            st.success(f"{t['start_recording']} (simulated).")

    if st.session_state["recording_active"]:
        # Start WebRTC streamer for audio capture
        _start_ws_sender_once()
        # ensure global stream state is updated for callback thread
        _set_global_stream_state_from_session()

        ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=WSForwardingAudioProcessor,
        )

        # Simple status while recording (live results from backend if available)
        st.info("Recording... stop to view results. Live streaming to backend enabled.")
        # Drain results_queue in main thread and expose a recent live_result
        try:
            rq = st.session_state.get("results_queue")
            if rq is not None:
                drained = None
                while not rq.empty():
                    drained = rq.get_nowait()
                if drained is not None:
                    st.session_state["live_result"] = drained if isinstance(drained, dict) else st.session_state.get("live_result")
        except Exception:
            pass

        live = st.session_state.get("live_result")
        if isinstance(live, dict):
            # Accumulate transcript
            trans = live.get("transcription", {}) or {}
            new_text = (trans.get("text") or "").strip()
            if new_text:
                # naive dedup: append only if differs from recent tail
                tail = st.session_state.get("live_transcript", "")[-400:]
                if new_text not in tail:
                    st.session_state["live_transcript"] = (st.session_state.get("live_transcript", "") + " " + new_text).strip()

            # Accumulate keywords
            kw = live.get("keywords", {}) or {}
            found = kw.get("keywords_found", []) or []
            try:
                st.session_state["live_keywords"].update({k for k in found if isinstance(k, str)})
            except Exception:
                st.session_state["live_keywords"] = set(found)
            st.session_state["live_keyword_score"] = max(st.session_state.get("live_keyword_score", 0.0), float(kw.get("keyword_score", 0.0)))

            # Add a flagged segment if risky or keywords found
            try:
                risk = int(live.get("risk", 0))
                label = live.get("label", "")
                rationale = live.get("rationale", "")
                elapsed = time.time() - (st.session_state.get("start_time") or time.time())
                if risk >= 60 or found:
                    st.session_state["call_segments"].append({
                        "t": round(elapsed, 1),
                        "risk": risk,
                        "label": label,
                        "rationale": rationale,
                        "keywords": ", ".join(sorted(set(found)))
                    })
            except Exception:
                pass

            # UI: live header
            st.caption("Live detection:")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Risk", f"{risk if 'risk' in locals() else live.get('risk', 0)}")
            with cols[1]:
                st.metric("Label", f"{label if 'label' in locals() else live.get('label', 'Safe')}")
            with cols[2]:
                st.caption(rationale if 'rationale' in locals() else live.get('rationale', ''))

            # UI: live transcript with keyword highlights (keep it reactive)
            if "_live_transcript_placeholder" not in st.session_state:
                st.session_state["_live_transcript_placeholder"] = st.empty()
            transcript_ph = st.session_state["_live_transcript_placeholder"]
            current_text = st.session_state.get("live_transcript", "")
            if current_text:
                transcript_ph.markdown("**Live transcription:**\n\n" + highlight_keywords(current_text, list(st.session_state.get("live_keywords", set()))), unsafe_allow_html=True)
            else:
                transcript_ph.markdown("**Live transcription:** (listening...) ")

            # UI: keyword chips
            if st.session_state.get("live_keywords"):
                st.markdown("**Detected scam words (live):**")
                chips = "".join(f"<span class='kw-chip'>{kw}</span>" for kw in sorted(st.session_state["live_keywords"], key=str.lower))
                st.markdown(chips, unsafe_allow_html=True)

        # Stop recording control
        if st.button(t["stop_recording"]):
            st.session_state["recording_active"] = False
            st.session_state["show_report"] = True
            st.success(t["stop_recording"])
            # Gracefully signal sender to stop
            try:
                q = st.session_state.get("ws_queue")
                if q:
                    q.put(None)
            except Exception:
                pass

    if st.session_state["show_report"]:
        # Generate dynamic summary based on detected keywords
        keywords_str = ", ".join(st.session_state["detected_keywords"]) or "None"
        summary = f"Keyword detected: {keywords_str}"

        st.subheader(t["summary"])
        st.write(f"{t['summary']}: {summary}")
        
        # Display flagged segments with timestamps (all labels for recording)
        segments = st.session_state["call_segments"]
        if segments:
            df = pd.DataFrame(segments)
            st.dataframe(df)
        else:
            st.write(t["no_segments"])
        
        # Download report as CSV
        csv = pd.DataFrame(segments).to_csv(index=False)
        st.download_button(t["download_report"], csv, "recording_report.csv", "text/csv")

        # Button to reset and start a new recording
        if st.button("Start New Recording"):
            st.session_state["recording_active"] = False
            st.session_state["show_report"] = False
            st.session_state["call_segments"] = []
            st.session_state["start_time"] = None
            st.session_state["detected_keywords"] = set()
            st.rerun()

elif option == t["upload_file"]:
    st.header(t["upload_file"])

    uploaded_file = st.file_uploader(t["choose_file"], type=["wav", "mp3", "m4a", "flac"]) 

    if uploaded_file is not None:
        st.write(f"{t['processing_file']}: {uploaded_file.name}")
        file_bytes = uploaded_file.getvalue()
        lang_code = LANG_TO_CODE.get(language, "en")
        with st.spinner(t["processing_file"] + "..."):
            result = analyze_audio_with_backend(uploaded_file.name, file_bytes, lang_code)

        if "error" in result:
            st.error(f"Analysis error: {result['error']}")
            if result.get("detail"):
                st.caption(result["detail"])
            st.stop()

        analysis = result.get("result", {})
        is_scam = analysis.get("is_scam", False)
        conf = analysis.get("confidence", 0.0)
        rationale = analysis.get("rationale", "")
        details = analysis.get("details", {})
        transcription = details.get("transcription", {})
        keywords_info = details.get("keywords", {})
        found = keywords_info.get("keywords_found", [])

        # Header result
        st.markdown(f"### {'Scam detected' if is_scam else 'Safe'} — Confidence: {conf:.0%}")
        if rationale:
            st.write(rationale)

        # Show transcribed text with highlighted scam words
        text = transcription.get("text", "")
        if text:
            st.markdown("**Transcription (scam words highlighted):**")
            st.markdown(highlight_keywords(text, found), unsafe_allow_html=True)
            st.caption(f"Language: {transcription.get('language','?')} | Confidence: {transcription.get('confidence',0):.0%}")
        else:
            st.info("No transcription available")

        # Show list of scam words
        st.markdown("**Detected scam words:**")
        if found:
            chips = "".join(f"<span class='kw-chip'>{kw}</span>" for kw in sorted(set(found), key=str.lower))
            st.markdown(chips, unsafe_allow_html=True)
            st.caption(f"Keyword score: {keywords_info.get('keyword_score',0):.0%} | Suspicious: {'Yes' if keywords_info.get('is_suspicious') else 'No'}")
        else:
            st.write("None")