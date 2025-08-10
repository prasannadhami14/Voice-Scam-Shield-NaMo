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

        else:
            st.write("None")