import streamlit as st
import pandas as pd
import time
import random
import requests  # Placeholder for future backend integration
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av

# Placeholder for backend URL (uncomment and set when backend is available)
BACKEND_URL = "http://127.0.0.1:8000"
PROCESS_AUDIO_ENDPOINT = f"{BACKEND_URL}/audio"
GENERATE_REPORT_ENDPOINT = f"{BACKEND_URL}/alert"

# Simulated keyword list (empty initially, small chance of detection for testing)
keywords = []  # Empty to start; add keywords via backend later

# Home page
st.title("Voice Scam Shield (Standalone Audio Recorder)")

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

    # Audio callback to simulate transcription
    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        # Simulate transcription; no actual audio processing
        # TODO: For backend integration, send audio frame to backend
        # Example:
        audio_data = frame.to_ndarray().tobytes()
        response = requests.post(PROCESS_AUDIO_ENDPOINT, data=audio_data)
        if response.status_code == 200:
            data = response.json()
            # Update with real transcript, keyword, score, label
        return frame

    # Start/stop recording
    if not st.session_state["recording_active"] and not st.session_state["show_report"]:
        if st.button(t["start_recording"]):
            st.session_state["recording_active"] = True
            st.session_state["call_segments"] = []
            st.session_state["detected_keywords"] = set()
            st.session_state["start_time"] = time.time()
            st.session_state["show_report"] = False
            st.success(f"{t['start_recording']} (simulated).")

    if st.session_state["recording_active"]:
        # Start WebRTC streamer for audio capture
        ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"audio": True, "video": False},
            audio_frame_callback=audio_frame_callback,
        )

        # Placeholders for live updates
        risk_placeholder = st.empty()
        transcript_placeholder = st.empty()

        # Simulate real-time transcription and analysis
        elapsed_time = int(time.time() - st.session_state["start_time"]) if st.session_state["start_time"] else 0
        segment_count = len(st.session_state["call_segments"])
        for _ in range(100):
            if not st.session_state["recording_active"] or not ctx.state.playing:
                break
            # Simulate transcription
            transcript = {
                "user": f"User spoke something (segment {segment_count + 1})",
                "caller": f"Caller responded (segment {segment_count + 1})"
            }
            # Simulate scam detection
            score = random.randint(0, 100)
            if score < 50:
                label = "Safe"
                keyword = None
            elif score < 80:
                label = "Suspicious"
                keyword = "test keyword" if random.random() < 0.2 else None  # 20% chance of keyword
            else:
                label = "Scam"
                keyword = "test keyword" if random.random() < 0.2 else None
            st.session_state["current_data"] = {"transcript": transcript, "keyword": keyword, "score": score, "label": label}

            # Update risk display with keyword
            color = {"Safe": "green", "Suspicious": "yellow", "Scam": "red"}.get(label, "gray")
            risk_placeholder.markdown(
                f"<h3 style='color:{color}'>Risk: {label} ({score}%)</h3><p>Keyword: {keyword or 'None'}</p>",
                unsafe_allow_html=True
            )

            # Update transcript without timestamps
            transcript_placeholder.markdown(
                f"**User:** {transcript['user']}<br>**Caller:** {transcript['caller']}",
                unsafe_allow_html=True
            )

            # Add segment to report with timestamp
            timestamp = f"00:{elapsed_time // 60:02d}:{elapsed_time % 60:02d}"
            segment = {
                "Timestamp": timestamp,
                "Label": label,
                "Keyword": keyword or "None"
            }
            st.session_state["call_segments"].append(segment)
            if keyword:
                st.session_state["detected_keywords"].add(keyword)

            # Stop Recording button
            if st.button(t["stop_recording"]):
                st.session_state["recording_active"] = False
                st.session_state["show_report"] = True
                st.rerun()
                break

            time.sleep(2)
            elapsed_time += 2
            segment_count += 1
            st.rerun()

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

    uploaded_file = st.file_uploader(t["choose_file"], type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        # Simulate file processing
        st.write(f"{t['processing_file']}: {uploaded_file.name}")
        time.sleep(1)
        
        # TODO: For backend, send file to backend for transcription
        # Example:
        files = {'file': uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/analyze_file", files=files)
        if response.status_code == 200:
            report_data = response.json()
            segments = report_data['segments']
        else:
        # Simulate transcription based on file name length
            segment_count = min(max(len(uploaded_file.name) // 5, 1), 5)  # 1-5 segments
        report_segments = []
        detected_keywords = set()
        for i in range(segment_count):
            score = random.randint(0, 100)
            if score < 50:
                label = "Safe"
                keyword = None
            elif score < 80:
                label = "Suspicious"
                keyword = "test keyword" if random.random() < 0.2 else None
            else:
                label = "Scam"
                keyword = "test keyword" if random.random() < 0.2 else None
            if label in ["Suspicious", "Scam"]:
                segment = {
                    "Timestamp": f"00:00:{i*15:02d}",
                    "Label": label,
                    "Keyword": keyword or "None"
                }
                report_segments.append(segment)
                if keyword:
                    detected_keywords.add(keyword)

        keywords_str = ", ".join(detected_keywords) or "None"
        summary = f"Keyword detected: {keywords_str}"
        
        st.subheader(t["summary"])
        st.write(f"{t['summary']}: {summary}")
        
        # Display only Suspicious and Scam segments
        if report_segments:
            df = pd.DataFrame(report_segments)
            st.dataframe(df)
        else:
            st.write(t["no_segments"])
        
        # Download report as CSV
        csv = pd.DataFrame(report_segments).to_csv(index=False)
        st.download_button(t["download_report"], csv, "file_report.csv", "text/csv")