import streamlit as st
import threading
import queue
import requests
import time
import json
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# Assume backend URLs (replace with actual FastAPI endpoints)
BACKEND_URL = "http://localhost:8000"  # Or your backend URL
START_CALL_URL = f"{BACKEND_URL}/start_call"
PROCESS_AUDIO_URL = f"{BACKEND_URL}/process_audio"
END_CALL_URL = f"{BACKEND_URL}/end_call"
PROCESS_FILE_URL = f"{BACKEND_URL}/process_file"

# Shared state for thread safety
lock = threading.Lock()
shared_data = {"score": None, "label": "Safe", "rationale": "", "transcript": {"user": "", "caller": ""}, "session_id": None}

# Home page
st.title("Voice Scam Shield")

option = st.selectbox("Choose Mode", ("Live Call", "Upload File"))

if option == "Live Call":
    st.header("Live Call Monitoring")

    # Start call session on backend
    if "session_id" not in shared_data or shared_data["session_id"] is None:
        response = requests.post(START_CALL_URL)
        if response.status_code == 200:
            shared_data["session_id"] = response.json()["session_id"]
            st.success(f"Call session started: {shared_data['session_id']}")
        else:
            st.error("Failed to start call session.")
            st.stop()

    # WebRTC configuration for STUN
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # Queue for audio frames
    audio_queue = queue.Queue()

    # Audio callback: Put frame in queue for processing
    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        audio_queue.put(frame)
        return frame  # Echo back for now (can modify if needed)

    # Start WebRTC streamer for audio capture (simulating call audio from mic)
    ctx = webrtc_streamer(
        key="live-call",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        audio=True,
        video=False,
        audio_frame_callback=audio_frame_callback,
    )

    # Thread to process audio queue and send to backend for VAD, diarization, detection
    def process_audio_thread():
        while True:
            if not audio_queue.empty():
                frame = audio_queue.get()
                pcm = frame.to_ndarray()  # Get raw audio data
                audio_bytes = pcm.tobytes()  # Convert to bytes

                # Send to backend with session_id
                payload = {"session_id": shared_data["session_id"], "audio_data": audio_bytes}
                try:
                    response = requests.post(PROCESS_AUDIO_URL, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        with lock:
                            shared_data["score"] = data.get("score", 0)
                            shared_data["label"] = data.get("label", "Safe")
                            shared_data["rationale"] = data.get("rationale", "")
                            shared_data["transcript"] = data.get("transcript", {"user": "", "caller": ""})
                except Exception as e:
                    pass  # Handle errors gracefully
            time.sleep(0.01)  # Avoid busy loop

    # Start processing thread if not already running
    if "audio_thread" not in st.session_state:
        thread = threading.Thread(target=process_audio_thread, daemon=True)
        thread.start()
        st.session_state["audio_thread"] = thread

    # Placeholders for live updates
    risk_placeholder = st.empty()
    transcript_placeholder = st.empty()

    # End Call button
    if st.button("End Call"):
        # End call on backend and fetch report
        response = requests.post(END_CALL_URL, json={"session_id": shared_data["session_id"]})
        if response.status_code == 200:
            report = response.json()
            st.subheader("After-Call Report")
            st.write(f"Call Summary: {report.get('summary', 'No summary available')}")
            
            # Display flagged segments with timestamps
            segments = report.get("segments", [])
            if segments:
                df = pd.DataFrame(segments)
                st.dataframe(df)
            else:
                st.write("No flagged segments.")
            
            # Reset session
            shared_data["session_id"] = None
        else:
            st.error("Failed to end call.")
    else:
        # Real-time update loop while call is active
        while ctx.state.playing:
            with lock:
                score = shared_data["score"]
                label = shared_data["label"]
                rationale = shared_data["rationale"]
                transcript = shared_data["transcript"]

            # Update risk display
            color = {"Safe": "green", "Suspicious": "yellow", "Scam": "red"}.get(label, "gray")
            risk_placeholder.markdown(
                f"<h3 style='color:{color}'>Risk: {label} ({score}%)</h3><p>Rationale: {rationale}</p>",
                unsafe_allow_html=True
            )

            # Update separated transcripts (from diarization)
            transcript_placeholder.markdown(
                f"**User:** {transcript.get('user', '')}<br>**Caller:** {transcript.get('caller', '')}",
                unsafe_allow_html=True
            )

            time.sleep(1)  # Update every second to reduce reruns

elif option == "Upload File":
    st.header("Upload Audio File for Analysis")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        # Send file to backend for processing
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
        response = requests.post(PROCESS_FILE_URL, files=files)
        
        if response.status_code == 200:
            report = response.json()
            st.subheader("Analysis Report")
            st.write(f"Summary: {report.get('summary', 'No summary available')}")
            
            # Display flagged segments with timestamps
            segments = report.get("segments", [])
            if segments:
                df = pd.DataFrame(segments)
                st.dataframe(df)
            else:
                st.write("No flagged segments.")
        else:
            st.error("Failed to process file.")