# audio.py: Handles audio streaming and processing via WebSocket.

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from .. ..speech.transcribe import transcribe_audio  # Adjust path as needed
from .. ..speech.scam_detection import detect_scam
from .. ..alerts.anti_spoofing import detect_synthetic
from .. ..alerts.tts_alerts import generate_tts_alert

router = APIRouter()

@router.websocket("/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            transcription = transcribe_audio(audio_chunk, language="en")
            scam_result = detect_scam(transcription)
            risk_score = scam_result["risk"]
            label = scam_result["label"]
            rationale = scam_result["rationale"]
            is_synthetic = detect_synthetic(audio_chunk)
            if is_synthetic:
                label = "Scam"
                risk_score = max(risk_score, 90)
                rationale += " (Synthetic voice detected)"
            if risk_score >= 50:
                generate_tts_alert(label, rationale, language="en")  # Generate audio alert
                alert_message = {"type": "alert", "message": f"Warning: {label} - {rationale}"}
            else:
                alert_message = {"type": "status", "message": "Safe"}
            await websocket.send_json({
                "risk": risk_score,
                "label": label,
                "rationale": rationale,
                "alert": alert_message
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()