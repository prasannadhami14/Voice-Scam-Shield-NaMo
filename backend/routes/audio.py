# audio.py: Handles audio streaming and processing via WebSocket.

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
import asyncio
import numpy as np
import soundfile as sf
import io
import sys
import os

# Add the speech directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from speech.speech_processing import RealTimeTranscriber
    from speech.config import StreamingConfig
    from speech.anti_spoofing.synthetic_voice_detector import SyntheticVoiceDetector
    from speech.anti_spoofing.alert_system import AlertSystem
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports for development
    pass

router = APIRouter()

# Initialize components
synthetic_detector = None
alert_system = None
transcriber = None

def initialize_components():
    """Initialize speech processing components."""
    global synthetic_detector, alert_system, transcriber
    
    try:
        # Initialize anti-spoofing components
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'speech', 'anti_spoofing', 'synthetic_voice_detector.pkl')
        if os.path.exists(model_path):
            synthetic_detector = SyntheticVoiceDetector(model_path)
            alert_system = AlertSystem(model_path=model_path)
            print("✅ Anti-spoofing components initialized")
        else:
            print("⚠️ Anti-spoofing model not found, using basic detection only")
            alert_system = AlertSystem()
        
        # Initialize transcriber
        config = StreamingConfig()
        transcriber = RealTimeTranscriber(config)
        print("✅ Speech processing components initialized")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        # Continue with basic functionality

@router.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_components()

@router.websocket("/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            try:
                audio_data, sample_rate = sf.read(io.BytesIO(audio_chunk))
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Convert to mono
            except Exception as e:
                print(f"Error reading audio: {e}")
                continue
            
            # Process audio chunk
            result = await process_audio_chunk(audio_data, sample_rate)
            # Include partial transcription text if present for smoother UI
            try:
                trans = result.get("transcription", {}) if isinstance(result, dict) else {}
                if trans and trans.get("text"):
                    # truncate to last 160 chars to reduce payload size
                    trans["text"] = trans.get("text", "")[-160:]
                    result["transcription"] = trans
            except Exception:
                pass
            
            # Send result back
            await websocket.send_json(result)
            
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in audio stream: {e}")
        await websocket.close()

async def process_audio_chunk(audio_data: np.ndarray, sample_rate: int) -> dict:
    """Process an audio chunk and return detection results."""
    try:
        # Initialize components if not already done
        if transcriber is None:
            initialize_components()
        
        # Basic scam detection using transcriber
        if transcriber:
            # Prefer async if available
            try:
                result = await transcriber.process_audio_chunk_async(audio_data)
            except Exception:
                result = transcriber.process_audio_chunk(audio_data)
        else:
            result = {"risk": 0, "label": "Safe", "rationale": "System not initialized"}
        
        # Anti-spoofing detection
        if synthetic_detector and alert_system:
            try:
                # Save temporary audio file for anti-spoofing
                temp_file = "temp_audio_chunk.wav"
                sf.write(temp_file, audio_data, sample_rate)
                
                # Detect synthetic voice
                synthetic_result = synthetic_detector.predict(temp_file)
                is_synthetic = synthetic_result.get('is_synthetic', False)
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Update result based on anti-spoofing
                if is_synthetic:
                    result["label"] = "Scam"
                    result["risk"] = max(result.get("risk", 0), 90)
                    result["rationale"] = f"{result.get('rationale', '')} (Synthetic voice detected)"
                    result["is_synthetic"] = True
                
            except Exception as e:
                print(f"Anti-spoofing error: {e}")
                result["anti_spoofing_error"] = str(e)
        
        return result
        
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return {
            "risk": 0,
            "label": "Error",
            "rationale": f"Processing error: {str(e)}"
        }

@router.post("/process")
async def process_audio_file(audio_file: bytes):
    """Process an uploaded audio file."""
    try:
        # Convert bytes to numpy array
        audio_data, sample_rate = sf.read(io.BytesIO(audio_file))
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Process the audio
        result = await process_audio_chunk(audio_data, sample_rate)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.post("/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    """Process an uploaded audio file via multipart form."""
    try:
        # Read the uploaded file
        file_content = await file.read()
        
        # Convert to numpy array
        audio_data, sample_rate = sf.read(io.BytesIO(file_content))
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Process the audio
        result = await process_audio_chunk(audio_data, sample_rate)
        
        return {
            "status": "success",
            "filename": file.filename,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

@router.get("/health")
async def health_check():
    """Check audio processing system health."""
    return {
        "status": "healthy",
        "components": {
            "transcriber": transcriber is not None,
            "synthetic_detector": synthetic_detector is not None,
            "alert_system": alert_system is not None
        }
    }