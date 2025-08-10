# Handles alert endpoints and integrates with anti-spoofing system.

from fastapi import APIRouter, HTTPException
import sys
import os

# Add the speech directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from speech.anti_spoofing.alert_system import AlertSystem
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for development
    pass

router = APIRouter()

# Initialize alert system
alert_system = None

def initialize_alert_system():
    """Initialize the alert system."""
    global alert_system
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'speech', 'anti_spoofing', 'synthetic_voice_detector.pkl')
        if os.path.exists(model_path):
            alert_system = AlertSystem(model_path=model_path)
            print("✅ Alert system initialized with anti-spoofing")
        else:
            alert_system = AlertSystem()
            print("✅ Alert system initialized (basic mode)")
    except Exception as e:
        print(f"❌ Error initializing alert system: {e}")
        alert_system = None

@router.on_event("startup")
async def startup_event():
    """Initialize alert system on startup."""
    initialize_alert_system()

@router.post("/send")
async def send_alert(risk: int, label: str, rationale: str, language: str = "en"):
    """Send an alert with the specified parameters."""
    try:
        if alert_system is None:
            initialize_alert_system()
        
        if alert_system:
            # Create alert using the alert system
            alert = alert_system.create_alert(label.lower(), language, {
                "risk": risk,
                "rationale": rationale
            })
            
            return {
                "status": "Alert sent",
                "details": {
                    "risk": risk,
                    "label": label,
                    "rationale": rationale,
                    "language": language,
                    "message": alert.get("message", ""),
                    "audio_file": alert.get("audio_file", "")
                }
            }
        else:
            return {
                "status": "Alert system not available",
                "details": {"risk": risk, "label": label, "rationale": rationale}
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending alert: {str(e)}")

@router.post("/synthetic-voice")
async def synthetic_voice_alert(audio_file: bytes, language: str = "en"):
    """Generate alert for synthetic voice detection."""
    try:
        if alert_system is None:
            initialize_alert_system()
        
        if alert_system:
            # Save temporary audio file
            temp_file = "temp_synthetic_alert.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_file)
            
            try:
                # Detect synthetic voice
                result = alert_system.detect_synthetic_voice(temp_file)
                
                # Create appropriate alert
                if result.get('is_synthetic', False):
                    alert = alert_system.create_alert('scam', language, result)
                    risk_level = 'scam'
                else:
                    alert = alert_system.create_alert('safe', language, result)
                    risk_level = 'safe'
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                return {
                    "status": "success",
                    "detection": result,
                    "alert": alert,
                    "risk_level": risk_level
                }
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise e
        else:
            raise HTTPException(status_code=500, detail="Alert system not available")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing synthetic voice alert: {str(e)}")

@router.get("/summary")
async def get_summary():
    """Get alert system summary."""
    try:
        if alert_system is None:
            initialize_alert_system()
        
        if alert_system:
            return {
                "status": "Alert system available",
                "components": {
                    "synthetic_voice_detector": alert_system.synthetic_detector is not None,
                    "tts_available": hasattr(alert_system, 'tts_client') and alert_system.tts_client is not None,
                    "supported_languages": ["en", "es", "fr", "ne", "hi", "sa"]
                }
            }
        else:
            return {"status": "Alert system not available"}
            
    except Exception as e:
        return {"status": "Error", "error": str(e)}

@router.get("/health")
async def health_check():
    """Check alert system health."""
    return {
        "status": "healthy" if alert_system is not None else "unavailable",
        "alert_system": alert_system is not None
    }