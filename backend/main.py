#!/usr/bin/env python3
"""
Backend API for Voice Scam Shield.
Provides endpoints for audio analysis and scam detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import asyncio
from typing import Dict, Any
import json

# Import our integrated detection system
import sys
# Add backend directory for 'routes' imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Add repo root to sys.path so we can import the 'speech' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from speech.anti_spoofing.integrated_detection import IntegratedDetectionSystem
from routes.audio import router as audio_router
from routes.alert import router as alert_router

app = FastAPI(
    title="Voice Scam Shield API",
    description="API for detecting voice scams using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the detection system
detection_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the detection system on startup."""
    global detection_system
    try:
        detection_system = IntegratedDetectionSystem()
        print("✅ Detection system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detection system: {e}")
        detection_system = None

# Include API routers
app.include_router(audio_router, prefix="/audio", tags=["audio"])
app.include_router(alert_router, prefix="/alert", tags=["alert"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Voice Scam Shield API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if detection_system is None:
        raise HTTPException(status_code=503, detail="Detection system not available")
    
    try:
        status = detection_system.get_system_status()
        return {
            "status": "healthy",
            "detection_system": status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Analyze uploaded audio file for scam detection.
    
    Args:
        file: Audio file to analyze
        language: Expected language code (en, es, fr, hi, ne)
    
    Returns:
        Analysis results with scam detection
    """
    if detection_system is None:
        raise HTTPException(status_code=503, detail="Detection system not available")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        try:
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Analyze the audio
            result = await detection_system.analyze_audio(temp_file.name, language)
            
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported languages."""
    if detection_system is None:
        raise HTTPException(status_code=503, detail="Detection system not available")
    
    try:
        status = detection_system.get_system_status()
        return {
            "languages": status.get("languages_supported", []),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get languages: {str(e)}")

@app.post("/test-audio")
async def test_audio_endpoint():
    """Test endpoint for development."""
    return {
        "message": "Audio test endpoint working",
        "status": "success"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status": "error"
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Voice Scam Shield Backend...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )