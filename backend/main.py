# Entry point for the FastAPI application
# This file sets up the FastAPI app, includes routes, and handles CORS for frontend integration.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.audio import router as audio_router  # Import audio routes
from routes.alert import router as alert_router  # Import alert routes

# Initialize FastAPI app
app = FastAPI(
    title="Voice Scam Shield Backend",
    description="API for real-time scam detection in calls",
    version="1.0.0"
)

# Add CORS middleware to allow frontend (Streamlit) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio_router, prefix="/audio", tags=["audio"])
app.include_router(alert_router, prefix="/alert", tags=["alert"])

@app.get("/health")
async def health_check():
    """Check if API is running."""
    return {"status": "API is running"}