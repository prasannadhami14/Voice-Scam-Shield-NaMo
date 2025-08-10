#!/usr/bin/env python3
"""
Voice Scam Shield System Startup Script
This script installs dependencies and starts the backend and frontend services.
"""

import os
import sys
import subprocess
import time
import requests
import threading
from pathlib import Path

def run_command(command, cwd=None, shell=False):
    """Run a command and return the result."""
    try:
        if shell:
            result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        else:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Command failed: {command}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_backend_health():
    """Check if backend process is responding (do not depend on /health)."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Not in a virtual environment. Consider creating one:")
        print("   python -m venv .venv")
        print("   source .venv/bin/activate  # On macOS/Linux")
        print("   .venv\\Scripts\\activate     # On Windows")
    
    # Install requirements
    if run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]):
        print("✅ Dependencies installed successfully!")
        return True
    else:
        print("❌ Failed to install dependencies")
        return False

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting backend server...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Start backend in a separate thread
    def run_backend():
        # Prefer uvicorn to ensure proper ASGI startup
        run_command([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"], cwd=str(backend_dir), shell=False)
    
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    for i in range(30):  # Wait up to 30 seconds
        if check_backend_health():
            print("✅ Backend server is running!")
            return True
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/30)")
    
    print("❌ Backend failed to start within 30 seconds")
    return False

def start_frontend():
    """Start the Streamlit frontend."""
    print("🎨 Starting frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Start frontend
    if run_command([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"], cwd=str(frontend_dir)):
        print("✅ Frontend started successfully!")
        return True
    else:
        print("❌ Failed to start frontend")
        return False

def main():
    """Main startup function."""
    print("🎭 Voice Scam Shield System Startup")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    if not (current_dir / "backend").exists() or not (current_dir / "frontend").exists():
        print("❌ Please run this script from the Voice-Scam-Shield-NaMo directory")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Cannot continue without dependencies")
        return
    
    # Start backend
    if not start_backend():
        print("❌ Cannot continue without backend")
        return
    
    # Start frontend
    print("\n🎯 System is ready!")
    print("📱 Frontend: http://localhost:8501")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    print("\n🚀 Starting frontend...")
    start_frontend()

if __name__ == "__main__":
    main()
