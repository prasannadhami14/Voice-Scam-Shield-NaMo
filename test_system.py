#!/usr/bin/env python3
"""
Test script to verify Voice Scam Shield system components.
Run this after starting the backend to check system health.
"""

import requests
import json
import time
import sys
import os

# Add the speech directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'speech'))

def test_backend_health():
    """Test if the backend is running and healthy."""
    print("🔍 Testing backend health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is running: {data}")
            return True
        else:
            print(f"❌ Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print("   Start with: cd backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_audio_system():
    """Test the audio processing system."""
    print("\n🔍 Testing audio system...")
    
    try:
        response = requests.get("http://localhost:8000/audio/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Audio system: {data}")
            return True
        else:
            print(f"❌ Audio system responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing audio system: {e}")
        return False

def test_alert_system():
    """Test the alert system."""
    print("\n🔍 Testing alert system...")
    
    try:
        response = requests.get("http://localhost:8000/alert/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Alert system: {data}")
            return True
        else:
            print(f"❌ Alert system responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing alert system: {e}")
        return False

def test_api_endpoints():
    """Test basic API endpoints."""
    print("\n🔍 Testing API endpoints...")
    
    endpoints = [
        ("GET /", "http://localhost:8000/"),
        ("POST /start_call", "http://localhost:8000/start_call"),
    ]
    
    for name, url in endpoints:
        try:
            if name.startswith("GET"):
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, timeout=10)
            
            if response.status_code in [200, 201]:
                print(f"✅ {name}: Working")
            else:
                print(f"⚠️  {name}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")

def test_speech_components():
    """Test speech processing components."""
    print("\n🔍 Testing speech components...")
    
    try:
        # Test anti-spoofing model
        model_path = os.path.join("speech", "anti_spoofing", "synthetic_voice_detector.pkl")
        if os.path.exists(model_path):
            print(f"✅ Anti-spoofing model: Found ({os.path.getsize(model_path)} bytes)")
        else:
            print(f"❌ Anti-spoofing model: Not found")
        
        # Test speech processing
        try:
            from speech.config import StreamingConfig
            config = StreamingConfig()
            print(f"✅ Speech config: Loaded (sample_rate={config.sample_rate})")
        except ImportError as e:
            print(f"⚠️  Speech config: Import error - {e}")
        
        # Test keywords
        keywords_dir = os.path.join("speech", "keywords")
        if os.path.exists(keywords_dir):
            keyword_files = [f for f in os.listdir(keywords_dir) if f.endswith('.json')]
            print(f"✅ Keywords: {len(keyword_files)} language files found")
        else:
            print(f"❌ Keywords directory: Not found")
            
    except Exception as e:
        print(f"❌ Error testing speech components: {e}")

def test_file_upload():
    """Test file upload functionality."""
    print("\n🔍 Testing file upload...")
    
    try:
        # Create a simple test audio file (silence)
        import numpy as np
        import soundfile as sf
        
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples)
        
        # Save as temporary WAV file
        test_file = "test_audio.wav"
        sf.write(test_file, audio_data, sample_rate)
        
        # Test upload
        with open(test_file, "rb") as f:
            files = {"file": ("test_audio.wav", f, "audio/wav")}
            response = requests.post("http://localhost:8000/process_file", files=files, timeout=30)
        
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ File upload: Working - {data.get('summary', 'Processed')}")
            return True
        else:
            print(f"❌ File upload: Status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing file upload: {e}")
        return False

def main():
    """Run all tests."""
    print("🎭 Voice Scam Shield System Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        print("❌ Please run this script from the Voice-Scam-Shield-NaMo directory")
        return
    
    # Run tests
    tests = [
        ("Backend Health", test_backend_health),
        ("Audio System", test_audio_system),
        ("Alert System", test_alert_system),
        ("API Endpoints", test_api_endpoints),
        ("Speech Components", test_speech_components),
        ("File Upload", test_file_upload),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Frontend: http://localhost:8501")
        print("   2. Backend API: http://localhost:8000")
        print("   3. API Docs: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure backend is running: cd backend && python main.py")
        print("   2. Check all dependencies are installed: pip install -r requirements.txt")
        print("   3. Verify model files exist in speech/anti_spoofing/")

if __name__ == "__main__":
    main()
