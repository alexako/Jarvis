#!/usr/bin/env python3
"""
Test script for Piper TTS integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speech_analysis.tts import JarvisTTS, PiperTTS, TTSConfig

def test_piper_direct():
    """Test PiperTTS class directly"""
    print("Testing PiperTTS class directly...")
    
    config = TTSConfig()
    piper = PiperTTS(config)
    
    if piper.available:
        print("✓ Piper TTS is available")
        
        # Test synthesis
        test_text = "Hello, this is a test of Piper TTS integration."
        audio_data = piper.synthesize(test_text)
        
        if audio_data:
            print(f"✓ Synthesis successful - generated {len(audio_data)} bytes of audio")
        else:
            print("✗ Synthesis failed - no audio data generated")
    else:
        print("✗ Piper TTS is not available")
        print("Note: You may need to download voice models from https://github.com/rhasspy/piper/releases")

def test_jarvis_with_piper():
    """Test JarvisTTS with Piper engine"""
    print("\nTesting JarvisTTS with Piper engine...")
    
    try:
        jarvis = JarvisTTS(tts_engine="piper")
        print("✓ JarvisTTS initialized with Piper engine")
        
        if jarvis.tts_engine and jarvis.tts_engine.available:
            print("✓ Piper engine is available in JarvisTTS")
            
            # Test speak method
            test_text = "Good morning, sir. Piper TTS is now operational."
            print(f"Testing speech: '{test_text}'")
            
            # Note: This would actually play audio, so we'll just test the setup
            print("✓ JarvisTTS with Piper is ready for speech synthesis")
        else:
            print("✗ Piper engine is not available in JarvisTTS")
            
    except Exception as e:
        print(f"✗ Error initializing JarvisTTS with Piper: {e}")

def list_available_engines():
    """List all available TTS engines"""
    print("\nTesting all available TTS engines...")
    
    engines = ["pyttsx3", "coqui", "piper", "system"]
    
    for engine in engines:
        try:
            jarvis = JarvisTTS(tts_engine=engine)
            if engine == "system" or (jarvis.tts_engine and jarvis.tts_engine.available):
                print(f"✓ {engine.upper()} engine is available")
            else:
                print(f"✗ {engine.upper()} engine is not available")
        except Exception as e:
            print(f"✗ {engine.upper()} engine failed: {e}")

if __name__ == "__main__":
    print("Piper TTS Integration Test")
    print("=" * 40)
    
    test_piper_direct()
    test_jarvis_with_piper() 
    list_available_engines()
    
    print("\n" + "=" * 40)
    print("Test completed!")