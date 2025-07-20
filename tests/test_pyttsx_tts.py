#!/usr/bin/env python3
"""
Test script for PyttsxTTS - Debug and fix issues
"""

import sys
import os
import logging

# Add the speech_analysis directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'speech_analysis'))

from tts import PyttsxTTS, TTSConfig

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_pyttsx_initialization():
    """Test PyttsxTTS initialization"""
    print("=" * 50)
    print("Testing PyttsxTTS Initialization")
    print("=" * 50)
    
    try:
        config = TTSConfig()
        tts = PyttsxTTS(config)
        
        if tts.available:
            print("✅ PyttsxTTS initialized successfully")
            print(f"Engine available: {tts.available}")
            
            # Test voice properties
            if hasattr(tts.engine, 'getProperty'):
                voices = tts.engine.getProperty('voices')
                print(f"Available voices: {len(voices) if voices else 0}")
                
                if voices:
                    current_voice = tts.engine.getProperty('voice')
                    rate = tts.engine.getProperty('rate')
                    volume = tts.engine.getProperty('volume')
                    
                    print(f"Current voice ID: {current_voice}")
                    print(f"Speech rate: {rate}")
                    print(f"Volume: {volume}")
                    
                    # Show first few voices
                    print("\nFirst 3 available voices:")
                    for i, voice in enumerate(voices[:3]):
                        print(f"  {i+1}. {voice.name} ({voice.id})")
            
            return tts
        else:
            print("❌ PyttsxTTS failed to initialize")
            return None
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return None

def test_pyttsx_direct_speech(tts):
    """Test direct speech functionality"""
    print("\n" + "=" * 50)
    print("Testing Direct Speech")
    print("=" * 50)
    
    if not tts or not tts.available:
        print("❌ TTS not available for direct speech test")
        return False
    
    test_text = "Hello, this is a test of pyttsx3 direct speech functionality."
    print(f"Testing direct speech with: '{test_text}'")
    
    try:
        tts.speak_directly(test_text)
        print("✅ Direct speech completed successfully")
        return True
    except Exception as e:
        print(f"❌ Direct speech failed: {e}")
        return False

def test_pyttsx_synthesize(tts):
    """Test audio synthesis to bytes"""
    print("\n" + "=" * 50)
    print("Testing Audio Synthesis")
    print("=" * 50)
    
    if not tts or not tts.available:
        print("❌ TTS not available for synthesis test")
        return False, None
    
    test_text = "Testing audio synthesis to bytes."
    print(f"Synthesizing: '{test_text}'")
    
    try:
        audio_data = tts.synthesize(test_text)
        
        if audio_data:
            print(f"✅ Synthesis successful, got {len(audio_data)} bytes of audio data")
            return True, audio_data
        else:
            print("❌ Synthesis returned empty audio data")
            return False, None
            
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return False, None

def test_save_audio_file(audio_data):
    """Test saving audio data to file"""
    print("\n" + "=" * 50)
    print("Testing Audio File Save")
    print("=" * 50)
    
    if not audio_data:
        print("❌ No audio data to save")
        return False
    
    try:
        import wave
        
        output_file = "test_pyttsx_output.wav"
        
        # Create a proper WAV file
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050)  # Sample rate
            wav_file.writeframes(audio_data)
        
        # Check file size
        file_size = os.path.getsize(output_file)
        print(f"✅ Audio saved to '{output_file}' ({file_size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save audio file: {e}")
        return False

def test_voice_properties(tts):
    """Test voice property modifications"""
    print("\n" + "=" * 50)
    print("Testing Voice Property Modifications")
    print("=" * 50)
    
    if not tts or not tts.available:
        print("❌ TTS not available for voice property test")
        return False
    
    try:
        engine = tts.engine
        
        # Test different speech rates
        print("Testing different speech rates...")
        rates_to_test = [120, 180, 240]  # slow, normal, fast
        
        for rate in rates_to_test:
            print(f"Setting rate to {rate} WPM...")
            engine.setProperty('rate', rate)
            actual_rate = engine.getProperty('rate')
            print(f"Actual rate: {actual_rate}")
            
            # Test speech at this rate
            test_text = f"This is speech at {rate} words per minute."
            print(f"Speaking: '{test_text}'")
            tts.speak_directly(test_text)
        
        # Reset to default
        engine.setProperty('rate', 180)
        print("✅ Voice property test completed")
        return True
        
    except Exception as e:
        print(f"❌ Voice property test failed: {e}")
        return False

def main():
    """Main test function"""
    print("PyttsxTTS Comprehensive Test Suite")
    print("=" * 50)
    
    # Test initialization
    tts = test_pyttsx_initialization()
    
    if not tts:
        print("\n❌ Cannot proceed - TTS initialization failed")
        print("\nTroubleshooting steps:")
        print("1. Install pyttsx3: pip install pyttsx3")
        print("2. Check if espeak is installed (Linux): sudo apt-get install espeak espeak-data")
        print("3. On macOS, make sure system TTS is working: say 'hello'")
        return
    
    # Test direct speech
    direct_success = test_pyttsx_direct_speech(tts)
    
    # Test synthesis
    synth_success, audio_data = test_pyttsx_synthesize(tts)
    
    # Test saving audio
    if synth_success and audio_data:
        save_success = test_save_audio_file(audio_data)
    else:
        save_success = False
    
    # Test voice properties
    props_success = test_voice_properties(tts)
    
    # Final summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Initialization: {'✅ PASS' if tts else '❌ FAIL'}")
    print(f"Direct Speech:  {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"Synthesis:      {'✅ PASS' if synth_success else '❌ FAIL'}")
    print(f"File Save:      {'✅ PASS' if save_success else '❌ FAIL'}")
    print(f"Voice Props:    {'✅ PASS' if props_success else '❌ FAIL'}")
    
    if all([tts, direct_success, synth_success, save_success, props_success]):
        print("\n🎉 All tests passed! PyttsxTTS is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
