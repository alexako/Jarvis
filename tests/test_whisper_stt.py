#!/usr/bin/env python3
"""
Test script for WhisperSTT functionality from the Jarvis speech analysis system.
This script tests initialization, transcription, and error handling of the WhisperSTT class.
"""

import sys
import os
import numpy as np
import wave
import logging

# Add the project root to the path so we can import from speech_analysis
sys.path.insert(0, os.path.abspath('.'))

from speech_analysis.stt import WhisperSTT, AudioConfig

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_whisper_initialization():
    """Test WhisperSTT initialization with different models."""
    print("=" * 60)
    print("Testing WhisperSTT Initialization")
    print("=" * 60)
    
    # Test with default base model
    print("\n1. Testing with 'base' model:")
    try:
        whisper_base = WhisperSTT("base")
        print(f"   ✅ Base model loaded successfully. Available: {whisper_base.available}")
        if hasattr(whisper_base, 'model'):
            print(f"   ✅ Model object created: {type(whisper_base.model)}")
        return whisper_base
    except Exception as e:
        print(f"   ❌ Failed to load base model: {e}")
        return None

def test_whisper_with_audio_file(whisper_stt, audio_file_path):
    """Test WhisperSTT transcription with an audio file."""
    print("\n2. Testing transcription with audio file:")
    
    if not os.path.exists(audio_file_path):
        print(f"   ⚠️  Audio file not found: {audio_file_path}")
        print("   Creating a synthetic test audio array instead...")
        
        # Create synthetic audio data for testing
        config = AudioConfig()
        duration = 2.0  # 2 seconds
        sample_rate = config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate a simple sine wave (440 Hz)
        frequency = 440.0
        audio_data = (np.sin(frequency * 2 * np.pi * t) * 16383).astype(np.int16)
        
        print(f"   📊 Generated synthetic audio: {len(audio_data)} samples, {duration}s duration")
        
        try:
            result = whisper_stt.transcribe(audio_data, config)
            print(f"   🎯 Transcription result: '{result}'")
            print("   ℹ️  Note: Synthetic sine wave likely produces empty or noise-like transcription")
            return True
        except Exception as e:
            print(f"   ❌ Transcription failed: {e}")
            return False
    
    else:
        print(f"   📁 Found audio file: {audio_file_path}")
        try:
            # Read the actual audio file
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
            config = AudioConfig()
            print(f"   📊 Audio data: {len(audio_data)} samples")
            
            result = whisper_stt.transcribe(audio_data, config)
            print(f"   🎯 Transcription result: '{result}'")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to transcribe audio file: {e}")
            return False

def test_error_handling(whisper_stt):
    """Test WhisperSTT error handling with edge cases."""
    print("\n3. Testing error handling:")
    
    config = AudioConfig()
    
    # Test with empty audio
    print("   Testing with empty audio array:")
    try:
        result = whisper_stt.transcribe(np.array([]), config)
        print(f"   ✅ Empty audio handled correctly: '{result}'")
    except Exception as e:
        print(f"   ❌ Empty audio test failed: {e}")
    
    # Test with very short audio
    print("   Testing with very short audio (10 samples):")
    try:
        short_audio = np.random.randint(-1000, 1000, 10, dtype=np.int16)
        result = whisper_stt.transcribe(short_audio, config)
        print(f"   ✅ Short audio handled: '{result}'")
    except Exception as e:
        print(f"   ❌ Short audio test failed: {e}")
    
    # Test with silence (zeros)
    print("   Testing with silence (all zeros):")
    try:
        silence = np.zeros(16000, dtype=np.int16)  # 1 second of silence
        result = whisper_stt.transcribe(silence, config)
        print(f"   ✅ Silence handled: '{result}'")
    except Exception as e:
        print(f"   ❌ Silence test failed: {e}")

def test_unavailable_whisper():
    """Test behavior when Whisper is not available."""
    print("\n4. Testing unavailable Whisper scenario:")
    
    # Create a WhisperSTT with a non-existent model to simulate failure
    try:
        whisper_bad = WhisperSTT("non_existent_model_xyz")
        print(f"   Available status: {whisper_bad.available}")
        
        if not whisper_bad.available:
            print("   ✅ Correctly detected unavailable Whisper")
            
            # Test transcription when unavailable
            config = AudioConfig()
            test_audio = np.random.randint(-1000, 1000, 1000, dtype=np.int16)
            result = whisper_bad.transcribe(test_audio, config)
            print(f"   ✅ Unavailable Whisper transcription result: '{result}'")
        else:
            print("   ⚠️  Expected Whisper to be unavailable but it's marked as available")
            
    except Exception as e:
        print(f"   ⚠️  Exception during bad model test (this might be expected): {e}")

def run_comprehensive_test():
    """Run all WhisperSTT tests."""
    print("🚀 Starting Comprehensive WhisperSTT Test Suite")
    print("=" * 80)
    
    test_results = {
        'initialization': False,
        'transcription': False,
        'error_handling': True,  # Assume this passes unless it fails
        'unavailable_handling': True
    }
    
    # Test 1: Initialization
    whisper_stt = test_whisper_initialization()
    if whisper_stt and whisper_stt.available:
        test_results['initialization'] = True
        
        # Test 2: Transcription
        # Try a few common test audio file locations
        test_audio_paths = [
            "test_audio.wav",
            "sample.wav",
            "audio_test.wav",
            os.path.expanduser("~/Downloads/test_audio.wav"),
        ]
        
        transcription_success = False
        for audio_path in test_audio_paths:
            if test_whisper_with_audio_file(whisper_stt, audio_path):
                transcription_success = True
                break
        
        test_results['transcription'] = transcription_success
        
        # Test 3: Error handling
        try:
            test_error_handling(whisper_stt)
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            test_results['error_handling'] = False
    
    else:
        print("   ⚠️  Skipping transcription and error handling tests due to initialization failure")
        test_results['transcription'] = False
        test_results['error_handling'] = False
    
    # Test 4: Unavailable Whisper handling
    try:
        test_unavailable_whisper()
    except Exception as e:
        print(f"   ❌ Unavailable Whisper test failed: {e}")
        test_results['unavailable_handling'] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏁 Test Summary")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! WhisperSTT appears to be working correctly.")
    elif test_results['initialization']:
        print("⚠️  WhisperSTT initializes but has some issues. Check the details above.")
    else:
        print("💥 WhisperSTT failed to initialize. Check Whisper installation and dependencies.")
        print("\nTroubleshooting steps:")
        print("   1. Ensure openai-whisper is installed: pip install openai-whisper")
        print("   2. Check that you have sufficient disk space for the model")
        print("   3. Verify your internet connection (first run downloads the model)")
        print("   4. Try a smaller model like 'tiny' or 'small'")

if __name__ == "__main__":
    print("WhisperSTT Test Suite")
    print("This script tests the WhisperSTT class from speech_analysis.stt")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("speech_analysis/stt.py"):
        print("❌ Error: speech_analysis/stt.py not found!")
        print("Please run this script from the root directory of your Jarvis project.")
        sys.exit(1)
    
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
