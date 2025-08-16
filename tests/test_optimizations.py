#!/usr/bin/env python3
"""
Test script to verify performance optimizations
"""

import sys
import os
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import_performance():
    """Test that imports are working correctly"""
    print("Testing imports...")
    
    # Test STT import
    start_time = time.time()
    from src.audio.speech_analysis.stt import AudioBuffer, AudioConfig
    stt_import_time = time.time() - start_time
    print(f"STT import time: {stt_import_time:.4f}s")
    
    # Test TTS import
    start_time = time.time()
    from src.audio.speech_analysis.tts import JarvisTTS
    tts_import_time = time.time() - start_time
    print(f"TTS import time: {tts_import_time:.4f}s")
    
    # Test core assistant import
    start_time = time.time()
    from src.core.jarvis_assistant import EnhancedJarvisAssistant
    core_import_time = time.time() - start_time
    print(f"Core assistant import time: {core_import_time:.4f}s")
    
    return stt_import_time + tts_import_time + core_import_time

def test_audio_buffer_performance():
    """Test AudioBuffer performance optimizations"""
    print("\nTesting AudioBuffer performance...")
    
    from src.audio.speech_analysis.stt import AudioBuffer, AudioConfig
    import numpy as np
    
    # Create buffer
    buffer = AudioBuffer(debug=False)
    config = AudioConfig()
    
    # Generate test audio data
    test_chunks = []
    for i in range(100):
        # Create random audio chunk
        chunk = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
        test_chunks.append(chunk)
    
    # Test add_chunk performance
    start_time = time.time()
    for chunk in test_chunks:
        buffer.add_chunk(chunk, config)
    processing_time = time.time() - start_time
    print(f"Processed 100 audio chunks in {processing_time:.4f}s")
    
    return processing_time

def test_command_processing_performance():
    """Test command processing performance optimizations"""
    print("\nTesting command processing performance...")
    
    from src.commands.commands import JarvisCommands
    
    # Mock TTS and assistant
    class MockTTS:
        def speak_direct(self, text):
            pass
    
    class MockAssistant:
        def __init__(self):
            self.is_active = True
            self.is_listening = True
            self.prevent_feedback = False
        
        def speak_with_feedback_control(self, text):
            pass
    
    # Create commands instance
    mock_tts = MockTTS()
    mock_assistant = MockAssistant()
    
    commands = JarvisCommands(mock_tts, mock_assistant)
    
    # Test commands that should be fast
    test_commands = [
        "hello jarvis",
        "what time is it", 
        "jarvis status",
        "tell me a joke",
        "goodbye jarvis"
    ]
    
    start_time = time.time()
    for cmd in test_commands:
        commands.process_command(cmd)
    processing_time = time.time() - start_time
    print(f"Processed {len(test_commands)} commands in {processing_time:.4f}s")
    
    return processing_time

def main():
    """Run all performance tests"""
    print(" Jarvis Performance Optimization Tests")
    print("=" * 50)
    
    total_time = 0
    
    # Run import test
    total_time += test_import_performance()
    
    # Run audio buffer test
    total_time += test_audio_buffer_performance()
    
    # Run command processing test
    total_time += test_command_processing_performance()
    
    print("\n" + "=" * 50)
    print(f"Total test time: {total_time:.4f}s")
    print("All performance tests completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())