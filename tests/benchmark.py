#!/usr/bin/env python3
"""
Performance benchmark to compare optimizations
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def benchmark_audio_buffer():
    """Benchmark AudioBuffer performance"""
    print("Benchmarking AudioBuffer performance...")
    
    from src.audio.speech_analysis.stt import AudioBuffer, AudioConfig
    
    # Create buffer and config
    buffer = AudioBuffer(debug=False)
    config = AudioConfig()
    
    # Generate test data
    test_chunks = []
    for i in range(1000):
        chunk = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
        test_chunks.append(chunk)
    
    # Benchmark add_chunk method
    start_time = time.time()
    for chunk in test_chunks:
        buffer.add_chunk(chunk, config)
    end_time = time.time()
    
    processing_time = end_time - start_time
    chunks_per_second = len(test_chunks) / processing_time
    
    print(f"  Processed {len(test_chunks)} chunks in {processing_time:.4f}s")
    print(f"  Performance: {chunks_per_second:.0f} chunks/second")
    
    return processing_time

def benchmark_rms_calculation():
    """Benchmark RMS calculation performance"""
    print("Benchmarking RMS calculation...")
    
    from src.audio.speech_analysis.stt import _import_numpy
    np_module = _import_numpy()
    
    # Generate test data
    test_data = np_module.random.randint(-32768, 32767, 44100, dtype=np_module.int16)  # ~1 second of audio
    
    # Benchmark RMS calculation
    iterations = 1000
    
    start_time = time.time()
    for i in range(iterations):
        # Our optimized RMS calculation
        chunk_float = test_data.astype(np_module.float32)
        mean_square = np_module.mean(np_module.square(chunk_float))
        if np_module.isnan(mean_square) or mean_square <= 0:
            rms = 0.0
        else:
            rms = np_module.sqrt(mean_square)
    end_time = time.time()
    
    processing_time = end_time - start_time
    calculations_per_second = iterations / processing_time
    
    print(f"  Performed {iterations} RMS calculations in {processing_time:.4f}s")
    print(f"  Performance: {calculations_per_second:.0f} calculations/second")
    
    return processing_time

def benchmark_command_matching():
    """Benchmark command matching performance"""
    print("Benchmarking command matching...")
    
    from src.commands.commands import JarvisCommands
    
    # Mock TTS and assistant
    class MockTTS:
        def speak_direct(self, text):
            pass
    
    class MockAssistant:
        def speak_with_feedback_control(self, text):
            pass
    
    # Create commands instance
    mock_tts = MockTTS()
    mock_assistant = MockAssistant()
    commands = JarvisCommands(mock_tts, mock_assistant)
    
    # Test commands
    test_commands = [
        "hello jarvis",
        "what time is it", 
        "jarvis status",
        "tell me a joke",
        "goodbye jarvis",
        "what is the weather like today",
        "can you set a reminder for me",
        "play some music",
        "search the web for python tutorials",
        "send an email to my boss"
    ] * 100  # Repeat 100 times for better benchmarking
    
    start_time = time.time()
    for cmd in test_commands:
        commands.process_command(cmd)
    end_time = time.time()
    
    processing_time = end_time - start_time
    commands_per_second = len(test_commands) / processing_time
    
    print(f"  Processed {len(test_commands)} commands in {processing_time:.4f}s")
    print(f"  Performance: {commands_per_second:.0f} commands/second")
    
    return processing_time

def main():
    """Run all benchmarks"""
    print(" Jarvis Performance Benchmark")
    print("=" * 40)
    
    total_time = 0
    
    # Run benchmarks
    total_time += benchmark_audio_buffer()
    total_time += benchmark_rms_calculation()
    total_time += benchmark_command_matching()
    
    print("\n" + "=" * 40)
    print(f"Total benchmark time: {total_time:.4f}s")
    print("Benchmark completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())