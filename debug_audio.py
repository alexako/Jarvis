#!/usr/bin/env python3
"""
Debug Audio Script - Diagnose speech detection issues
"""

import time
import numpy as np
from speech_analysis.stt import JarvisSTT, AudioBuffer, AudioConfig

def debug_audio_levels():
    """Debug audio levels and speech detection"""
    print("🔍 Audio Debug Mode")
    print("=" * 50)
    print("This will show RMS values and speech detection in real-time")
    print("Speak and then be quiet to test utterance completion")
    print("Press Ctrl+C to stop")
    print()
    
    # Create STT with debug enabled
    stt = JarvisSTT(debug=True)
    config = AudioConfig()
    
    print(f"Configuration:")
    print(f"  - Sample Rate: {config.sample_rate}")
    print(f"  - Chunk Size: {config.chunk_size}")
    print(f"  - Silence Threshold: {config.silence_threshold}")
    print(f"  - Silence Duration: {config.silence_duration}s")
    print(f"  - Max Recording Time: {config.max_recording_time}s")
    print()
    
    # Create temporary audio buffer with debug enabled
    temp_buffer = AudioBuffer(debug=True)
    
    try:
        # Open audio stream
        stream = stt.audio.open(
            format=config.format,
            channels=config.channels,
            rate=config.sample_rate,
            input=True,
            frames_per_buffer=config.chunk_size
        )
        
        print("🎤 Listening for audio... (30 second test)")
        start_time = time.time()
        chunk_count = 0
        
        while time.time() - start_time < 30:
            try:
                # Read audio chunk
                data = stream.read(config.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                chunk_count += 1
                
                # Calculate RMS for this chunk
                if len(audio_chunk) > 0:
                    rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
                else:
                    rms = 0.0
                
                # Check with buffer
                utterance_complete = temp_buffer.add_chunk(audio_chunk, config)
                
                # Print status every 10 chunks (roughly every 0.6 seconds)
                if chunk_count % 10 == 0:
                    print(f"Chunk {chunk_count:3d}: RMS={rms:6.1f}, Recording={temp_buffer.is_recording}, Speech={temp_buffer.speech_detected}, Silence={temp_buffer.silence_counter}")
                
                if utterance_complete:
                    audio_data = temp_buffer.get_audio_data()
                    print(f"\n🎉 UTTERANCE COMPLETE! Audio data length: {len(audio_data)}")
                    
                    # Try transcription
                    if len(audio_data) > 0:
                        print("🔄 Transcribing...")
                        transcription = stt.stt_engine.transcribe(audio_data, config)
                        print(f"📝 Result: '{transcription}'")
                    print()
                
            except Exception as e:
                print(f"Error reading audio: {e}")
                continue
                
            time.sleep(0.01)  # Small delay
        
        # Clean up
        stream.stop_stream()
        stream.close()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping debug...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("✅ Audio debug completed!")

if __name__ == "__main__":
    debug_audio_levels()