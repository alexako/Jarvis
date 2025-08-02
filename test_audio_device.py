#!/usr/bin/env python3
"""Quick test to verify USB audio interface quality"""

import pyaudio
import numpy as np
import time
import wave
import os

def test_audio_device():
    print("🔍 Testing audio devices...")
    
    # List devices
    audio = pyaudio.PyAudio()
    print("\n📡 Available input devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  {i}: {info['name']} - {info['maxInputChannels']} channels, {info['defaultSampleRate']}Hz")
    
    # Test device 3 (Scarlett 2i2)
    device_index = 3
    sample_rate = 16000
    channels = 2
    duration = 3
    
    print(f"\n🎤 Testing device {device_index} for {duration} seconds...")
    print("Speak now!")
    
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data = stream.read(1024)
            frames.append(data)
            
            # Calculate RMS for real-time feedback
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 100:  # Only show when there's audio
                print(f"📈 RMS: {rms:.1f}")
        
        stream.stop_stream()
        stream.close()
        
        # Save to file for analysis
        filename = "audio_test.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"✅ Audio saved to {filename}")
        print("🔍 You can play this back to verify quality")
        
        # Basic analysis
        all_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        max_amplitude = np.max(np.abs(all_audio))
        avg_rms = np.sqrt(np.mean(all_audio**2))
        
        print(f"\n📊 Audio Analysis:")
        print(f"   Max amplitude: {max_amplitude}")
        print(f"   Average RMS: {avg_rms:.1f}")
        print(f"   Dynamic range: {'Good' if max_amplitude > 1000 else 'Low'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing device: {e}")
        return False
    
    finally:
        audio.terminate()

if __name__ == "__main__":
    test_audio_device()