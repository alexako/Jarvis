#!/usr/bin/env python3
"""
Test script for voice model configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speech_analysis.tts import JarvisTTS
from config_manager import get_config

def test_voice_configuration():
    """Test voice model configuration options"""
    print("Voice Model Configuration Test")
    print("=" * 40)
    
    config = get_config()
    
    # Show current TTS configuration
    tts_config = config.get_tts_config()
    print(f"Default TTS engine: {tts_config.get('default_engine', 'system')}")
    print(f"Configured Piper voice: {tts_config.get('piper', {}).get('voice_model', 'Not set')}")
    print(f"Voice directory: {tts_config.get('piper', {}).get('voice_directory', 'Not set')}")
    print(f"Jarvis personality enabled: {tts_config.get('jarvis_personality', True)}")
    
    print("\nTesting voice model switching...")
    
    # Test different voice models
    voice_models = ["en_US-lessac-medium", "en_US-ryan-high"]
    
    for voice_model in voice_models:
        print(f"\nTesting voice model: {voice_model}")
        
        # Update config
        config.set('tts.piper.voice_model', voice_model)
        
        # Initialize TTS with updated config
        try:
            jarvis = JarvisTTS(tts_engine="piper")
            if jarvis.tts_engine and jarvis.tts_engine.available:
                print(f"✓ Successfully loaded {voice_model}")
            else:
                print(f"✗ Failed to load {voice_model}")
        except Exception as e:
            print(f"✗ Error with {voice_model}: {e}")
    
    print("\nTesting default engine from config...")
    
    # Test using config default
    try:
        jarvis_default = JarvisTTS()  # Should use config default
        engine_name = "system" if jarvis_default.tts_engine is None else type(jarvis_default.tts_engine).__name__
        print(f"✓ Default engine loaded: {engine_name}")
    except Exception as e:
        print(f"✗ Error loading default engine: {e}")
    
    print("\n" + "=" * 40)
    print("Configuration test completed!")

if __name__ == "__main__":
    test_voice_configuration()