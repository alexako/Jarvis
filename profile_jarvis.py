#!/usr/bin/env python3
"""
Performance profiling script for Jarvis
"""

import sys
import os
import cProfile
import pstats
import io

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def profile_imports():
    """Profile the import time of key modules"""
    import time
    
    start_time = time.time()
    from core.jarvis_assistant import EnhancedJarvisAssistant
    assistant_time = time.time() - start_time
    
    start_time = time.time()
    from ai.ai_brain import create_ai_brain
    ai_time = time.time() - start_time
    
    start_time = time.time()
    from commands.commands import JarvisCommands
    commands_time = time.time() - start_time
    
    start_time = time.time()
    from context.jarvis_context import create_jarvis_context
    context_time = time.time() - start_time
    
    print(f"Import times:")
    print(f"  Assistant: {assistant_time:.4f}s")
    print(f"  AI Brain: {ai_time:.4f}s")
    print(f"  Commands: {commands_time:.4f}s")
    print(f"  Context: {context_time:.4f}s")

def profile_initialization():
    """Profile the initialization time of key components"""
    import time
    
    # Profile assistant initialization
    start_time = time.time()
    from core.jarvis_assistant import EnhancedJarvisAssistant
    assistant = EnhancedJarvisAssistant(
        ai_enabled=False,  # Disable AI for faster initialization
        prevent_feedback=True,
        performance_mode="fast",  # Use fast performance mode
        ai_provider_preference="anthropic",
        enable_local_llm=False,  # Disable local LLM for faster initialization
        tts_engine="pyttsx3",  # Use lighter TTS engine
        enable_speaker_id=False  # Disable speaker ID for faster initialization
    )
    init_time = time.time() - start_time
    
    print(f"Assistant initialization time: {init_time:.4f}s")
    
    return assistant

def profile_command_processing(assistant):
    """Profile command processing performance"""
    import time
    
    # Test a few common commands
    test_commands = [
        "hello jarvis",
        "what time is it",
        "how are you",
        "tell me a joke"
    ]
    
    total_time = 0
    for command in test_commands:
        start_time = time.time()
        assistant.commands.process_command(command)
        command_time = time.time() - start_time
        total_time += command_time
        print(f"  '{command}': {command_time:.4f}s")
    
    print(f"Average command processing time: {total_time/len(test_commands):.4f}s")

def profile_ai_brain():
    """Profile AI brain initialization"""
    import time
    
    start_time = time.time()
    from ai.ai_brain import create_ai_brain
    
    # Create a minimal AI config for testing
    ai_config = {
        "providers": {
            "anthropic": {
                "enabled": False,  # Disable for faster testing
                "model": "claude-3-haiku-20240307",
                "priority": 1
            },
            "deepseek": {
                "enabled": False,  # Disable for faster testing
                "model": "deepseek-chat",
                "priority": 2
            },
            "local": {
                "enabled": False,  # Disable for faster testing
                "priority": 3
            }
        },
        "fallback_enabled": True,
        "health_check_interval": 300
    }
    
    brain_manager = create_ai_brain(ai_config)
    init_time = time.time() - start_time
    
    print(f"AI Brain initialization time: {init_time:.4f}s")

def main():
    """Main profiling function"""
    print("Jarvis Performance Profiling")
    print("=" * 40)
    
    # Profile imports
    print("\n1. Module Import Performance:")
    profile_imports()
    
    # Profile initialization
    print("\n2. Component Initialization Performance:")
    assistant = profile_initialization()
    
    # Profile AI brain
    print("\n3. AI Brain Performance:")
    profile_ai_brain()
    
    # Profile command processing
    print("\n4. Command Processing Performance:")
    profile_command_processing(assistant)
    
    print("\nProfiling complete!")

if __name__ == "__main__":
    main()