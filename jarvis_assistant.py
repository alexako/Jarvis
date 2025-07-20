#!/usr/bin/env python3

import time
import datetime
import threading
import logging
from speech_analysis import JarvisSTT, JarvisTTS
from commands import JarvisCommands, create_ai_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JarvisAssistant:
    """Complete Jarvis Voice Assistant with STT and TTS"""

    def __init__(self, ai_enabled=False, prevent_feedback=False, performance_mode=None):
        self.performance_mode = performance_mode
        self.stt = JarvisSTT(stt_engine="whisper", model_name="base", performance_mode=performance_mode)
        self.tts = JarvisTTS(tts_engine="system")
        self.is_active = False
        self.is_listening = False
        self.ai_enabled = ai_enabled
        self.prevent_feedback = prevent_feedback
        self.is_speaking = False  # Track when TTS is active
        
        # Create AI configuration
        ai_config = create_ai_config(
            anthropic_enabled=self.ai_enabled,     # Use Claude
            openai_enabled=self.ai_enabled,        # OpenAI as fallback
            prefer_anthropic=True                  # Claude first, OpenAI fallback
        )
        
        # Initialize centralized command system
        self.commands = JarvisCommands(self.tts, self)
        
        # Set up STT callbacks - only use speech callback to avoid double processing
        self.stt.set_speech_callback(self.on_speech_received)
        # self.stt.set_wake_word_callback(self.on_wake_word_detected)  # Disabled to prevent double greeting
        
        logger.info(f"Jarvis Assistant initialized with centralized command system, performance mode: {performance_mode or 'default'}")
    
    def on_wake_word_detected(self):
        """Handle wake word detection"""
        logger.info("🚨 Wake word detected!")
        self.is_active = True
        self.tts.speak_direct("Yes, sir. How may I assist you?")
    
    def on_speech_received(self, text):
        """Handle speech input and wake word detection"""
        logger.info(f"📝 Speech received: '{text}'")
        
        # Filter out empty, very short, or meaningless transcriptions
        text_clean = text.strip()

        if not text_clean or len(text_clean) < 2 or text_clean in [".", "..", "..."]:
            logger.info(f"Ignoring very short transcription: '{text_clean}'")
            return
        
        text_lower = text_clean.lower()
        wake_words = ["jarvis", "hey jarvis"]
        
        # Check for wake word in the transcribed text
        contains_wake_word = any(wake_word in text_lower for wake_word in wake_words)
        is_pure_wake_word = any(text_lower == wake_word for wake_word in wake_words)
        
        # Handle wake word activation
        if contains_wake_word and not self.is_active:
            logger.info("🚨 Wake word detected in speech - activating!")
            self.is_active = True
            self.speak_without_feedback("Yes, sir. How may I assist you?")
            
            # After activation, don't process the wake word text as a command
            logger.info("Wake word activation complete - skipping command processing for this utterance")
            return
        
        # Process commands only if active
        if self.is_active:
            self.commands.process_command(text)
        else:
            logger.info("Jarvis is not active - say 'Jarvis' to activate")
    
    def start(self):
        """Start the voice assistant"""
        print("🤖 Starting Jarvis Voice Assistant...")
        print("=" * 60)
        if self.performance_mode:
            print(f"⚡ Performance Mode: {self.performance_mode}")
        print("🎙️  Say 'Jarvis' to activate")
        print("📢 Available commands:")
        print("   • Time & Date: 'time', 'date', 'what time is it'")
        print("   • Greetings: 'hello', 'hi', 'good morning/afternoon/evening'")
        print("   • Status: 'how are you', 'status', 'system status'")
        print("   • System Info: 'battery', 'memory', 'disk space'")
        print("   • Entertainment: 'joke', 'tell me a joke'")
        print("   • Help: 'help', 'what can you do', 'commands'")
        print("   • Control: 'stop listening', 'shutdown', 'goodbye'")
        print("   • Identity: 'who are you', 'introduce yourself'")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start with greeting
        self.tts.speak_direct("Good day, sir. Jarvis voice assistant is now online. Say my name to begin.")
        
        # Start listening
        self.is_listening = True
        self.stt.start_listening()
        
        try:
            while self.is_listening:
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
        finally:
            self.stop()
    
    def speak_without_feedback(self, text):
        """Speak text with optional feedback prevention"""
        if self.prevent_feedback and self.is_listening:
            # Temporarily pause listening to prevent feedback
            logger.info("🔇 Pausing listening during speech")
            self.stt.stop_listening()
            self.is_speaking = True
            
            # Speak the text
            self.tts.speak_direct(text)
            
            # Resume listening after a short delay
            time.sleep(0.5)  # Give time for audio to finish
            if self.is_listening:  # Only resume if we were originally listening
                logger.info("🎤 Resuming listening after speech")
                self.stt.start_listening()
            self.is_speaking = False
        else:
            # Normal speech without feedback prevention
            self.tts.speak_direct(text)
    
    def stop(self):
        """Stop the voice assistant"""
        self.is_listening = False
        self.stt.stop_listening()
        print("✨ Jarvis Assistant stopped.")

def main():
    """Main entry point"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Jarvis Voice Assistant')
    parser.add_argument('--prevent-feedback', action='store_true', 
                       help='Enable feedback prevention during speech')
    parser.add_argument('--fast', action='store_true', 
                       help='Use fast performance mode (smaller models, faster response)')
    parser.add_argument('--balanced', action='store_true', 
                       help='Use balanced performance mode (default settings)')
    parser.add_argument('--accurate', action='store_true', 
                       help='Use accurate performance mode (larger models, better quality)')
    parser.add_argument('--enable-ai', action='store_true', 
                       help='Enable AI features (Claude and OpenAI) for advanced interactions')
    
    args = parser.parse_args()
    
    # Determine performance mode
    performance_mode = None
    mode_count = sum([args.fast, args.balanced, args.accurate])
    
    if mode_count > 1:
        print("Error: Only one performance mode can be specified")
        sys.exit(1)
    elif args.fast:
        performance_mode = "fast"
    elif args.balanced:
        performance_mode = "balanced"
    elif args.accurate:
        performance_mode = "accurate"

    # Display settings
    if args.prevent_feedback:
        print("🔇 Feedback prevention enabled")
    if performance_mode:
        print(f"⚡ Performance mode: {performance_mode}")
    if args.enable_ai:
        print("🤖 AI features enabled (Claude and OpenAI)")
    
    assistant = JarvisAssistant(prevent_feedback=args.prevent_feedback, 
                                ai_enabled=args.enable_ai,
                               performance_mode=performance_mode)
    assistant.start()

if __name__ == "__main__":
    main()
