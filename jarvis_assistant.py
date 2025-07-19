#!/usr/bin/env python3

import time
import datetime
import threading
import logging
from speech_analysis import JarvisSTT, JarvisTTS
from commands import JarvisCommands

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JarvisAssistant:
    """Complete Jarvis Voice Assistant with STT and TTS"""
    
    def __init__(self):
        self.stt = JarvisSTT(stt_engine="whisper", model_name="base")
        self.tts = JarvisTTS(tts_engine="system")
        self.is_active = False
        self.is_listening = False
        
        # Initialize centralized command system
        self.commands = JarvisCommands(self.tts, self)
        
        # Set up STT callbacks
        self.stt.set_speech_callback(self.on_speech_received)
        self.stt.set_wake_word_callback(self.on_wake_word_detected)
        
        logger.info("Jarvis Assistant initialized with centralized command system")
    
    def on_wake_word_detected(self):
        """Handle wake word detection"""
        logger.info("🚨 Wake word detected!")
        self.is_active = True
        self.tts.speak_direct("Yes, sir. How may I assist you?")
    
    def on_speech_received(self, text):
        """Handle speech input"""
        logger.info(f"📝 Speech received: '{text}'")
        
        # Check if this is just the wake word (avoid double processing)
        text_lower = text.lower().strip()
        wake_words = ["jarvis", "hey jarvis"]
        is_pure_wake_word = any(text_lower == wake_word for wake_word in wake_words)
        
        if is_pure_wake_word and self.is_active:
            # Already activated by wake word callback, skip processing
            logger.info("Pure wake word detected - already processed by wake word callback")
            return
        
        if self.is_active or "jarvis" in text_lower:
            # Activate if "jarvis" is mentioned but not already active
            if "jarvis" in text_lower and not self.is_active:
                self.is_active = True
                logger.info("Activated via speech text - ready for commands")
                # Don't greet again if this is just a wake word
                if not is_pure_wake_word:
                    self.tts.speak_direct("Yes, sir. How may I assist you?")
                return
            
            # Process the command using centralized system
            self.commands.process_command(text)
        else:
            logger.info("Jarvis is not active - say 'Jarvis' to activate")
    
    def start(self):
        """Start the voice assistant"""
        print("🤖 Starting Jarvis Voice Assistant...")
        print("=" * 60)
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
    
    def stop(self):
        """Stop the voice assistant"""
        self.is_listening = False
        self.stt.stop_listening()
        print("✨ Jarvis Assistant stopped.")

def main():
    """Main entry point"""
    assistant = JarvisAssistant()
    assistant.start()

if __name__ == "__main__":
    main()
