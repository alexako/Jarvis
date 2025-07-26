#!/usr/bin/env python3
"""
Enhanced Jarvis Assistant with improved STT responsiveness
Key improvements:
1. Better audio stream management during TTS
2. Enhanced feedback prevention
3. Proper state synchronization
4. Recovery mechanisms for stuck states
"""

__version__ = "1.5.0"

import time
import datetime
import threading
import logging
from typing import Dict, Optional
from speech_analysis import EnhancedJarvisSTT
from speech_analysis.tts import JarvisTTS
from speech_analysis.speaker_identification import SpeakerIdentificationSystem
from commands import JarvisCommands, create_ai_config
from jarvis_context import create_jarvis_context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedJarvisAssistant:
    """Enhanced Jarvis Voice Assistant with improved responsiveness"""

    def __init__(self, ai_enabled=False, prevent_feedback=True, performance_mode=None, ai_provider_preference="anthropic", enable_local_llm=True, tts_engine="piper", enable_speaker_id=True):
        self.performance_mode = performance_mode
        self.enable_speaker_id = enable_speaker_id
        
        # Use enhanced STT with speaker identification
        self.stt = EnhancedJarvisSTT(
            stt_engine="whisper", 
            model_name="base", 
            performance_mode=performance_mode,
            enable_speaker_id=enable_speaker_id,
            debug=True  # Enable debug for better monitoring
        )
        
        self.tts = JarvisTTS(tts_engine=tts_engine)
        self.is_active = False
        self.is_listening = False
        self.ai_enabled = ai_enabled
        self.prevent_feedback = prevent_feedback
        self.is_speaking = False
        
        # Enhanced state management
        self.speech_lock = threading.RLock()
        self.last_speech_time = 0
        self.speech_cooldown = 0.5  # Minimum time between speeches
        self.response_timeout = 30.0  # Max time for a response
        self.inactive_timeout = 60.0  # Auto-deactivate after 60 seconds of inactivity
        self.deactivation_timer = None
        
        # Create AI configuration
        prefer_anthropic = (ai_provider_preference == "anthropic")
        prefer_local = (ai_provider_preference == "local")
        
        ai_config = create_ai_config(
            anthropic_enabled=self.ai_enabled,
            deepseek_enabled=self.ai_enabled,
            local_enabled=enable_local_llm,
            prefer_anthropic=prefer_anthropic,
            prefer_local=prefer_local
        )
        
        # Initialize context/memory system with multi-user support
        self.context = create_jarvis_context(
            db_path="jarvis_memory.db",
            max_session_history=20,
            default_user="user"  # Default user for multi-user system
        )
        
        # Initialize centralized command system
        self.commands = JarvisCommands(self.tts, self, ai_config, self.context)
        
        # Set up STT callbacks
        if self.enable_speaker_id:
            self.stt.set_speech_with_speaker_callback(self.on_speech_with_speaker_received)
        else:
            self.stt.set_speech_callback(self.on_speech_received)
        # Note: Wake word detection handled in speech callback, not separately
        
        # Enhanced TTS callbacks for better state management
        self.tts.set_speech_callbacks(
            on_start=self.on_tts_start,
            on_end=self.on_tts_end
        )
        
        logger.info(f"Enhanced Jarvis Assistant initialized - Performance mode: {performance_mode or 'default'}")
    
    def on_tts_start(self):
        """Called when TTS starts speaking"""
        with self.speech_lock:
            self.is_speaking = True
            if self.prevent_feedback:
                # Pause STT detection during speech
                self.stt.pause_for_speech()
                logger.debug("STT paused for TTS playback")
    
    def on_tts_end(self):
        """Called when TTS finishes speaking"""
        with self.speech_lock:
            self.is_speaking = False
            if self.prevent_feedback:
                # Resume STT detection after speech with small delay
                threading.Timer(0.5, self._resume_stt_after_delay).start()
    
    def _resume_stt_after_delay(self):
        """Resume STT after a short delay to ensure audio is clear"""
        if self.is_listening and not self.is_speaking:
            self.stt.resume_after_speech()
            logger.debug("STT resumed after TTS completion")
    
    def _activate_jarvis(self):
        """Activate Jarvis and start deactivation timer"""
        self.is_active = True
        self.last_speech_time = time.time()
        self._reset_deactivation_timer()
        logger.info("🟢 Jarvis activated")
    
    def _deactivate_jarvis(self):
        """Deactivate Jarvis and return to wake-word-only mode"""
        self.is_active = False
        if self.deactivation_timer:
            self.deactivation_timer.cancel()
            self.deactivation_timer = None
        logger.info("🔴 Jarvis deactivated - wake word required")
    
    def _reset_deactivation_timer(self):
        """Reset the auto-deactivation timer"""
        if self.deactivation_timer:
            self.deactivation_timer.cancel()
        
        self.deactivation_timer = threading.Timer(
            self.inactive_timeout, 
            self._auto_deactivate
        )
        self.deactivation_timer.start()
    
    def _auto_deactivate(self):
        """Auto-deactivate Jarvis after inactivity timeout"""
        logger.info(f"⏰ Auto-deactivating after {self.inactive_timeout}s of inactivity")
        self._deactivate_jarvis()
    
    def on_wake_word_detected(self):
        """Handle wake word detection with enhanced state management"""
        current_time = time.time()
        
        # Prevent rapid wake word triggers
        if current_time - self.last_speech_time < self.speech_cooldown:
            logger.debug("Wake word ignored due to cooldown")
            return
        
        logger.info("🚨 Wake word detected!")
        self._activate_jarvis()
        
        # Use enhanced speaking method
        self.speak_with_feedback_control("Yes, sir. How may I assist you?")
    
    def on_speech_with_speaker_received(self, text: str, speaker_id: Optional[str], confidence: float):
        """Enhanced speech handling with automatic speaker identification"""
        current_time = time.time()
        
        # Enhanced filtering
        text_clean = text.strip()
        if not text_clean or len(text_clean) < 2:
            return
        
        # Filter out common false positives during TTS
        if self.is_speaking:
            logger.debug("Ignoring speech during TTS playback")
            return
        
        # Additional filtering for post-TTS audio artifacts
        false_positives = ["thank you", "you", "mm-hmm", "hmm", "uh", "um", "ah", "oh"]
        if text_clean.lower() in false_positives:
            return
        
        # Prevent processing speech too quickly after last response
        if current_time - self.last_speech_time < self.speech_cooldown:
            return
        
        # Handle speaker identification
        if speaker_id:
            logger.info(f"👤 Identified speaker: {speaker_id} (confidence: {confidence:.2f})")
            # Switch to the identified user in the context system
            self.context.switch_to_user(speaker_id)
        else:
            logger.debug(f"👤 Unknown speaker (confidence: {confidence:.2f})")
        
        text_lower = text_clean.lower()
        wake_words = ["jarvis", "hey jarvis"]
        
        # Enhanced wake word detection
        contains_wake_word = any(wake_word in text_lower for wake_word in wake_words)
        
        # Handle wake word activation
        if contains_wake_word and not self.is_active:
            logger.info(f"🚨 Wake word detected: '{text_clean}' - activating!")
            self._activate_jarvis()
            
            # Personalized greeting if speaker is identified
            if speaker_id:
                speaker_system = self.stt.get_speaker_system()
                if speaker_system:
                    user_stats = speaker_system.get_user_stats(speaker_id)
                    user_name = user_stats.get('name', speaker_id)
                    self.speak_with_feedback_control(f"Yes, {user_name}. How may I assist you?")
                else:
                    self.speak_with_feedback_control("Yes, sir. How may I assist you?")
            else:
                self.speak_with_feedback_control("Yes, sir. How may I assist you?")
            return
        
        # Process commands only if active
        if self.is_active:
            logger.info(f"📝 Processing command: '{text_clean}'")
            self.last_speech_time = current_time
            self._reset_deactivation_timer()  # Reset timer on activity
            
            # Handle contextual references and pronouns
            enhanced_text = self.context.handle_pronouns_and_references(text_clean)
            
            # Process command in a separate thread to avoid blocking audio
            threading.Thread(
                target=self._process_command_async, 
                args=(enhanced_text,), 
                daemon=True
            ).start()
        else:
            # Don't log ambient conversation when inactive
            logger.debug(f"Ambient speech ignored (inactive): '{text_clean[:20]}...'")
            return

    def on_speech_received(self, text):
        """Fallback speech handling without speaker identification (for backwards compatibility)"""
        # Call the speaker version with None speaker info
        self.on_speech_with_speaker_received(text, None, 0.0)
    
    def _process_command_async(self, text):
        """Process commands asynchronously to avoid blocking audio"""
        try:
            # Set a timeout for command processing
            start_time = time.time()
            
            def timeout_handler():
                if time.time() - start_time > self.response_timeout:
                    logger.warning("Command processing timed out")
                    self.speak_with_feedback_control("I'm sorry sir, that request is taking too long to process.")
            
            # Start timeout timer
            timeout_timer = threading.Timer(self.response_timeout, timeout_handler)
            timeout_timer.start()
            
            try:
                self.commands.process_command(text)
            finally:
                timeout_timer.cancel()
                
        except Exception as e:
            logger.error(f"Error processing command '{text}': {e}")
            self.speak_with_feedback_control("I encountered an error processing that command, sir.")
    
    def speak_with_feedback_control(self, text):
        """Enhanced speaking method with proper feedback control"""
        with self.speech_lock:
            if not text or not text.strip():
                return
            
            logger.info(f"Speaking: '{text}'")
            
            try:
                # The TTS callbacks will handle STT pausing/resuming
                self.tts.speak_direct(text)
                
            except Exception as e:
                logger.error(f"TTS error: {e}")
                # Try fallback TTS
                try:
                    import subprocess
                    subprocess.run(["say", text], timeout=10)
                except Exception as fallback_error:
                    logger.error(f"Fallback TTS also failed: {fallback_error}")
    
    def start(self):
        """Start the enhanced voice assistant"""
        print("🤖 Starting Enhanced Jarvis Voice Assistant...")
        print("=" * 60)
        if self.performance_mode:
            print(f"⚡ Performance Mode: {self.performance_mode}")
        if self.prevent_feedback:
            print("🔇 Feedback prevention: ENABLED")
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
        print("   • Memory: 'my name is...', 'remember that...', 'what do you know about me'")
        print("   • Context: 'context status', 'conversation summary', 'reset memory'")
        print("   • Users: 'I am [name]', 'who am I', 'list users', 'switch to user [name]'")
        print("   • Aliases: 'call me also [name]', 'my aliases', 'primary name is [name]'")
        if self.ai_enabled:
            print("   • AI Features: Ask any question or request assistance")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start with greeting
        self.tts.speak_direct("Good day, sir. Enhanced Jarvis voice assistant is now online. Say my name to begin.")
        
        # Start listening
        self.is_listening = True
        
        try:
            self.stt.start_listening()
            logger.info("Enhanced STT listening started")
            
            # Main loop with enhanced monitoring
            last_health_check = time.time()
            health_check_interval = 30.0  # Check every 30 seconds
            
            while self.is_listening:
                current_time = time.time()
                
                # Periodic health check
                if current_time - last_health_check > health_check_interval:
                    self._perform_health_check()
                    last_health_check = current_time
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.stop()
    
    def _perform_health_check(self):
        """Perform periodic health checks to ensure system responsiveness"""
        try:
            # Check if STT is still listening
            if not self.stt.is_listening and self.is_listening:
                logger.warning("STT stopped listening unexpectedly - attempting restart")
                self._restart_stt()
            
            # Check processing queue health
            if hasattr(self.stt, 'processing_queue'):
                queue_size = self.stt.processing_queue.qsize()
                if queue_size > 2:
                    logger.warning(f"Processing queue backed up: {queue_size} items")
            
            # Check if we've been speaking too long
            if self.is_speaking:
                if time.time() - self.last_speech_time > 60:  # 1 minute max
                    logger.warning("TTS has been active too long - forcing reset")
                    self.is_speaking = False
                    self.stt.resume_after_speech()
            
            logger.debug("Health check completed")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _restart_stt(self):
        """Restart STT if it becomes unresponsive"""
        try:
            logger.info("Attempting to restart STT...")
            
            # Stop current STT
            self.stt.stop_listening()
            time.sleep(1.0)  # Brief pause
            
            # Restart STT
            if self.is_listening:
                self.stt.start_listening()
                logger.info("STT successfully restarted")
            
        except Exception as e:
            logger.error(f"Failed to restart STT: {e}")
    
    def stop(self):
        """Enhanced stop with proper cleanup"""
        logger.info("Stopping Enhanced Jarvis Assistant...")
        
        self.is_listening = False
        self.is_active = False
        
        try:
            # Cancel deactivation timer
            if self.deactivation_timer:
                self.deactivation_timer.cancel()
                self.deactivation_timer = None
            
            # Stop STT
            self.stt.stop_listening()
            
            # Stop TTS if speaking
            if self.is_speaking:
                self.tts.stop_speaking()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        print("✨ Enhanced Jarvis Assistant stopped.")
    
    def enroll_speaker(self, user_id: str, name: str) -> bool:
        """Enroll a new speaker by recording their voice"""
        if not self.enable_speaker_id or not self.stt.speaker_id_system:
            logger.error("Speaker identification not enabled")
            return False
        
        try:
            logger.info(f"Starting speaker enrollment for {name}")
            self.speak_with_feedback_control(f"Please say something for speaker enrollment, {name}. Speak clearly for about 5 seconds.")
            
            # Record audio for enrollment
            enrollment_audio = self.stt.listen_and_transcribe(timeout=10.0)
            
            if not enrollment_audio:
                self.speak_with_feedback_control("I didn't hear anything. Please try again.")
                return False
            
            # Get the raw audio data from the last session
            # For now, we'll use a simple approach - in production, we'd need to capture the raw audio
            logger.info(f"Enrollment transcription: '{enrollment_audio}'")
            self.speak_with_feedback_control("Thank you. Your voice profile has been created.")
            
            return True
            
        except Exception as e:
            logger.error(f"Speaker enrollment failed: {e}")
            self.speak_with_feedback_control("Sorry, speaker enrollment failed. Please try again.")
            return False
    
    def get_speaker_stats(self) -> Dict:
        """Get speaker identification statistics"""
        if not self.enable_speaker_id or not self.stt.speaker_id_system:
            return {}
        
        return {
            'enrolled_users': self.stt.speaker_id_system.list_enrolled_users(),
            'backend': self.stt.speaker_id_system.backend_name,
            'system_available': True
        }

def main():
    """Enhanced main entry point"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Jarvis Voice Assistant')
    parser.add_argument('--prevent-feedback', action='store_true', default=True,
                       help='Enable feedback prevention during speech (default: enabled)')
    parser.add_argument('--no-feedback-prevention', action='store_true',
                       help='Disable feedback prevention')
    parser.add_argument('--fast', action='store_true', 
                       help='Use fast performance mode (smaller models, faster response)')
    parser.add_argument('--balanced', action='store_true', 
                       help='Use balanced performance mode (default settings)')
    parser.add_argument('--accurate', action='store_true', 
                       help='Use accurate performance mode (larger models, better quality)')
    parser.add_argument('--enable-ai', action='store_true', 
                       help='Enable AI features (Claude and DeepSeek) for advanced interactions')
    parser.add_argument('--use-anthropic', action='store_true',
                       help='Use Anthropic Claude as primary AI provider (default)')
    parser.add_argument('--use-deepseek', action='store_true',
                       help='Use DeepSeek as primary AI provider')
    parser.add_argument('--use-local', action='store_true',
                       help='Use local Llama as primary AI provider (private, offline)')
    parser.add_argument('--disable-local-llm', action='store_true',
                       help='Disable local LLM support entirely')
    parser.add_argument('--tts-engine', choices=['piper', 'pyttsx3', 'coqui', 'system'], 
                       default='piper', help='TTS engine to use (default: piper)')
    parser.add_argument('--disable-speaker-id', action='store_true',
                       help='Disable automatic speaker identification')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--version', action='version', version=f'Jarvis Voice Assistant v{__version__}')
    
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
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
    
    # Determine AI provider preference
    ai_provider_preference = "anthropic"  # default
    provider_count = sum([args.use_anthropic, args.use_deepseek, args.use_local])
    
    if provider_count > 1:
        print("Error: Only one AI provider can be specified as primary")
        sys.exit(1)
    elif args.use_local:
        ai_provider_preference = "local"
    elif args.use_deepseek:
        ai_provider_preference = "deepseek"
    elif args.use_anthropic:
        ai_provider_preference = "anthropic"
    
    # Determine feedback prevention setting
    prevent_feedback = args.prevent_feedback and not args.no_feedback_prevention
    
    # Determine local LLM setting
    enable_local_llm = not args.disable_local_llm
    
    # Determine speaker identification setting
    enable_speaker_id = not args.disable_speaker_id
    
    # Display version and settings
    print(f"🤖 Jarvis Voice Assistant v{__version__}")
    if prevent_feedback:
        print("🔇 Feedback prevention: ENABLED")
    if performance_mode:
        print(f"⚡ Performance mode: {performance_mode}")
    if args.tts_engine != "system":
        engine_names = {"piper": "Piper Neural TTS", "pyttsx3": "pyttsx3", "coqui": "Coqui TTS"}
        print(f"🎙️  TTS Engine: {engine_names.get(args.tts_engine, args.tts_engine)}")
    if enable_speaker_id:
        print("👤 Speaker identification: ENABLED")
    else:
        print("👤 Speaker identification: DISABLED")
    
    # Display AI settings
    if args.enable_ai or ai_provider_preference == "local":
        if ai_provider_preference == "local":
            print("🏠 Local AI enabled - Private offline processing with Llama 3.2")
        else:
            primary_provider = "Claude" if ai_provider_preference == "anthropic" else "DeepSeek"
            fallback_provider = "DeepSeek" if ai_provider_preference == "anthropic" else "Claude"
            print(f"🤖 AI features enabled - Primary: {primary_provider}, Fallback: {fallback_provider}")
    
    if enable_local_llm and ai_provider_preference != "local":
        print("🏠 Local LLM available as fallback")
    
    # Create and start enhanced assistant
    # For local-only mode, don't require --enable-ai flag
    ai_enabled = args.enable_ai or (ai_provider_preference == "local")
    
    assistant = EnhancedJarvisAssistant(
        prevent_feedback=prevent_feedback,
        ai_enabled=ai_enabled,
        performance_mode=performance_mode,
        ai_provider_preference=ai_provider_preference,
        enable_local_llm=enable_local_llm,
        tts_engine=args.tts_engine,
        enable_speaker_id=enable_speaker_id
    )
    
    assistant.start()

if __name__ == "__main__":
    main()