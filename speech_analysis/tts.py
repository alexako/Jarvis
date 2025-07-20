"""
Jarvis TTS Pipeline - Because your assistant needs to sound like Paul Bettany, not a dying robot.
This handles text-to-speech with multiple engines and voice cloning capabilities.
"""

import asyncio
import threading
import time
import os
import io
import wave
import tempfile
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyaudio
import queue
import json
import sys

# Add parent directory to path to import config_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_manager import get_config

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TTSConfig:
    """TTS configuration settings"""

    sample_rate: int = 22050
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    voice_speed: int = 180  # words per minute
    voice_pitch: float = 1.0
    voice_volume: float = 0.9
    jarvis_personality: bool = True


class AudioPlayer:
    """Handles audio playback - because we need to hear this beauty"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.is_playing = False
        self.playback_queue = queue.Queue()
        self.playback_thread = None
        
    def play_audio_data(self, audio_data: bytes):
        """Play audio data directly"""
        if self.is_playing:
            logger.warning("Already playing audio, queuing...")
            
        self.playback_queue.put(audio_data)
        
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()

    def _playback_worker(self):
        """Worker thread for audio playback"""
        while True:
            try:
                audio_data = self.playback_queue.get(timeout=1.0)
                if audio_data is None:  # Shutdown signal
                    break
                    
                self.is_playing = True
                
                # Open stream for playback
                stream = self.audio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    output=True,
                    frames_per_buffer=self.config.chunk_size
                )
                
                # Play audio in chunks
                for i in range(0, len(audio_data), self.config.chunk_size):
                    chunk = audio_data[i:i + self.config.chunk_size]
                    stream.write(chunk)
                
                stream.stop_stream()
                stream.close()
                
                self.is_playing = False
                self.playback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback error: {e}")
                self.is_playing = False

    def stop_playback(self):
        """Stop all playback"""
        self.playback_queue.put(None)  # Shutdown signal
        self.is_playing = False

    def __del__(self):
        """Cleanup"""
        self.stop_playback()
        if hasattr(self, 'audio'):
            self.audio.terminate()


class PyttsxTTS:
    """pyttsx3-based TTS - The reliable workhorse"""

    def __init__(self, config: TTSConfig):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.available = True
            self.config = config

            # Configure voice properties
            self._setup_voice()
            logger.info("pyttsx3 TTS engine initialized")

        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self.available = False

    def _setup_voice(self):
        """Setup voice properties for Jarvis-like speech"""
        if not self.available:
            return

        # Get available voices
        voices = self.engine.getProperty('voices')

        # Try to find a suitable voice (prefer male, British if available)
        selected_voice = None
        for voice in voices:
            voice_name = voice.name.lower()
            if 'male' in voice_name or 'david' in voice_name or 'daniel' in voice_name:
                selected_voice = voice.id
                break

        if selected_voice:
            self.engine.setProperty('voice', selected_voice)
            logger.info(f"Selected voice: {selected_voice}")

        # Set speech properties
        self.engine.setProperty('rate', self.config.voice_speed)
        self.engine.setProperty('volume', self.config.voice_volume)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes"""
        if not self.available:
            logger.error("pyttsx3 not available")
            return b""

        try:
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Save speech to file
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()

            # Read audio data
            with wave.open(tmp_path, 'rb') as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())

            # Cleanup
            os.unlink(tmp_path)

            return audio_data

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return b""

    def speak_directly(self, text: str):
        """Speak text directly without returning audio data"""
        if not self.available:
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Direct speech failed: {e}")


class CoquiTTS:
    """Coqui TTS - The voice cloning champion"""


    def __init__(self, config: TTSConfig, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name=model_name)
            self.available = True
            self.config = config

            logger.info(f"Coqui TTS initialized with model: {model_name}")

        except ImportError:
            logger.error("Coqui TTS not installed. Run: pip install TTS")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            self.available = False

    def synthesize(self, text: str, speaker_wav: Optional[str] = None) -> bytes:
        """Synthesize text with optional speaker cloning"""
        if not self.available:
            logger.error("Coqui TTS not available")
            return b""

        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Synthesize speech
            if speaker_wav and os.path.exists(speaker_wav):
                # Use voice cloning
                self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=tmp_path)
                logger.info(f"Voice cloned synthesis using: {speaker_wav}")
            else:
                # Use default voice
                self.tts.tts_to_file(text=text, file_path=tmp_path)

            # Read audio data
            with wave.open(tmp_path, 'rb') as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())

            # Cleanup
            os.unlink(tmp_path)

            return audio_data

        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            return b""

    def clone_voice(self, text: str, source_audio_path: str) -> bytes:
        """Clone voice from source audio file"""
        return self.synthesize(text, speaker_wav=source_audio_path)


class JarvisPersonality:
    """Jarvis personality and speech patterns - The secret sauce"""


    def __init__(self):
        self.formal_responses = {
            "greeting": [
                "Good morning, sir.",
                "Good afternoon, sir.",
                "Good evening, sir.",
                "How may I assist you today, sir?",
                "At your service, sir."
            ],
            "acknowledgment": [
                "Certainly, sir.",
                "Right away, sir.",
                "Of course, sir.",
                "Consider it done, sir.",
                "Immediately, sir."
            ],
            "thinking": [
                "Let me check on that for you, sir.",
                "One moment please, sir.",
                "Allow me to process that, sir.",
                "Searching for that information, sir."
            ],
            "error": [
                "I'm afraid I don't understand, sir.",
                "Could you please rephrase that, sir?",
                "I'm having difficulty with that request, sir.",
                "I don't have that information at the moment, sir."
            ],
            "goodbye": [
                "Until next time, sir.",
                "Have a pleasant day, sir.",
                "Goodbye, sir.",
                "I'll be here when you need me, sir."
            ]
        }

        self.context_responses = {
            "time": "The time is {time}, sir.",
            "weather": "The weather is {weather}, sir. Shall I adjust the climate control?",
            "reminder": "You have {count} reminders, sir. Shall I review them?",
            "calendar": "Your next appointment is {appointment}, sir."
        }

    def enhance_response(self, text: str, context: str = "general") -> str:
        """Add Jarvis personality to response text"""
        if not text:
            return text

        # Add formal address if not present
        if "sir" not in text.lower() and len(text) > 10:
            if text.endswith('.'):
                text = text[:-1] + ", sir."
            else:
                text += ", sir."

        # Add context-specific enhancements
        if context == "confirmation":
            text = f"Certainly. {text}"
        elif context == "information":
            text = f"According to my records, {text.lower()}"
        elif context == "action":
            text = f"Right away. {text}"

        return text

    def get_contextual_response(self, response_type: str, **kwargs) -> str:
        """Get contextual response with variables"""
        if response_type in self.context_responses:
            return self.context_responses[response_type].format(**kwargs)
        elif response_type in self.formal_responses:
            import random
            return random.choice(self.formal_responses[response_type])
        else:
            return "I'm at your service, sir."


class JarvisTTS:
    """Main TTS coordinator - The voice of your digital butler"""


    def __init__(self, 
                 tts_engine: str = "pyttsx3",
                 model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
                 jarvis_voice_path: Optional[str] = None):

        self.config = TTSConfig()
        self.personality = JarvisPersonality()
        self.player = AudioPlayer(self.config)
        self.jarvis_voice_path = jarvis_voice_path

        # Initialize TTS engine
        if tts_engine.lower() == "pyttsx3":
            self.tts_engine = PyttsxTTS(self.config)
        elif tts_engine.lower() == "coqui":
            self.tts_engine = CoquiTTS(self.config, model_name)
        elif tts_engine.lower() == "system":
            self.tts_engine = None  # Use system say directly
        else:
            raise ValueError(f"Unknown TTS engine: {tts_engine}")

        # Callbacks
        self.on_speech_start_callback: Optional[Callable[[], None]] = None
        self.on_speech_end_callback: Optional[Callable[[], None]] = None
        
        # Non-blocking TTS support
        config = get_config()
        optimizations = config.get_optimizations_config()
        self.non_blocking_tts = optimizations.get('non_blocking_tts', False)
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.is_speech_thread_running = False

        logger.info(f"Jarvis TTS initialized with {tts_engine} engine, non-blocking: {self.non_blocking_tts}")

    def set_speech_callbacks(self, 
                           on_start: Optional[Callable[[], None]] = None,
                           on_end: Optional[Callable[[], None]] = None):
        """Set callbacks for speech events"""
        self.on_speech_start_callback = on_start
        self.on_speech_end_callback = on_end

    def _speech_worker(self):
        """Worker thread for non-blocking speech processing"""
        self.is_speech_thread_running = True
        while self.is_speech_thread_running:
            try:
                speech_item = self.speech_queue.get(timeout=1.0)
                if speech_item is None:  # Shutdown signal
                    break
                
                text, context, use_personality = speech_item
                self._speak_blocking(text, context, use_personality)
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")

    def speak(self, text: str, context: str = "general", use_personality: bool = True):
        """Speak text with Jarvis personality"""
        if not text:
            return
        
        if self.non_blocking_tts:
            # Queue speech for non-blocking processing
            self.speech_queue.put((text, context, use_personality))
            
            # Start worker thread if not running
            if not self.speech_thread or not self.speech_thread.is_alive():
                self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
                self.speech_thread.start()
        else:
            # Process speech immediately (blocking)
            self._speak_blocking(text, context, use_personality)

    def _speak_blocking(self, text: str, context: str = "general", use_personality: bool = True):
        """Internal blocking speech implementation"""
        if not text:
            return

        # Enhance text with personality
        if use_personality and self.config.jarvis_personality:
            enhanced_text = self.personality.enhance_response(text, context)
        else:
            enhanced_text = text

        logger.info(f"Speaking: '{enhanced_text}'")

        # Trigger start callback
        if self.on_speech_start_callback:
            self.on_speech_start_callback()

        try:
            # Handle system TTS engine (when tts_engine is None)
            if self.tts_engine is None:
                # Use system say directly
                self._fallback_system_say(enhanced_text)
            else:
                # Synthesize speech with other engines
                if isinstance(self.tts_engine, CoquiTTS) and self.jarvis_voice_path:
                    # Use voice cloning if available
                    audio_data = self.tts_engine.clone_voice(enhanced_text, self.jarvis_voice_path)
                else:
                    audio_data = self.tts_engine.synthesize(enhanced_text)

                if audio_data:
                    # Play audio
                    self.player.play_audio_data(audio_data)

                    # Wait for playback to complete
                    while self.player.is_playing:
                        time.sleep(0.1)

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")

        finally:
            # Trigger end callback
            if self.on_speech_end_callback:
                self.on_speech_end_callback()

    def speak_contextual(self, response_type: str, **kwargs):
        """Speak contextual response"""
        response = self.personality.get_contextual_response(response_type, **kwargs)
        self.speak(response, context=response_type)

    def speak_direct(self, text: str):
        """Speak text directly without personality enhancement"""
        if self.tts_engine is None:  # System TTS
            self._fallback_system_say(text)
        elif isinstance(self.tts_engine, PyttsxTTS):
            # Try pyttsx3 first
            try:
                self.tts_engine.speak_directly(text)
            except Exception as e:
                logger.warning(f"pyttsx3 failed, using system say: {e}")
                self._fallback_system_say(text)
        else:
            self.speak(text, use_personality=False)
    
    def _calculate_speech_timeout(self, text: str) -> float:
        """Calculate appropriate timeout based on text length and speech rate"""
        # Get configuration values
        config = get_config()
        tts_config = config.get('tts', {})
        
        speech_rate = tts_config.get('speech_rate_wpm', 150)  # words per minute
        buffer_factor = tts_config.get('timeout_buffer_factor', 0.5)
        min_timeout = tts_config.get('min_timeout_seconds', 30)
        max_timeout = tts_config.get('max_timeout_seconds', 300)
        
        # Estimate speaking time
        word_count = len(text.split())
        estimated_time = (word_count / speech_rate) * 60  # Convert to seconds
        
        # Add buffer time for processing and pauses
        buffer_time = max(10, estimated_time * buffer_factor)  # At least 10 seconds buffer
        
        # Set timeout with configured minimum and maximum
        timeout = max(min_timeout, estimated_time + buffer_time)
        timeout = min(max_timeout, timeout)
        
        return timeout

    def _fallback_system_say(self, text: str):
        """Fallback to macOS say command with dynamic timeout"""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Darwin":  # macOS
                # Calculate appropriate timeout based on text length
                timeout = self._calculate_speech_timeout(text)
                
                logger.info(f"Speaking text ({len(text.split())} words, {timeout:.1f}s timeout): '{text[:100]}{'...' if len(text) > 100 else ''}'")
                
                # Use macOS say command with dynamic timeout
                subprocess.run(["say", "-v", "Daniel", text], check=True, timeout=timeout)
                logger.info(f"System say completed successfully")
            else:
                logger.error("System say fallback not available on this platform")
                
        except subprocess.TimeoutExpired:
            word_count = len(text.split())
            logger.error(f"System say timed out after {timeout:.1f}s for {word_count} words. Text may be too long or system is overloaded.")
        except Exception as e:
            logger.error(f"System say failed: {e}")

    def stop_speaking(self):
        """Stop current speech"""
        self.player.stop_playback()
        
        # Stop non-blocking speech thread if running
        if self.non_blocking_tts and self.is_speech_thread_running:
            self.is_speech_thread_running = False
            self.speech_queue.put(None)  # Send shutdown signal

    def set_jarvis_voice(self, voice_sample_path: str):
        """Set Jarvis voice sample for cloning"""
        if os.path.exists(voice_sample_path):
            self.jarvis_voice_path = voice_sample_path
            logger.info(f"Jarvis voice set to: {voice_sample_path}")
        else:
            logger.error(f"Voice sample not found: {voice_sample_path}")

    def test_voice(self):
        """Test the voice with sample Jarvis phrases"""
        test_phrases = [
            "Good morning, sir. How may I assist you today?",
            "Certainly, sir. Right away.",
            "I'm sorry, sir. I don't have that information at the moment.",
            "Your meeting is scheduled for 3 PM, sir. Shall I send a reminder?",
            "All systems are functioning normally, sir."
        ]

        print("Testing Jarvis voice...")
        for phrase in test_phrases:
            print(f"Speaking: {phrase}")
            self.speak(phrase)
            time.sleep(2)  # Pause between phrases

    def create_voice_sample(self, text: str, output_path: str):
        """Create a voice sample file for testing"""
        try:
            audio_data = self.tts_engine.synthesize(text)

            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_data)

            logger.info(f"Voice sample created: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create voice sample: {e}")


# Voice training utilities

class JarvisVoiceTrainer:
    """Train Jarvis voice from audio samples - The voice coach"""


    def __init__(self, training_data_dir: str = "jarvis_training_data"):
        self.training_dir = Path(training_data_dir)
        self.training_dir.mkdir(exist_ok=True)

    def extract_audio_from_video(self, video_path: str, output_dir: str = None):
        """Extract audio from video files (Iron Man clips)"""
        if output_dir is None:
            output_dir = self.training_dir / "extracted_audio"

        try:
            import subprocess

            output_path = Path(output_dir) / f"{Path(video_path).stem}.wav"

            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", video_path,
                "-acodec", "pcm_s16le",
                "-ar", "22050",
                "-ac", "1",
                str(output_path)
            ]

            subprocess.run(cmd, check=True)
            logger.info(f"Audio extracted to: {output_path}")
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")

    def segment_audio(self, audio_path: str, output_dir: str = None):
        """Segment audio into sentences using silence detection"""
        if output_dir is None:
            output_dir = self.training_dir / "segments"

        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence

            # Load audio
            audio = AudioSegment.from_wav(audio_path)

            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=500,  # 0.5 seconds
                silence_thresh=audio.dBFS - 14,
                keep_silence=100
            )
            
            # Save segments
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 1000:  # Only save chunks > 1 second
                    chunk.export(output_dir / f"segment_{i:03d}.wav", format="wav")
            
            logger.info(f"Audio segmented into {len(chunks)} pieces")
            
        except Exception as e:
            logger.error(f"Audio segmentation failed: {e}")


# Example usage and testing

if __name__ == "__main__":
    def on_speech_start():
        print("Jarvis is speaking")


    def on_speech_end():
        print("Jarvis finished speaking")

    # Create TTS instance
    # jarvis = JarvisTTS(tts_engine="pyttsx3")  # Use "coqui" for voice cloning
    jarvis = JarvisTTS(tts_engine="system")  # Use "coqui" for voice cloning

    # Set callbacks
    jarvis.set_speech_callbacks(on_speech_start, on_speech_end)

    # Test basic speech
    print("Testing basic speech...")
    jarvis.speak("Hello, this is Jarvis speaking.")

    # Test contextual responses
    print("\nTesting contextual responses...")
    jarvis.speak_contextual("greeting")
    jarvis.speak_contextual("acknowledgment")
    jarvis.speak_contextual("time", time="3:30 PM")

    # Test personality enhancement
    print("\nTesting personality enhancement...")
    jarvis.speak("The weather is sunny today", context="information")
    jarvis.speak("I will set a reminder for you", context="action")

    # Run voice test
    print("\nRunning full voice test...")
    jarvis.test_voice()

    print("Jarvis TTS pipeline test complete!")

