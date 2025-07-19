#!/usr/bin/env python3
"""
Jarvis STT Pipeline - Because your assistant needs to actually hear you, duh.
This is the foundation for speech-to-text processing with both Whisper and Vosk options.
"""

import threading
import time
import numpy as np
import pyaudio
import wave
import io
import json
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from collections import deque
import logging
import sys
import os

# Add parent directory to path to import config_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_manager import get_config

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio configuration settings"""

    def __init__(self):
        config = get_config()
        audio_config = config.get_audio_config()
        
        self.sample_rate: int = audio_config.get('sample_rate', 16000)
        self.channels: int = audio_config.get('channels', 1)
        self.chunk_size: int = audio_config.get('chunk_size', 1024)
        
        # Handle format string conversion
        format_str = audio_config.get('format', 'paInt16')
        if format_str == 'paInt16':
            self.format: int = pyaudio.paInt16
        else:
            self.format: int = pyaudio.paInt16  # default fallback
            
        self.silence_threshold: float = audio_config.get('silence_threshold', 50.0)
        self.silence_duration: float = audio_config.get('silence_duration', 0.3)
        self.max_recording_time: float = audio_config.get('max_recording_time', 10.0)


class AudioBuffer:
    """Circular buffer for audio data with voice activity detection"""

    def __init__(self, max_size: int = 32000, debug: bool = False):  # ~2 seconds at 16kHz
        self.buffer = deque(maxlen=max_size)
        self.is_recording = False
        self.silence_counter = 0
        self.speech_detected = False
        self.recent_rms = deque(maxlen=10)  # Track recent RMS values
        self.background_rms = 0.0
        self.recording_chunks = 0  # Track how long we've been recording
        self.debug = debug
        
    def add_chunk(self, chunk: np.ndarray, config: AudioConfig) -> bool:
        """Add audio chunk and detect speech activity"""
        # Calculate RMS for voice activity detection
        if len(chunk) == 0:
            rms = 0.0
        else:
            mean_square = np.mean(chunk.astype(np.float64) ** 2)
            # Handle edge cases: NaN, negative values, or very small values
            if np.isnan(mean_square) or mean_square < 0:
                rms = 0.0
            else:
                rms = np.sqrt(mean_square)
        
        # Track recent RMS values for adaptive thresholding
        self.recent_rms.append(rms)
        
        # Update background noise estimate when not speaking
        if not self.speech_detected and len(self.recent_rms) >= 5:
            self.background_rms = np.mean(self.recent_rms)
        
        # Use more reasonable adaptive thresholds
        if self.background_rms > 0:
            # Adaptive thresholds based on background noise - but keep silence threshold reasonable
            speech_threshold = max(config.silence_threshold * 2.0, self.background_rms + 30.0)
            silence_threshold = min(config.silence_threshold, self.background_rms + 10.0)
        else:
            # Initial thresholds when background is not established
            speech_threshold = config.silence_threshold * 3.0  # Higher threshold for speech
            silence_threshold = config.silence_threshold * 0.5  # Lower threshold for silence
        
        # Debug: Print RMS values occasionally to help tune threshold
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        # Get debug config for RMS logging interval
        debug_config = get_config().get_debug_config()
        rms_logging_interval = debug_config.get('rms_logging_interval', 500)
            
        if self.debug and self._debug_counter % rms_logging_interval == 0:
            logger.info(f"RMS: {rms:.1f}, Speech Threshold: {speech_threshold:.1f}, Silence Threshold: {silence_threshold:.1f}, Background: {self.background_rms:.1f}, Speech: {self.speech_detected}")
        
        # Speech detection: use higher threshold
        if not self.speech_detected and rms > speech_threshold:
            if self.debug: logger.info(f"Speech detected - starting recording (RMS: {rms:.1f} > {speech_threshold:.1f})")
            self.speech_detected = True
            self.is_recording = True
            self.silence_counter = 0
            self.recording_chunks = 0
        # Silence detection: use lower threshold, but only if speech was detected
        elif self.speech_detected and rms <= silence_threshold:
            self.silence_counter += 1
            # If silence for configured duration, stop recording
            silence_threshold_chunks = config.silence_duration * config.sample_rate / config.chunk_size
            
            # Debug: Show silence progress occasionally
            if self.debug and self.silence_counter % 3 == 0:  # Every 3 chunks
                logger.info(f"Silence counting: {self.silence_counter}/{silence_threshold_chunks:.1f} chunks (RMS: {rms:.1f} <= {silence_threshold:.1f})")
            
            if self.silence_counter > silence_threshold_chunks:
                logger.info(f"✅ Silence detected - stopping recording (silence for {self.silence_counter} chunks, threshold: {silence_threshold_chunks:.1f})")
                self.is_recording = False
                return True  # Signal that we have a complete utterance
        # Continue speech: reset silence counter if above silence threshold but below speech threshold
        elif self.speech_detected and rms > silence_threshold:
            self.silence_counter = 0
        
        # Check for maximum recording time (prevent getting stuck)
        if self.is_recording:
            self.recording_chunks += 1
            max_recording_chunks = config.max_recording_time * config.sample_rate / config.chunk_size
            
            if self.recording_chunks > max_recording_chunks:
                logger.warning(f"Maximum recording time reached ({config.max_recording_time}s) - forcing utterance completion")
                self.is_recording = False
                return True  # Force utterance completion
            
            self.buffer.extend(chunk)
            
        return False

    def get_audio_data(self) -> np.ndarray:
        """Get the complete audio data and reset buffer"""
        if not self.buffer:
            return np.array([])
        
        data = np.array(list(self.buffer))
        self.buffer.clear()
        self.speech_detected = False
        self.is_recording = False
        self.silence_counter = 0
        self.recording_chunks = 0
        return data


class WhisperSTT:
    """Whisper-based STT implementation - The heavyweight championÃ¢"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            config = get_config()
            model_name = config.get('stt.whisper.default_model', 'tiny.en')
            
        try:
            import whisper
            self.model = whisper.load_model(model_name)
            self.available = True
            logger.info(f"Whisper model '{model_name}' loaded successfully")
        except ImportError:
            logger.error("Whisper not installed. Run: pip install openai-whisper")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.available = False

    def transcribe(self, audio_data: np.ndarray, config: AudioConfig) -> str:
        """Transcribe audio using Whisper"""
        if not self.available:
            return "Whisper not available"
        
        if len(audio_data) == 0:
            return ""
        
        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Whisper expects the audio to be properly formatted
            result = self.model.transcribe(audio_float, language="en")
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""


class VoskSTT:
    """Vosk-based STT implementation - The lightweight speedsterÃ¢"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            config = get_config()
            model_path = config.get('stt.vosk.default_model', 'vosk-model-small-en-us-0.15')
            
        try:
            import vosk
            self.model = vosk.Model(model_path)
            self.available = True
            logger.info(f"Vosk model loaded from {model_path}")
        except ImportError:
            logger.error("Vosk not installed. Run: pip install vosk")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            self.available = False

    def transcribe(self, audio_data: np.ndarray, config: AudioConfig) -> str:
        """Transcribe audio using Vosk"""
        if not self.available:
            return "Vosk not available"
        
        if len(audio_data) == 0:
            return ""
        
        try:
            import vosk
            
            # Create recognizer
            rec = vosk.KaldiRecognizer(self.model, config.sample_rate)
            
            # Convert to bytes
            audio_bytes = audio_data.tobytes()
            
            # Process audio
            if rec.AcceptWaveform(audio_bytes):
                result = json.loads(rec.Result())
                return result.get("text", "").strip()
            else:
                partial = json.loads(rec.PartialResult())
                return partial.get("partial", "").strip()
                
        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return ""


class WakeWordDetector:
    """Simple wake word detection - because we need to know when to listenÃ¢"""

    def __init__(self, wake_words: list = None):
        if wake_words is None:
            config = get_config()
            wake_words = config.get_wake_words()
            
        self.wake_words = [word.lower() for word in wake_words]
        self.last_detection = 0
        self.cooldown = 2.0  # seconds

    def detect(self, text: str) -> bool:
        """Detect wake word in transcribed text"""
        if not text:
            return False
        
        text_lower = text.lower()
        current_time = time.time()
        
        # Check cooldown to prevent spam
        if current_time - self.last_detection < self.cooldown:
            return False
        
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                self.last_detection = current_time
                logger.info(f"Wake word detected: {wake_word}")
                return True
        
        return False


class JarvisSTT:
    """Main STT coordinator - The brains of the operationÃ¢"""

    def __init__(self, 
                 stt_engine: str = None,
                 model_name: str = None,
                 wake_words: list = None,
                 debug: bool = None):
        
        # Load config values if not provided
        config = get_config()
        if stt_engine is None:
            stt_engine = config.get('stt.default_engine', 'whisper')
        if debug is None:
            debug = config.get('debug.audio_processing', False)
        
        self.config = AudioConfig()
        self.audio_buffer = AudioBuffer(debug=debug)
        self.wake_detector = WakeWordDetector(wake_words)
        self.is_listening = False
        self.is_processing = False
        self.debug = debug
        
        # Initialize STT engine
        if stt_engine.lower() == "whisper":
            self.stt_engine = WhisperSTT(model_name)
        elif stt_engine.lower() == "vosk":
            self.stt_engine = VoskSTT(model_name)
        else:
            raise ValueError(f"Unknown STT engine: {stt_engine}")

        logger.info("STT Engine loaded")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Callback for when speech is detected
        self.on_speech_callback: Optional[Callable[[str], None]] = None
        self.on_wake_word_callback: Optional[Callable[[], None]] = None

    def set_speech_callback(self, callback: Callable[[str], None]):
        """Set callback for when speech is transcribed"""
        self.on_speech_callback = callback

    def set_wake_word_callback(self, callback: Callable[[], None]):
        """Set callback for when wake word is detected"""
        self.on_wake_word_callback = callback

    def start_listening(self):
        """Start continuous listening"""
        if self.is_listening:
            logger.warning("Already listening")
            return
        
        self.is_listening = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        logger.info("Started listening for audio...")

    def stop_listening(self):
        """Stop listening"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        logger.info("Stopped listening")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for continuous processing"""
        if not self.is_listening:
            return (None, pyaudio.paComplete)
        
        # Convert audio data to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to buffer and check for complete utterance
        utterance_complete = self.audio_buffer.add_chunk(audio_chunk, self.config)
        
        if utterance_complete:
            logger.info(f"🎆 Utterance complete! is_processing: {self.is_processing}")
            if not self.is_processing:
                if self.debug: logger.info("🚀 Starting transcription thread...")
                # Process the complete utterance in a separate thread
                threading.Thread(target=self._process_audio, daemon=True).start()
            else:
                logger.warning("Cannot start transcription - already processing")
        
        return (in_data, pyaudio.paContinue)

    def _process_audio(self):
        """Process complete audio utterance"""
        logger.info(f"🔄 _process_audio called, is_processing: {self.is_processing}")
        
        if self.is_processing:
            logger.warning("Already processing audio, skipping")
            return
        
        self.is_processing = True
        logger.info("🎯 Starting audio processing...")
        
        try:
            # Get audio data from buffer
            audio_data = self.audio_buffer.get_audio_data()
            
            if len(audio_data) > 0:
                # Transcribe audio
                transcription = self.stt_engine.transcribe(audio_data, self.config)
                
                if transcription:
                    logger.info(f"Transcribed: '{transcription}'")
                    
                    # Check for wake word
                    if self.wake_detector.detect(transcription):
                        if self.on_wake_word_callback:
                            self.on_wake_word_callback()
                    
                    # Call speech callback
                    if self.on_speech_callback:
                        self.on_speech_callback(transcription)
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
        
        finally:
            self.is_processing = False

    def listen_and_transcribe(self, timeout: float = 10.0) -> str:
        """Listen from microphone and return transcription text
        
        Args:
            timeout: Maximum time to wait for speech (seconds)
            
        Returns:
            str: The transcribed text, or empty string if no speech detected
        """
        if self.debug: logger.info(f"Starting listen_and_transcribe with {timeout}s timeout")
        
        # Create a temporary audio buffer for this session
        temp_buffer = AudioBuffer(debug=self.debug)
        transcription = ""
        
        try:
            # Open audio stream for recording
            stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            if self.debug: logger.info("Listening for speech...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Read audio chunk
                try:
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to buffer and check for complete utterance
                    utterance_complete = temp_buffer.add_chunk(audio_chunk, self.config)
                    
                    if utterance_complete:
                        logger.info("Complete utterance detected, transcribing...")
                        # Get the audio data
                        audio_data = temp_buffer.get_audio_data()
                        
                        if len(audio_data) > 0:
                            # Transcribe the audio
                            transcription = self.stt_engine.transcribe(audio_data, self.config)
                            logger.info(f"Transcription result: '{transcription}'")
                        break
                        
                except Exception as e:
                    logger.warning(f"Audio read error: {e}")
                    continue
                    
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
            if self.debug and not transcription:
                logger.info("No speech detected within timeout period")
            
            return transcription.strip() if transcription else ""
            
        except Exception as e:
            logger.error(f"Error in listen_and_transcribe: {e}")
            return ""
    
    def transcribe_file(self, filename: str) -> str:
        """Transcribe audio from file - for testing"""
        try:
            # Read audio file
            with wave.open(filename, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            return self.stt_engine.transcribe(audio_data, self.config)
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return ""

    def __del__(self):
        """Cleanup"""
        self.stop_listening()
        if hasattr(self, 'audio'):
            self.audio.terminate()

# Example usage and testing

if __name__ == "__main__":
    def on_speech(text: str):
        print(f"Speech detected: {text}")

    def on_wake_word():
        print("Wake word detected! Jarvis is listening...")

    # Create STT instance
    jarvis = JarvisSTT(stt_engine="whisper", model_name="base")

    # Set callbacks
    jarvis.set_speech_callback(on_speech)
    jarvis.set_wake_word_callback(on_wake_word)

    # Start listening
    try:
        jarvis.start_listening()
        print("Listening... Say 'Jarvis' to activate. Press Ctrl+C to stop.")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        jarvis.stop_listening()

    print("Jarvis STT pipeline stopped.")

