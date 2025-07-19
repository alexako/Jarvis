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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio configuration settings"""

    def __init__(self, performance_mode: str = None):
        config = get_config()
        audio_config = config.get_audio_config(performance_mode)
        optimizations = config.get_optimizations_config()
        
        self.sample_rate: int = audio_config.get('sample_rate', 16000)
        self.channels: int = audio_config.get('channels', 1)
        self.chunk_size: int = audio_config.get('chunk_size', 1024)
        
        # Handle format string conversion
        format_str = audio_config.get('format', 'paInt16')
        if format_str == 'paInt16':
            self.format: int = pyaudio.paInt16
        else:
            self.format: int = pyaudio.paInt16  # default fallback
            
        # FIXED: Much more reasonable thresholds
        self.silence_threshold: float = max(150.0, audio_config.get('silence_threshold', 150.0))  # Increased from 30
        self.silence_duration: float = max(2.5, audio_config.get('silence_duration', 2.5))  # Increased from 0.8
        self.max_recording_time: float = max(20.0, audio_config.get('max_recording_time', 20.0))  # Increased from 6
        
        # Performance optimizations
        self.smart_silence_detection: bool = optimizations.get('smart_silence_detection', True)
        self.audio_compression: bool = optimizations.get('audio_compression', False)


class AudioBuffer:
    """FIXED: Circular buffer for audio data with proper voice activity detection"""

    def __init__(self, max_size: int = 320000, debug: bool = False):  # FIXED: 20 seconds at 16kHz instead of 2
        self.buffer = deque(maxlen=max_size)
        self.is_recording = False
        self.silence_counter = 0
        self.speech_detected = False
        self.recent_rms = deque(maxlen=50)  # FIXED: Track more samples for better adaptation
        self.background_rms = 0.0
        self.recording_chunks = 0
        self.debug = debug
        self.speech_chunks = 0  # NEW: Track how much actual speech we've captured
        
    def add_chunk(self, chunk: np.ndarray, config: AudioConfig) -> bool:
        """Add audio chunk and detect speech activity with better logic"""
        # Calculate RMS for voice activity detection
        if len(chunk) == 0:
            rms = 0.0
        else:
            mean_square = np.mean(chunk.astype(np.float64) ** 2)
            if np.isnan(mean_square) or mean_square < 0:
                rms = 0.0
            else:
                rms = np.sqrt(mean_square)
        
        # Track recent RMS values for adaptive thresholding
        self.recent_rms.append(rms)
        
        # FIXED: Better background noise estimation
        if not self.speech_detected and len(self.recent_rms) >= 10:
            # Use median instead of mean for more robust background estimation
            self.background_rms = np.median(list(self.recent_rms))
        
        # FIXED: More intelligent adaptive thresholds
        if self.background_rms > 0:
            # Speech threshold should be significantly above background
            speech_threshold = max(config.silence_threshold, self.background_rms * 4.0)
            # Silence threshold should be closer to background but still above it
            silence_threshold = max(config.silence_threshold * 0.3, self.background_rms * 1.5)
        else:
            speech_threshold = config.silence_threshold * 2.0
            silence_threshold = config.silence_threshold * 0.4
        
        # Debug logging
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        debug_config = get_config().get_debug_config()
        rms_logging_interval = debug_config.get('rms_logging_interval', 500)
            
        if self.debug and self._debug_counter % rms_logging_interval == 0:
            logger.info(f"RMS: {rms:.1f}, Speech Threshold: {speech_threshold:.1f}, Silence Threshold: {silence_threshold:.1f}, Background: {self.background_rms:.1f}, Speech: {self.speech_detected}, Chunks: {self.speech_chunks}")
        
        # Speech detection: use higher threshold
        if not self.speech_detected and rms > speech_threshold:
            if self.debug: 
                logger.info(f"🎤 Speech detected - starting recording (RMS: {rms:.1f} > {speech_threshold:.1f})")
            self.speech_detected = True
            self.is_recording = True
            self.silence_counter = 0
            self.recording_chunks = 0
            self.speech_chunks = 0
            
        elif self.speech_detected:
            if rms > silence_threshold:
                # Still speech or above silence threshold
                self.silence_counter = 0
                if rms > speech_threshold:
                    self.speech_chunks += 1  # Count chunks with strong speech
            else:
                # Below silence threshold
                self.silence_counter += 1
                
                silence_threshold_chunks = config.silence_duration * config.sample_rate / config.chunk_size
                
                if self.debug and self.silence_counter % 5 == 0:
                    logger.info(f"Silence counting: {self.silence_counter}/{silence_threshold_chunks:.1f} chunks (RMS: {rms:.1f} <= {silence_threshold:.1f})")
                
                # FIXED: Only stop if we've captured meaningful speech
                if self.silence_counter > silence_threshold_chunks and self.speech_chunks > 5:
                    logger.info(f"✅ Speech complete - {self.speech_chunks} speech chunks, {self.silence_counter} silence chunks")
                    self.is_recording = False
                    return True
        
        # Check for maximum recording time (prevent getting stuck)
        if self.is_recording:
            self.recording_chunks += 1
            max_recording_chunks = config.max_recording_time * config.sample_rate / config.chunk_size
            
            if self.recording_chunks > max_recording_chunks:
                # Only force completion if we have some speech
                if self.speech_chunks > 3:
                    logger.warning(f"Maximum recording time reached - completing with {self.speech_chunks} speech chunks")
                    self.is_recording = False
                    return True
                else:
                    # Reset if we haven't captured meaningful speech
                    logger.warning("Maximum recording time reached with minimal speech - resetting")
                    self.speech_detected = False
                    self.is_recording = False
                    self.buffer.clear()
                    return False
            
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
        self.speech_chunks = 0
        return data


class WhisperSTT:
    """Whisper-based STT implementation - The heavyweight championÃ¢"""

    def __init__(self, model_name: str = None, performance_mode: str = None):
        config = get_config()
        optimizations = config.get_optimizations_config()
        
        if model_name is None:
            if performance_mode:
                stt_config = config.get_stt_config(performance_mode)
                model_name = stt_config.get('whisper', {}).get('default_model', 'tiny.en')
            else:
                model_name = config.get('stt.whisper.default_model', 'tiny.en')
        
        self.gpu_acceleration = optimizations.get('gpu_acceleration', False)
        self.model_name = model_name
            
        try:
            import whisper
            
            if self.gpu_acceleration:
                import torch
                if torch.cuda.is_available():
                    self.model = whisper.load_model(model_name, device="cuda")
                    logger.info(f"Whisper model '{model_name}' loaded with GPU acceleration")
                else:
                    logger.warning("GPU acceleration requested but CUDA not available, falling back to CPU")
                    self.model = whisper.load_model(model_name)
                    logger.info(f"Whisper model '{model_name}' loaded on CPU")
            else:
                self.model = whisper.load_model(model_name)
                logger.info(f"Whisper model '{model_name}' loaded on CPU")
            
            self.available = True
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
            
            # FIXED: Better Whisper parameters for longer speech
            result = self.model.transcribe(
                audio_float, 
                language="en",
                temperature=0.0,  # More deterministic
                compression_ratio_threshold=2.4,  # Less aggressive filtering
                logprob_threshold=-1.0,  # Less aggressive filtering
                no_speech_threshold=0.6,  # Slightly less strict
                word_timestamps=True,  # Better for longer audio
                verbose=False
            )
            
            text = result.get("text", "").strip()
            
            # FIXED: Much less aggressive filtering
            if len(text) > 1 and not text in [".", "..", "..."]:
                return text
            else:
                return ""
            
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
                 debug: bool = None,
                 performance_mode: str = None):
        
        # Load config values if not provided
        config = get_config()
        if stt_engine is None:
            if performance_mode:
                stt_config = config.get_stt_config(performance_mode)
                stt_engine = stt_config.get('default_engine', 'whisper')
            else:
                stt_engine = config.get('stt.default_engine', 'whisper')
        if debug is None:
            debug = config.get('debug.audio_processing', False)
        
        self.performance_mode = performance_mode
        self.config = AudioConfig(performance_mode)
        self.audio_buffer = AudioBuffer(debug=debug)
        self.wake_detector = WakeWordDetector(wake_words)
        self.is_listening = False
        self.is_processing = False
        self.debug = debug
        
        # Initialize STT engine with performance mode
        if stt_engine.lower() == "whisper":
            self.stt_engine = WhisperSTT(model_name, performance_mode)
        elif stt_engine.lower() == "vosk":
            self.stt_engine = VoskSTT(model_name)
        else:
            raise ValueError(f"Unknown STT engine: {stt_engine}")

        logger.info(f"STT Engine loaded with performance mode: {performance_mode or 'default'}")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Callbacks
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

