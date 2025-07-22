#!/usr/bin/env python3
"""
Enhanced Jarvis STT Pipeline - Unified with all improvements
This is the foundation for speech-to-text processing with both Whisper and Vosk options.
Key features:
1. Better audio stream state management
2. Non-blocking speech processing
3. Proper buffer reset mechanisms
4. Enhanced feedback prevention
5. Audio stream recovery logic
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
import queue

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
            
        # Enhanced thresholds for better responsiveness
        self.silence_threshold: float = max(150.0, audio_config.get('silence_threshold', 150.0))
        self.silence_duration: float = max(1.5, audio_config.get('silence_duration', 1.5))  # Reduced from 2.5
        self.max_recording_time: float = max(15.0, audio_config.get('max_recording_time', 15.0))  # Reduced from 20
        
        # Performance optimizations
        self.smart_silence_detection: bool = optimizations.get('smart_silence_detection', True)
        self.audio_compression: bool = optimizations.get('audio_compression', False)


class AudioBuffer:
    """Enhanced audio buffer with better state management and responsiveness"""

    def __init__(self, max_size: int = 240000, debug: bool = False):  # 15 seconds at 16kHz
        self.buffer = deque(maxlen=max_size)
        self.is_recording = False
        self.silence_counter = 0
        self.speech_detected = False
        self.recent_rms = deque(maxlen=30)  # Reduced for faster adaptation
        self.background_rms = 0.0
        self.recording_chunks = 0
        self.debug = debug
        self.speech_chunks = 0
        self.last_reset_time = time.time()
        self.min_speech_chunks = 3  # Minimum speech chunks before considering valid
        self.adaptive_silence_threshold = 150.0
        
        # Enhanced state tracking
        self.consecutive_silence = 0
        self.speech_energy_history = deque(maxlen=10)
        self.is_paused = False  # For feedback prevention
        
    def pause_detection(self):
        """Pause speech detection (during TTS playback)"""
        self.is_paused = True
        logger.debug("Audio detection paused")
        
    def resume_detection(self):
        """Resume speech detection after TTS"""
        self.is_paused = False
        self.reset_state()
        logger.debug("Audio detection resumed")
        
    def reset_state(self):
        """Reset buffer state for fresh detection"""
        self.buffer.clear()
        self.speech_detected = False
        self.is_recording = False
        self.silence_counter = 0
        self.recording_chunks = 0
        self.speech_chunks = 0
        self.consecutive_silence = 0
        self.speech_energy_history.clear()
        self.last_reset_time = time.time()
        
        if self.debug:
            logger.debug("Audio buffer state reset")
    
    def add_chunk(self, chunk: np.ndarray, config: AudioConfig) -> bool:
        """Enhanced chunk processing with better responsiveness"""
        if self.is_paused:
            return False
            
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
        self.speech_energy_history.append(rms)
        
        # Enhanced background noise estimation
        if not self.speech_detected and len(self.recent_rms) >= 5:  # Faster adaptation
            # Use 75th percentile for more robust background estimation
            sorted_rms = sorted(list(self.recent_rms))
            self.background_rms = sorted_rms[int(len(sorted_rms) * 0.75)]
        
        # Dynamic threshold calculation
        if self.background_rms > 0:
            # More aggressive speech detection for better responsiveness
            speech_threshold = max(config.silence_threshold, self.background_rms * 3.0)  # Reduced from 4.0
            silence_threshold = max(config.silence_threshold * 0.4, self.background_rms * 1.8)  # Slightly higher
        else:
            speech_threshold = config.silence_threshold * 1.5  # Reduced multiplier
            silence_threshold = config.silence_threshold * 0.5
        
        # Update adaptive silence threshold for quick adaptation
        if rms > speech_threshold:
            self.adaptive_silence_threshold = max(self.adaptive_silence_threshold * 0.95, silence_threshold)
        else:
            self.adaptive_silence_threshold = min(self.adaptive_silence_threshold * 1.02, speech_threshold * 0.8)
        
        # Enhanced speech detection logic
        if not self.speech_detected and rms > speech_threshold:
            # Additional validation: check if this isn't just a noise spike
            recent_energy = list(self.speech_energy_history)[-3:] if len(self.speech_energy_history) >= 3 else [rms]
            avg_recent_energy = np.mean(recent_energy)
            
            if avg_recent_energy > speech_threshold * 0.7:  # More lenient validation
                if self.debug: 
                    logger.info(f"🎤 Speech detected - starting recording (RMS: {rms:.1f} > {speech_threshold:.1f})")
                self.speech_detected = True
                self.is_recording = True
                self.silence_counter = 0
                self.consecutive_silence = 0
                self.recording_chunks = 0
                self.speech_chunks = 0
                
        elif self.speech_detected:
            if rms > self.adaptive_silence_threshold:
                # Still speech or above adaptive threshold
                self.silence_counter = 0
                self.consecutive_silence = 0
                if rms > speech_threshold * 0.8:  # Count as strong speech
                    self.speech_chunks += 1
            else:
                # Below silence threshold
                self.silence_counter += 1
                self.consecutive_silence += 1
                
                # Enhanced completion criteria
                silence_threshold_chunks = config.silence_duration * config.sample_rate / config.chunk_size
                min_silence_chunks = max(5, silence_threshold_chunks * 0.6)  # Minimum silence needed
                
                if self.debug and self.silence_counter % 3 == 0:
                    logger.debug(f"Silence: {self.silence_counter}/{silence_threshold_chunks:.1f}, Speech chunks: {self.speech_chunks}")
                
                # Complete utterance if sufficient silence and speech
                should_complete = (
                    self.silence_counter > min_silence_chunks and 
                    self.speech_chunks >= self.min_speech_chunks and
                    self.consecutive_silence > 3  # Ensure sustained silence
                )
                
                if should_complete:
                    logger.info(f"✅ Speech complete - {self.speech_chunks} speech chunks, {self.silence_counter} silence chunks")
                    return True
        
        # Enhanced timeout handling
        if self.is_recording:
            self.recording_chunks += 1
            max_recording_chunks = config.max_recording_time * config.sample_rate / config.chunk_size
            
            if self.recording_chunks > max_recording_chunks:
                if self.speech_chunks >= 2:  # More lenient minimum
                    logger.warning(f"Maximum recording time reached - completing with {self.speech_chunks} speech chunks")
                    return True
                else:
                    logger.warning("Maximum recording time reached with minimal speech - resetting")
                    self.reset_state()
                    return False
            
            # Only add to buffer if recording
            self.buffer.extend(chunk)
            
        return False

    def get_audio_data(self) -> np.ndarray:
        """Get the complete audio data and reset buffer"""
        if not self.buffer:
            return np.array([])
        
        data = np.array(list(self.buffer))
        
        # Reset state for next detection
        self.reset_state()
        
        return data


class WhisperSTT:
    """Enhanced Whisper-based STT implementation"""

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
        """Enhanced transcription with better filtering"""
        if not self.available:
            return "Whisper not available"
        
        if len(audio_data) == 0:
            return ""
        
        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Enhanced Whisper parameters for better responsiveness
            result = self.model.transcribe(
                audio_float, 
                language="en",
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=False,  # Disable for faster processing
                verbose=False,
                fp16=False  # Disable FP16 for stability
            )
            
            text = result.get("text", "").strip()
            
            # Enhanced filtering for better quality
            if len(text) > 1 and not text in [".", "..", "...", "you", "thank you"]:
                # Additional filter for common false positives
                false_positives = ["mm-hmm", "hmm", "uh", "um", "ah", "oh"]
                if text.lower() not in false_positives and len(text.split()) >= 1:
                    return text
            
            return ""
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""


class VoskSTT:
    """Vosk-based STT implementation - The lightweight speedster"""

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
    """Enhanced wake word detection with better filtering"""

    def __init__(self, wake_words: list = None):
        if wake_words is None:
            config = get_config()
            wake_words = config.get_wake_words()
            
        self.wake_words = [word.lower() for word in wake_words]
        self.last_detection = 0
        self.cooldown = 1.5  # Reduced cooldown for better responsiveness

    def detect(self, text: str) -> bool:
        """Enhanced wake word detection"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        current_time = time.time()
        
        # Check cooldown to prevent spam
        if current_time - self.last_detection < self.cooldown:
            return False
        
        # Enhanced wake word matching
        for wake_word in self.wake_words:
            # Direct match
            if wake_word in text_lower:
                self.last_detection = current_time
                logger.info(f"Wake word detected: {wake_word} in '{text}'")
                return True
            
            # Fuzzy matching for partial words
            words = text_lower.split()
            for word in words:
                # Check if wake word is close to any word in the transcription
                if wake_word in word or word in wake_word:
                    if len(word) >= 3:  # Avoid matching very short words
                        self.last_detection = current_time
                        logger.info(f"Wake word detected (fuzzy): {wake_word} ~ {word}")
                        return True
        
        return False


class JarvisSTT:
    """Enhanced STT coordinator with improved responsiveness and all modern features"""

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
        
        # Enhanced processing queue for non-blocking operation
        self.processing_queue = queue.Queue(maxsize=3)  # Limit queue size
        self.processing_thread = None
        self.processing_thread_running = False
        
        # Audio stream state management
        self.stream_lock = threading.RLock()
        self.stream = None
        self.audio = None
        
        # Initialize STT engine with performance mode
        if stt_engine.lower() == "whisper":
            self.stt_engine = WhisperSTT(model_name, performance_mode)
        elif stt_engine.lower() == "vosk":
            self.stt_engine = VoskSTT(model_name)
        else:
            raise ValueError(f"Unknown STT engine: {stt_engine}")

        logger.info(f"Enhanced STT Engine loaded with performance mode: {performance_mode or 'default'}")
        
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
        """Enhanced start listening with better stream management"""
        with self.stream_lock:
            if self.is_listening:
                logger.warning("Already listening")
                return
            
            try:
                # Initialize PyAudio if needed
                if self.audio is None:
                    self.audio = pyaudio.PyAudio()
                
                # Start processing thread
                self._start_processing_thread()
                
                self.is_listening = True
                
                # Open audio stream with enhanced settings
                self.stream = self.audio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=self._audio_callback,
                    start=False  # Don't start immediately
                )
                
                # Reset buffer state
                self.audio_buffer.reset_state()
                
                # Start the stream
                self.stream.start_stream()
                logger.info("Enhanced audio listening started")
                
            except Exception as e:
                logger.error(f"Failed to start listening: {e}")
                self.is_listening = False
                raise

    def stop_listening(self):
        """Enhanced stop listening with proper cleanup"""
        with self.stream_lock:
            if not self.is_listening:
                return
            
            self.is_listening = False
            
            try:
                # Stop processing thread
                self._stop_processing_thread()
                
                # Stop and close stream
                if self.stream:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                
                logger.info("Enhanced audio listening stopped")
                
            except Exception as e:
                logger.error(f"Error stopping listening: {e}")

    def pause_for_speech(self):
        """Pause speech detection during TTS playback"""
        logger.debug("Pausing speech detection for TTS")
        self.audio_buffer.pause_detection()

    def resume_after_speech(self):
        """Resume speech detection after TTS playback"""
        logger.debug("Resuming speech detection after TTS")
        self.audio_buffer.resume_detection()

    def _start_processing_thread(self):
        """Start the processing thread"""
        if not self.processing_thread_running:
            self.processing_thread_running = True
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()
            logger.debug("Processing thread started")

    def _stop_processing_thread(self):
        """Stop the processing thread"""
        if self.processing_thread_running:
            self.processing_thread_running = False
            # Add sentinel to wake up the thread
            try:
                self.processing_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Clear any remaining items
            while not self.processing_queue.empty():
                try:
                    self.processing_queue.get_nowait()
                except queue.Empty:
                    break
                    
            logger.debug("Processing thread stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Enhanced audio callback with better error handling"""
        if not self.is_listening:
            return (None, pyaudio.paComplete)
        
        try:
            # Convert audio data to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            
            # Add to buffer and check for complete utterance
            utterance_complete = self.audio_buffer.add_chunk(audio_chunk, self.config)
            
            if utterance_complete:
                # Get audio data immediately
                audio_data = self.audio_buffer.get_audio_data()
                
                # Queue for processing (non-blocking)
                try:
                    self.processing_queue.put(audio_data, block=False)
                    if self.debug:
                        logger.debug("Audio queued for processing")
                except queue.Full:
                    logger.warning("Processing queue full, dropping audio")
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)

    def _processing_worker(self):
        """Enhanced processing worker thread"""
        logger.debug("Processing worker started")
        
        while self.processing_thread_running:
            try:
                # Get audio data from queue
                audio_data = self.processing_queue.get(timeout=1.0)
                
                if audio_data is None:  # Sentinel for shutdown
                    break
                
                if len(audio_data) > 0:
                    # Transcribe audio
                    transcription = self.stt_engine.transcribe(audio_data, self.config)
                    
                    if transcription:
                        logger.info(f"Transcribed: '{transcription}'")
                        
                        # Call speech callback (wake word detection handled at assistant level)
                        if self.on_speech_callback:
                            try:
                                self.on_speech_callback(transcription)
                            except Exception as e:
                                logger.error(f"Speech callback error: {e}")
                        
                        # Only call wake word callback if no speech callback is set (for backwards compatibility)
                        elif self.on_wake_word_callback and self.wake_detector.detect(transcription):
                            try:
                                self.on_wake_word_callback()
                            except Exception as e:
                                logger.error(f"Wake word callback error: {e}")
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
        
        logger.debug("Processing worker stopped")

    def listen_and_transcribe(self, timeout: float = 10.0) -> str:
        """Enhanced listen and transcribe with better state management"""
        if self.debug: 
            logger.info(f"Starting listen_and_transcribe with {timeout}s timeout")
        
        # Create a temporary enhanced buffer for this session
        temp_buffer = AudioBuffer(debug=self.debug)
        transcription = ""
        
        try:
            # Use existing audio instance or create new one
            if self.audio is None:
                audio = pyaudio.PyAudio()
            else:
                audio = self.audio
            
            # Open audio stream for recording
            stream = audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            if self.debug: 
                logger.info("Listening for speech...")
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
            
            # Only close audio if we created it
            if self.audio is None:
                audio.terminate()
            
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
        """Enhanced cleanup"""
        try:
            self.stop_listening()
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
        except:
            pass


# Backward compatibility aliases
EnhancedJarvisSTT = JarvisSTT  # For any code that might still reference the old enhanced class


# Example usage and testing
if __name__ == "__main__":
    def on_speech(text: str):
        print(f"Speech detected: {text}")

    def on_wake_word():
        print("Wake word detected! Jarvis is listening...")

    # Create STT instance
    jarvis = JarvisSTT(stt_engine="whisper", model_name="base", debug=True)

    # Set callbacks
    jarvis.set_speech_callback(on_speech)
    jarvis.set_wake_word_callback(on_wake_word)

    # Start listening
    try:
        jarvis.start_listening()
        print("Enhanced STT listening... Say 'Jarvis' to activate. Press Ctrl+C to stop.")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        jarvis.stop_listening()

    print("Enhanced Jarvis STT pipeline stopped.")