"""
Jarvis Speech Analysis Module

This module provides speech-to-text and text-to-speech capabilities using Whisper, Vosk,
pyttsx3, and Coqui TTS engines, along with wake word detection and voice activity detection.
"""

from .stt import JarvisSTT, WhisperSTT, VoskSTT, AudioConfig, AudioBuffer, WakeWordDetector, EnhancedJarvisSTT
from .tts import JarvisTTS, PyttsxTTS, PiperTTS, CoquiTTS, TTSConfig, JarvisPersonality, AudioPlayer
from .speaker_identification import SpeakerIdentificationSystem, SpeakerProfile, IdentificationResult

__all__ = [
    'JarvisSTT',
    'WhisperSTT', 
    'VoskSTT',
    'AudioConfig',
    'AudioBuffer',
    'WakeWordDetector',
    'EnhancedJarvisSTT',
    'JarvisTTS',
    'PyttsxTTS',
    'PiperTTS',
    'CoquiTTS',
    'TTSConfig',
    'JarvisPersonality',
    'AudioPlayer',
    'SpeakerIdentificationSystem',
    'SpeakerProfile',
    'IdentificationResult'
]
