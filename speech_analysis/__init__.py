"""
Jarvis Speech Analysis Module

This module provides speech-to-text and text-to-speech capabilities using Whisper, Vosk,
pyttsx3, and Coqui TTS engines, along with wake word detection and voice activity detection.
"""

from .stt import JarvisSTT, WhisperSTT, VoskSTT, AudioConfig, AudioBuffer, WakeWordDetector
from .tts import JarvisTTS, PyttsxTTS, CoquiTTS, TTSConfig, JarvisPersonality, AudioPlayer
from .improved_stt import EnhancedJarvisSTT

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
    'CoquiTTS',
    'TTSConfig',
    'JarvisPersonality',
    'AudioPlayer'
]
