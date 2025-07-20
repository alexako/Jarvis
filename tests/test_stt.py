#!/usr/bin/env python3

import time
import datetime
import logging
from speech_analysis.stt import JarvisSTT

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stt_transcription_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global counters for statistics
transcription_count = 0
wake_word_count = 0
start_time = None

def on_speech(text):
    global transcription_count
    transcription_count += 1
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    # Log and print the transcription
    message = f"[{timestamp}] TRANSCRIPTION #{transcription_count}: '{text}'"
    logger.info(message)
    print(f"\n🎤 {message}")
    
    # Log transcription statistics
    if text.strip():
        word_count = len(text.split())
        logger.info(f"Word count: {word_count} words")
        print(f"   📊 Word count: {word_count} words")

def on_wake_word():
    global wake_word_count
    wake_word_count += 1
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    # Log and print wake word detection
    message = f"[{timestamp}] 🚨 WAKE WORD DETECTED! (#{wake_word_count})"
    logger.warning(message)
    print(f"\n{message}")
    print("   Jarvis is now actively listening...")

# Create STT instance with detailed logging
print("🤖 Initializing Jarvis STT Engine...")
logger.info("Initializing Jarvis STT with Whisper base model")
jarvis = JarvisSTT(stt_engine="whisper", model_name="base")

# Set callbacks
jarvis.set_speech_callback(on_speech)
jarvis.set_wake_word_callback(on_wake_word)

print("\n" + "="*60)
print("🎙️  JARVIS SPEECH TRANSCRIPTION TEST")
print("="*60)
print("📝 All transcriptions will be logged to 'stt_transcription_log.txt'")
print("🎯 Say 'Jarvis' or 'Hey Jarvis' to test wake word detection")
print("⏱️  Test will run continuously - press Ctrl+C to stop")
print("="*60)

start_time = datetime.datetime.now()
logger.info(f"Starting continuous transcription test at {start_time}")

try:
    jarvis.start_listening()
    print(f"\n✅ Started listening at {start_time.strftime('%H:%M:%S')}")
    print("🔊 Speak clearly and I'll transcribe everything...\n")
    
    # Keep running until interrupted
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\n⏹️  Stopping transcription...")
    logger.info("Transcription stopped by user")
finally:
    jarvis.stop_listening()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    # Log final statistics
    print("\n" + "="*60)
    print("📊 TRANSCRIPTION SESSION SUMMARY")
    print("="*60)
    print(f"⏰ Session duration: {duration}")
    print(f"🎤 Total transcriptions: {transcription_count}")
    print(f"🚨 Wake words detected: {wake_word_count}")
    print(f"📝 Log file: stt_transcription_log.txt")
    print("="*60)
    
    logger.info(f"Session ended. Duration: {duration}, Transcriptions: {transcription_count}, Wake words: {wake_word_count}")
    print("\n✨ Transcription test completed!")
