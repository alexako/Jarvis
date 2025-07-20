#!/usr/bin/env python3

import time
import datetime
import threading
import logging
import numpy as np
from speech_analysis.stt import JarvisSTT, AudioConfig

class LiveTranscriber:
    def __init__(self, chunk_duration=3.0, log_file='stt_transcription_log.txt'):
        self.jarvis = JarvisSTT()
        self.chunk_duration = chunk_duration  # seconds to record before transcribing
        self.running = False
        self.transcription_count = 0
        self.wake_word_count = 0
        self.start_time = None
        self.log_file = log_file
        
        # Set up logging to file and console
        self.logger = logging.getLogger('live_transcriber')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def transcribe_chunk(self, audio_data):
        """Transcribe a chunk of audio data"""
        if len(audio_data) == 0:
            return ""
            
        try:
            # Transcribe using Whisper
            transcription = self.jarvis.stt_engine.transcribe(audio_data, self.jarvis.config)
            return transcription.strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def process_transcription(self, text):
        """Process and display transcription results"""
        if not text:
            return
            
        self.transcription_count += 1
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Log transcription to file and display
        transcription_msg = f"TRANSCRIPTION #{self.transcription_count}: '{text}'"
        self.logger.info(transcription_msg)
        
        # Display transcription
        print(f"\n🎤 [{timestamp}] #{self.transcription_count}: '{text}'")
        
        # Check for wake word
        if self.jarvis.wake_detector.detect(text):
            self.wake_word_count += 1
            wake_msg = f"WAKE WORD DETECTED! (#{self.wake_word_count}) in: '{text}'"
            self.logger.warning(wake_msg)
            print(f"   🚨 WAKE WORD DETECTED! (#{self.wake_word_count})")
        
        # Show and log word count
        word_count = len(text.split())
        word_count_msg = f"Word count: {word_count} words"
        self.logger.info(word_count_msg)
        print(f"   📊 {word_count} words")
    
    def live_transcription_loop(self):
        """Main live transcription loop"""
        print(f"🔄 Starting live transcription (chunks of {self.chunk_duration}s)...")
        
        while self.running:
            try:
                # Record for the specified duration
                print(f"\n🎙️  Recording for {self.chunk_duration}s... (Speak now!)")
                
                # Clear any existing buffer and start fresh
                self.jarvis.audio_buffer.buffer.clear()
                self.jarvis.audio_buffer.is_recording = True
                
                # Record for the chunk duration
                start_record = time.time()
                while time.time() - start_record < self.chunk_duration and self.running:
                    time.sleep(0.1)
                
                # Get the recorded audio
                audio_data = self.jarvis.audio_buffer.get_audio_data()
                
                if len(audio_data) > 100:  # Only transcribe if we have significant audio
                    print(f"   📊 Processing {len(audio_data)} audio samples...")
                    transcription = self.transcribe_chunk(audio_data)
                    self.process_transcription(transcription)
                else:
                    print("   🔇 No significant audio detected")
                    
            except Exception as e:
                print(f"❌ Error in transcription loop: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the live transcription system"""
        print("\n" + "="*60)
        print("🎙️  LIVE TRANSCRIPTION SYSTEM")
        print("="*60)
        print(f"⏱️  Recording in {self.chunk_duration}s chunks")
        print("🎯 Say 'Jarvis' or 'Hey Jarvis' for wake word detection")
        print("⏹️  Press Ctrl+C to stop")
        print(f"📝 Logging to: {self.log_file}")
        print("="*60)
        
        self.start_time = datetime.datetime.now()
        self.running = True
        
        # Log session start
        session_start_msg = f"=== LIVE TRANSCRIPTION SESSION STARTED at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
        self.logger.info(session_start_msg)
        
        try:
            # Start audio listening
            self.jarvis.start_listening()
            print(f"\n✅ Started at {self.start_time.strftime('%H:%M:%S')}")
            
            # Run the transcription loop
            self.live_transcription_loop()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping live transcription...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the live transcription system"""
        self.running = False
        self.jarvis.stop_listening()
        
        if self.start_time:
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            
            # Log session summary
            summary_msg = f"=== SESSION SUMMARY: Duration: {duration}, Transcriptions: {self.transcription_count}, Wake words: {self.wake_word_count} ==="
            self.logger.info(summary_msg)
            
            session_end_msg = f"=== LIVE TRANSCRIPTION SESSION ENDED at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
            self.logger.info(session_end_msg)
            
            print("\n" + "="*60)
            print("📊 LIVE TRANSCRIPTION SUMMARY")
            print("="*60)
            print(f"⏰ Session duration: {duration}")
            print(f"🎤 Total transcriptions: {self.transcription_count}")
            print(f"🚨 Wake words detected: {self.wake_word_count}")
            if self.transcription_count > 0:
                avg_per_min = (self.transcription_count * 60) / duration.total_seconds()
                print(f"📈 Average transcriptions/min: {avg_per_min:.1f}")
            print(f"📝 Full log saved to: {self.log_file}")
            print("="*60)
        
        print("\n✨ Live transcription completed!")

def main():
    # Create live transcriber with 3-second chunks
    transcriber = LiveTranscriber(chunk_duration=3.0)
    
    # Start live transcription
    transcriber.start()

if __name__ == "__main__":
    main()
