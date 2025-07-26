#!/usr/bin/env python3
"""
Simple Speaker Training Script - Minimal Dependencies
Train speaker recognition with live recording or audio files.
"""

import argparse
import os
import sys
import wave
import time
import numpy as np
import pyaudio

# Add current directory to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speech_analysis.speaker_identification import SpeakerIdentificationSystem
from config_manager import get_config


class SimpleSpeakerTrainer:
    """Simple speaker trainer with minimal dependencies"""
    
    def __init__(self):
        """Initialize trainer"""
        self.config = get_config()
        self.speaker_system = SpeakerIdentificationSystem()
        
        # Audio settings from config
        audio_config = self.config.get_audio_config()
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 2)
        self.chunk_size = audio_config.get('chunk_size', 1024)
        self.device_index = audio_config.get('input_device_index')
        
        print(f"🔧 Audio config: {self.sample_rate}Hz, {self.channels} channels, device {self.device_index}")
    
    def record_audio(self, duration=5.0):
        """Record audio from microphone"""
        print(f"🎤 Recording for {duration} seconds...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Configure stream parameters
            stream_params = {
                'format': pyaudio.paInt16,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if self.device_index is not None:
                stream_params['input_device_index'] = self.device_index
            
            # Open stream
            stream = audio.open(**stream_params)
            
            # Record audio
            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * duration)
            
            print("🔴 Recording... speak now!")
            for _ in range(num_chunks):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            print("✅ Recording complete!")
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert stereo to mono if needed
            if self.channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            return audio_array
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
        finally:
            audio.terminate()
    
    def load_audio_file(self, file_path):
        """Load audio from WAV or MP3 file"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        try:
            file_ext = file_path.lower().split('.')[-1]
            
            if file_ext == 'wav':
                # Load WAV file
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    
                    # Convert stereo to mono if needed
                    if channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    
                    print(f"✅ Loaded {file_path}: {len(audio_data)} samples at {sample_rate}Hz")
                    return audio_data
                    
            elif file_ext == 'mp3':
                # For MP3, we need additional libraries - provide helpful error
                print("❌ MP3 support requires 'pydub' library")
                print("   Install with: pip install pydub")
                print("   Convert to WAV format for simpler usage")
                return None
            else:
                print(f"❌ Unsupported file format: {file_ext}")
                print("   Supported formats: WAV")
                return None
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def train_speaker_live(self, speaker_name, num_samples=3, duration=5.0):
        """Train speaker using live microphone recording"""
        print(f"\n👤 Training speaker: {speaker_name}")
        print(f"Recording {num_samples} samples of {duration} seconds each")
        
        successful_samples = 0
        
        for i in range(num_samples):
            input(f"\nPress Enter when ready for sample {i+1}/{num_samples}...")
            
            audio_data = self.record_audio(duration)
            if audio_data is not None and len(audio_data) > 0:
                # Enroll the sample
                if self.speaker_system.enroll_user(speaker_name, speaker_name, audio_data, self.sample_rate):
                    successful_samples += 1
                    print(f"✅ Sample {i+1} enrolled successfully")
                else:
                    print(f"⚠️  Sample {i+1} enrollment failed")
            else:
                print(f"❌ Sample {i+1} recording failed")
        
        if successful_samples > 0:
            print(f"\n🎉 Training complete! Enrolled {successful_samples}/{num_samples} samples for {speaker_name}")
            return True
        else:
            print(f"\n❌ Training failed - no samples enrolled")
            return False
    
    def train_speaker_from_files(self, speaker_name, file_paths):
        """Train speaker using audio files"""
        print(f"\n👤 Training speaker: {speaker_name}")
        print(f"Processing {len(file_paths)} audio files")
        
        successful_samples = 0
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n📁 Processing file {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            audio_data = self.load_audio_file(file_path)
            if audio_data is not None and len(audio_data) > 0:
                # Enroll the sample
                if self.speaker_system.enroll_user(speaker_name, speaker_name, audio_data, self.sample_rate):
                    successful_samples += 1
                    print(f"✅ File {i} enrolled successfully")
                else:
                    print(f"⚠️  File {i} enrollment failed")
            else:
                print(f"❌ File {i} processing failed")
        
        if successful_samples > 0:
            print(f"\n🎉 Training complete! Enrolled {successful_samples}/{len(file_paths)} files for {speaker_name}")
            return True
        else:
            print(f"\n❌ Training failed - no samples enrolled")
            return False
    
    def list_speakers(self):
        """List all enrolled speakers"""
        try:
            users = self.speaker_system.list_enrolled_users()
            
            if not users:
                print("📭 No speakers enrolled")
                return
            
            print(f"\n👥 Enrolled speakers ({len(users)}):")
            print("-" * 40)
            
            for user in users:
                name = user.get('name', 'Unknown')
                user_id = user.get('user_id', 'Unknown')
                samples = user.get('samples', 0)
                
                print(f"🎤 {name}")
                print(f"   ID: {user_id}")
                print(f"   Samples: {samples}")
                print()
                
        except Exception as e:
            print(f"❌ Error listing speakers: {e}")
    
    def delete_speaker(self, speaker_name):
        """Delete a speaker"""
        try:
            if self.speaker_system.delete_user(speaker_name):
                print(f"✅ Deleted speaker: {speaker_name}")
                return True
            else:
                print(f"❌ Failed to delete speaker: {speaker_name}")
                return False
        except Exception as e:
            print(f"❌ Error deleting speaker: {e}")
            return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Simple Speaker Recognition Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with live recording (3 samples, 5 seconds each)
  python simple_train.py --name Alex --live
  
  # Train with custom recording settings
  python simple_train.py --name Alex --live --samples 5 --duration 3
  
  # Train from audio files
  python simple_train.py --name Alex --files voice1.wav voice2.wav
  
  # List enrolled speakers
  python simple_train.py --list
  
  # Delete a speaker
  python simple_train.py --delete Alex
        """
    )
    
    # Main actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--name', type=str, help='Speaker name to train')
    group.add_argument('--list', action='store_true', help='List enrolled speakers')
    group.add_argument('--delete', type=str, help='Delete a speaker')
    
    # Training method (for --name)
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('--live', action='store_true', help='Use live recording (default)')
    method_group.add_argument('--files', nargs='+', help='Use audio files')
    
    # Recording options
    parser.add_argument('--samples', type=int, default=3, help='Number of samples (default: 3)')
    parser.add_argument('--duration', type=float, default=5.0, help='Sample duration in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    try:
        trainer = SimpleSpeakerTrainer()
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return 1
    
    # Execute command
    try:
        if args.name:
            if args.files:
                success = trainer.train_speaker_from_files(args.name, args.files)
            else:
                # Default to live recording
                success = trainer.train_speaker_live(args.name, args.samples, args.duration)
            return 0 if success else 1
            
        elif args.list:
            trainer.list_speakers()
            return 0
            
        elif args.delete:
            success = trainer.delete_speaker(args.delete)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())