#!/usr/bin/env python3
"""
Speaker Identification Test Script
Simple, effective testing without STT complexity
"""

import argparse
import sys
import os
import numpy as np
import pyaudio
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from speech_analysis.speaker_identification import SpeakerIdentificationSystem
from config_manager import get_config


class SpeakerTester:
    """Simple speaker identification tester"""
    
    def __init__(self):
        self.speaker_system = SpeakerIdentificationSystem()
        
        # Get audio config
        config = get_config()
        audio_config = config.get_audio_config()
        
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 2)
        self.chunk_size = audio_config.get('chunk_size', 1024)
        self.device_index = audio_config.get('input_device_index')
    
    def record_audio(self, duration=3.0):
        """Record audio directly"""
        print(f"🎤 Recording for {duration} seconds...")
        
        audio = pyaudio.PyAudio()
        
        try:
            stream_params = {
                'format': pyaudio.paInt16,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if self.device_index is not None:
                stream_params['input_device_index'] = self.device_index
            
            stream = audio.open(**stream_params)
            
            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * duration)
            
            print("🔴 Recording... speak now!")
            for _ in range(num_chunks):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            print("✅ Recording complete!")
            
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
    
    def enroll_speaker(self, name, num_samples=3, duration=5.0):
        """Enroll a speaker with multiple samples"""
        print(f"\n👤 Enrolling speaker: {name}")
        print(f"Recording {num_samples} samples of {duration} seconds each")
        
        successful_samples = 0
        
        for i in range(num_samples):
            input(f"\nPress Enter when ready for sample {i+1}/{num_samples}...")
            
            audio_data = self.record_audio(duration)
            if audio_data is not None and len(audio_data) > 0:
                if self.speaker_system.enroll_user(name, name, audio_data, self.sample_rate):
                    successful_samples += 1
                    print(f"✅ Sample {i+1} enrolled successfully")
                else:
                    print(f"⚠️  Sample {i+1} enrollment failed")
            else:
                print(f"❌ Sample {i+1} recording failed")
        
        if successful_samples > 0:
            print(f"\n🎉 Enrollment complete! {successful_samples}/{num_samples} samples enrolled for {name}")
            return True
        else:
            print(f"\n❌ Enrollment failed - no samples recorded")
            return False
    
    def test_identification(self, num_tests=3):
        """Test speaker identification"""
        users = self.speaker_system.list_enrolled_users()
        if not users:
            print("❌ No speakers enrolled. Use --enroll first.")
            return False
        
        print(f"👥 Enrolled users:")
        for user in users:
            name = user.get('name', 'Unknown')
            samples = user.get('samples', 0)
            print(f"  • {name} ({samples} samples)")
        
        print(f"\n🧪 Testing speaker identification...")
        
        correct_identifications = 0
        
        for i in range(num_tests):
            input(f"\nPress Enter for test {i+1}/{num_tests}...")
            
            audio_data = self.record_audio(3.0)
            
            if audio_data is not None and len(audio_data) > 0:
                print("🔍 Analyzing voice...")
                result = self.speaker_system.identify_speaker(audio_data, self.sample_rate)
                
                if result.user_id:
                    print(f"✅ Identified: {result.user_id} (confidence: {result.confidence:.2f})")
                    correct_identifications += 1
                else:
                    print(f"❓ Unknown speaker (best confidence: {result.confidence:.2f})")
            else:
                print("❌ No audio recorded")
        
        success_rate = (correct_identifications / num_tests) * 100
        print(f"\n📊 Test Results: {correct_identifications}/{num_tests} successful identifications ({success_rate:.1f}%)")
        
        return success_rate > 50
    
    def list_users(self):
        """List all enrolled users"""
        users = self.speaker_system.list_enrolled_users()
        
        if not users:
            print("📭 No speakers enrolled")
            return
        
        print(f"👥 Enrolled speakers ({len(users)}):")
        for user in users:
            name = user.get('name', 'Unknown')
            user_id = user.get('user_id', 'Unknown')
            samples = user.get('samples', 0)
            last_seen = user.get('last_seen', 'Never')
            
            print(f"🎤 {name}")
            print(f"   ID: {user_id}")
            print(f"   Samples: {samples}")
            print(f"   Last seen: {last_seen}")
            print()
    
    def delete_user(self, user_id):
        """Delete a user profile"""
        if self.speaker_system.delete_user(user_id):
            print(f"✅ Deleted user: {user_id}")
            return True
        else:
            print(f"❌ Failed to delete user: {user_id}")
            return False
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        try:
            stats = self.speaker_system.get_user_stats(user_id)
            if stats:
                print(f"📊 Statistics for {user_id}:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ No statistics found for user: {user_id}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the speaker identification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_speaker_identification.py --enroll "Alex"
    python test_speaker_identification.py --test
    python test_speaker_identification.py --list
    python test_speaker_identification.py --delete alex
        """
    )
    
    parser.add_argument('--enroll', type=str, metavar='NAME', 
                       help='Enroll a new speaker with the given name')
    parser.add_argument('--test', action='store_true', 
                       help='Test speaker identification')
    parser.add_argument('--list', action='store_true', 
                       help='List all enrolled users')
    parser.add_argument('--delete', type=str, metavar='USER_ID', 
                       help='Delete a user profile')
    parser.add_argument('--stats', type=str, metavar='USER_ID', 
                       help='Show statistics for a user')
    
    args = parser.parse_args()
    
    if not any([args.enroll, args.test, args.list, args.delete, args.stats]):
        parser.print_help()
        return 1
    
    try:
        tester = SpeakerTester()
        
        if args.enroll:
            success = tester.enroll_speaker(args.enroll)
            return 0 if success else 1
        
        elif args.test:
            success = tester.test_identification()
            return 0 if success else 1
        
        elif args.list:
            tester.list_users()
            return 0
        
        elif args.delete:
            success = tester.delete_user(args.delete)
            return 0 if success else 1
        
        elif args.stats:
            tester.get_user_stats(args.stats)
            return 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())