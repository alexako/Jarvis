#!/usr/bin/env python3
"""
Test script for Jarvis Audio Streaming API
Tests the new audio streaming endpoints to ensure proper functionality
"""

import asyncio
import aiohttp
import time
import json
import wave
import tempfile
import os
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://127.0.0.1:8000"
TEST_TEXT = "Hello! This is Jarvis testing the new audio streaming capability. The streaming audio should play smoothly in real-time chunks."

class AudioStreamingTester:
    """Test class for audio streaming functionality"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test if the API server is running and healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Health check passed: {health_data['healthy']}")
                    print(f"   Components: {health_data['components']}")
                    return health_data['healthy']
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_regular_chat(self) -> Dict[str, Any]:
        """Test regular chat endpoint without streaming"""
        print("\n🧪 Testing regular chat endpoint...")
        
        payload = {
            "text": TEST_TEXT,
            "use_tts": True,
            "stream_audio": False
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Regular chat successful")
                    print(f"   Response: {data['response'][:100]}...")
                    print(f"   Provider: {data['provider_used']}")
                    print(f"   Processing time: {data['processing_time']:.2f}s")
                    print(f"   Audio URL: {data.get('audio_url', 'None')}")
                    return data
                else:
                    print(f"❌ Regular chat failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                    return {}
        except Exception as e:
            print(f"❌ Regular chat error: {e}")
            return {}
    
    async def test_streaming_chat(self) -> Dict[str, Any]:
        """Test chat endpoint with streaming audio"""
        print("\n🧪 Testing streaming chat endpoint...")
        
        payload = {
            "text": TEST_TEXT,
            "use_tts": True,
            "stream_audio": True
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Streaming chat successful")
                    print(f"   Response: {data['response'][:100]}...")
                    print(f"   Provider: {data['provider_used']}")
                    print(f"   Processing time: {data['processing_time']:.2f}s")
                    print(f"   Stream URL: {data.get('stream_url', 'None')}")
                    return data
                else:
                    print(f"❌ Streaming chat failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                    return {}
        except Exception as e:
            print(f"❌ Streaming chat error: {e}")
            return {}
    
    async def test_audio_stream(self, stream_url: str) -> bool:
        """Test downloading audio stream"""
        print(f"\n🧪 Testing audio stream download from: {stream_url}")
        
        try:
            stream_url_full = f"{self.base_url}{stream_url}"
            
            # Create temporary file to save audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            bytes_received = 0
            chunks_received = 0
            start_time = time.time()
            
            async with self.session.get(stream_url_full) as response:
                if response.status == 200:
                    print(f"   Content-Type: {response.content_type}")
                    
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(4096):
                            f.write(chunk)
                            bytes_received += len(chunk)
                            chunks_received += 1
                            
                            # Print progress every 10 chunks
                            if chunks_received % 10 == 0:
                                elapsed = time.time() - start_time
                                print(f"   Received {chunks_received} chunks, {bytes_received} bytes in {elapsed:.2f}s")
                    
                    # Verify the audio file
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        try:
                            # Try to open as WAV file to verify format
                            with wave.open(temp_path, 'rb') as wav_file:
                                frames = wav_file.getnframes()
                                sample_rate = wav_file.getframerate()
                                duration = frames / sample_rate
                                
                                print(f"✅ Audio stream successful")
                                print(f"   Total bytes: {bytes_received}")
                                print(f"   Total chunks: {chunks_received}")
                                print(f"   Audio duration: {duration:.2f}s")
                                print(f"   Sample rate: {sample_rate}Hz")
                                
                                # Clean up temp file
                                os.unlink(temp_path)
                                return True
                        except Exception as e:
                            print(f"⚠️  Audio verification failed: {e}")
                            print(f"   File size: {os.path.getsize(temp_path)} bytes")
                            os.unlink(temp_path)
                            return False
                    else:
                        print(f"❌ No audio data received")
                        return False
                else:
                    print(f"❌ Audio stream failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
            return False
    
    async def test_direct_audio_stream(self) -> bool:
        """Test direct audio streaming endpoint"""
        print("\n🧪 Testing direct audio streaming endpoint...")
        
        payload = {
            "text": "This is a direct audio streaming test.",
            "chunk_size": 4096,
            "format": "wav",
            "quality": "medium"
        }
        
        try:
            bytes_received = 0
            chunks_received = 0
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/audio/stream", 
                json=payload
            ) as response:
                if response.status == 200:
                    print(f"   Content-Type: {response.content_type}")
                    print(f"   Request-ID: {response.headers.get('X-Request-ID', 'None')}")
                    
                    async for chunk in response.content.iter_chunked(4096):
                        bytes_received += len(chunk)
                        chunks_received += 1
                        
                        # Print progress every 5 chunks
                        if chunks_received % 5 == 0:
                            elapsed = time.time() - start_time
                            print(f"   Received {chunks_received} chunks, {bytes_received} bytes in {elapsed:.2f}s")
                    
                    print(f"✅ Direct audio stream successful")
                    print(f"   Total bytes: {bytes_received}")
                    print(f"   Total chunks: {chunks_received}")
                    return True
                else:
                    print(f"❌ Direct audio stream failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Direct audio stream error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all audio streaming tests"""
        print("🚀 Starting Jarvis Audio Streaming Tests")
        print("=" * 60)
        
        results = {}
        
        # 1. Health check
        results['health'] = await self.test_health_check()
        if not results['health']:
            print("\n❌ Server not healthy, skipping remaining tests")
            return results
        
        # 2. Regular chat test
        chat_data = await self.test_regular_chat()
        results['regular_chat'] = bool(chat_data)
        
        # 3. Streaming chat test
        streaming_data = await self.test_streaming_chat()
        results['streaming_chat'] = bool(streaming_data)
        
        # 4. Test audio stream if we got a stream URL
        if streaming_data and streaming_data.get('stream_url'):
            results['audio_stream'] = await self.test_audio_stream(streaming_data['stream_url'])
        else:
            results['audio_stream'] = False
            print("\n⚠️  No stream URL to test")
        
        # 5. Direct audio streaming test
        results['direct_stream'] = await self.test_direct_audio_stream()
        
        return results

async def main():
    """Main test runner"""
    print("Jarvis Audio Streaming API Test Suite")
    print("=====================================")
    
    async with AudioStreamingTester() as tester:
        results = await tester.run_all_tests()
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<20} {status}")
        
        print("-" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! Audio streaming is working correctly.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check the logs above for details.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Tests cancelled by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")