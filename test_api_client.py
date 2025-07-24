#!/usr/bin/env python3
"""
Jarvis API Test Client
Simple client to test the Jarvis REST API functionality
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class JarvisAPIClient:
    """Simple client for Jarvis API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def health(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def status(self) -> Dict[str, Any]:
        """Get system status"""
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def chat(self, text: str, use_tts: bool = False, ai_provider: Optional[str] = None) -> Dict[str, Any]:
        """Chat with Jarvis"""
        payload = {
            "text": text,
            "use_tts": use_tts,
            "ai_provider": ai_provider
        }
        
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_providers(self) -> Dict[str, Any]:
        """Get available AI providers"""
        response = self.session.get(f"{self.base_url}/providers")
        response.raise_for_status()
        return response.json()
    
    def download_audio(self, audio_url: str, filename: str):
        """Download audio file"""
        if audio_url.startswith('/'):
            audio_url = self.base_url + audio_url
        
        response = self.session.get(audio_url)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Audio saved to: {filename}")

def main():
    """Test the API"""
    print("🤖 Jarvis API Test Client")
    print("=" * 40)
    
    client = JarvisAPIClient()
    
    try:
        # Test health
        print("\n📊 Health Check:")
        health = client.health()
        print(f"   Overall: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
        
        # Test status
        print("\n📈 System Status:")
        status = client.status()
        print(f"   Version: {status['version']}")
        print(f"   Status: {status['status']}")
        print(f"   Uptime: {status['uptime']:.1f}s")
        print(f"   Local Mode: {status['local_mode']}")
        
        # Test providers
        print("\n🧠 AI Providers:")
        providers = client.get_providers()
        for name, info in providers.items():
            status_icon = "✅" if info['healthy'] else "❌"
            primary_icon = "⭐" if info['is_primary'] else ""
            print(f"   {status_icon} {name}: {info['name']} {primary_icon}")
        
        # Test chat with local AI
        print("\n💬 Testing Local AI Chat:")
        response = client.chat(
            "Hello Jarvis, tell me a quick joke about programming", 
            ai_provider="local"
        )
        print(f"   Request ID: {response['request_id']}")
        print(f"   Provider: {response['provider_used']}")
        print(f"   Processing Time: {response['processing_time']:.2f}s")
        print(f"   Response: {response['response']}")
        
        # Test chat with TTS
        print("\n🎙️ Testing TTS Generation:")
        response = client.chat(
            "Good afternoon, sir. This is a test of the TTS system.",
            use_tts=True,
            ai_provider="local"
        )
        print(f"   Audio URL: {response['audio_url']}")
        
        if response['audio_url']:
            # Download the audio file
            client.download_audio(response['audio_url'], "test_response.wav")
        
        print("\n✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Jarvis API server.")
        print("   Make sure the server is running with: python jarvis_api.py")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response.content:
            try:
                error_data = e.response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()