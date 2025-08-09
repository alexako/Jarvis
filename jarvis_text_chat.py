#!/usr/bin/env python3
"""
Jarvis Interactive Text Chat Client
Simple terminal-based chat interface for Jarvis with enhanced memory support
"""

import requests
import json
import sys
import time
from typing import Optional, Dict, Any
from datetime import datetime

class JarvisTextChat:
    """Interactive text chat client for Jarvis"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.chat_history = []
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def check_server(self) -> bool:
        """Check if the Jarvis API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except:
            return {}
    
    def chat(self, text: str, ai_provider: Optional[str] = None, user: Optional[str] = None) -> Dict[str, Any]:
        """Send chat message to Jarvis"""
        payload = {
            "text": text,
            "use_tts": False,
            "ai_provider": ai_provider,
            "user": user
        }
        
        try:
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                return {"response": "Service temporarily unavailable. Please wait a moment and try again.", "error": True}
            return {"response": f"Error: {e}", "error": True}
        except Exception as e:
            return {"response": f"Error: {e}", "error": True}
    
    def get_providers(self) -> Dict[str, Any]:
        """Get available AI providers"""
        try:
            response = self.session.get(f"{self.base_url}/providers")
            response.raise_for_status()
            return response.json()
        except:
            return {}
    
    def print_header(self):
        """Print chat header"""
        print("\n" + "="*60)
        print("🤖 JARVIS INTERACTIVE TEXT CHAT")
        print("="*60)
        print("Your AI assistant with enhanced memory capabilities")
        print("\nCommands:")
        print("  /help     - Show help")
        print("  /status   - Show system status")
        print("  /provider - Switch AI provider")
        print("  /user     - Switch user profile")
        print("  /memory   - Show memory stats")
        print("  /clear    - Clear screen")
        print("  /exit     - Exit chat")
        print("\nMemory Examples:")
        print("  'Remember that I'm allergic to peanuts'")
        print("  'My birthday is January 15th'")
        print("  'What am I allergic to?'")
        print("="*60 + "\n")
    
    def print_help(self):
        """Print help information"""
        print("\n📚 HELP")
        print("-" * 40)
        print("Chat naturally with Jarvis. He remembers:")
        print("• Personal information (allergies, birthday, etc.)")
        print("• Preferences (coffee order, favorite colors)")
        print("• Relationships (family, friends, colleagues)")
        print("• Work information (job, company, skills)")
        print("• Daily routines and schedules")
        print("\nExample inputs:")
        print("• 'What time is it?'")
        print("• 'Tell me a joke'")
        print("• 'Explain quantum computing'")
        print("• 'Remember that my wife's name is Sarah'")
        print("• 'What's my wife's name?'")
        print("-" * 40)
    
    def select_provider(self):
        """Let user select AI provider"""
        providers = self.get_providers()
        if not providers:
            print("❌ Could not fetch providers")
            return None
        
        print("\n🧠 Available AI Providers:")
        provider_list = list(providers.keys())
        for i, (name, info) in enumerate(providers.items(), 1):
            status = "✅" if info['healthy'] else "❌"
            primary = " (current)" if info['is_primary'] else ""
            print(f"  {i}. {status} {info['name']}{primary}")
        
        print("\nSelect provider (1-{}, or Enter to keep current): ".format(len(provider_list)), end="")
        choice = input().strip()
        
        if not choice:
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(provider_list):
                selected = provider_list[idx]
                print(f"✅ Switched to {providers[selected]['name']}")
                return selected
        except:
            pass
        
        print("❌ Invalid selection")
        return None
    
    def show_status(self):
        """Show system status"""
        status = self.get_status()
        if not status:
            print("❌ Could not fetch status")
            return
        
        providers = self.get_providers()
        
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        print(f"Version: {status.get('version', 'Unknown')}")
        print(f"Status: {status.get('status', 'Unknown')}")
        print(f"Uptime: {status.get('uptime', 0):.1f} seconds")
        print(f"TTS Engine: {status.get('tts_engine', 'Unknown')}")
        print(f"Local Mode: {status.get('local_mode', False)}")
        
        if providers:
            print("\nAI Providers:")
            for name, info in providers.items():
                status_icon = "✅" if info['healthy'] else "❌"
                primary = " ⭐" if info['is_primary'] else ""
                print(f"  {status_icon} {info['name']}{primary}")
        print("-" * 40)
    
    def clear_screen(self):
        """Clear the terminal screen"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        self.print_header()
    
    def run(self):
        """Run the interactive chat"""
        # Check if server is running
        if not self.check_server():
            print("❌ Jarvis API server is not running!")
            print("\nPlease start the server first:")
            print("  python jarvis_api.py")
            print("\nOr for API-only mode:")
            print("  python jarvis_api.py --api-only")
            return
        
        # Print header
        self.print_header()
        
        # Get initial status
        status = self.get_status()
        if status:
            print(f"✅ Connected to Jarvis v{status.get('version', 'Unknown')}")
            providers = self.get_providers()
            if providers:
                primary = next((info['name'] for name, info in providers.items() if info['is_primary']), 'Unknown')
                print(f"🧠 Using: {primary}")
        
        print("\nType your message or /help for commands\n")
        
        # Main chat loop
        current_provider = None
        current_user = None
        
        while True:
            try:
                # Get user input
                print("You: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower().split()[0]
                    
                    if command == '/exit' or command == '/quit':
                        print("\n👋 Goodbye!")
                        break
                    
                    elif command == '/help':
                        self.print_help()
                    
                    elif command == '/status':
                        self.show_status()
                    
                    elif command == '/provider':
                        selected = self.select_provider()
                        if selected:
                            current_provider = selected
                    
                    elif command == '/user':
                        print("Enter user name: ", end="")
                        user_name = input().strip()
                        if user_name:
                            current_user = user_name
                            print(f"✅ Switched to user: {user_name}")
                    
                    elif command == '/memory':
                        # Ask Jarvis about memory stats
                        response = self.chat("Show me my memory statistics", current_provider, current_user)
                        if not response.get('error'):
                            print(f"\nJarvis: {response['response']}")
                    
                    elif command == '/clear':
                        self.clear_screen()
                    
                    else:
                        print(f"❌ Unknown command: {command}")
                    
                    continue
                
                # Send message to Jarvis
                print("\n🤔 Thinking...", end="\r")
                start_time = time.time()
                
                response = self.chat(user_input, current_provider, current_user)
                
                elapsed = time.time() - start_time
                
                # Clear the thinking message
                print(" " * 20, end="\r")
                
                if response.get('error'):
                    print(f"❌ {response['response']}")
                else:
                    # Add to history
                    self.chat_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'user': user_input,
                        'jarvis': response['response'],
                        'provider': response.get('provider_used', 'unknown'),
                        'time': elapsed
                    })
                    
                    # Print response
                    print(f"Jarvis: {response['response']}")
                    
                    # Show metadata if interesting
                    if elapsed > 2.0:
                        print(f"       [{response.get('provider_used', 'unknown')} - {elapsed:.1f}s]")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Jarvis Interactive Text Chat')
    parser.add_argument('--url', default='http://127.0.0.1:8000',
                       help='Jarvis API server URL')
    parser.add_argument('--api-key', help='API key for authentication')
    
    args = parser.parse_args()
    
    # Create and run chat client
    chat = JarvisTextChat(base_url=args.url, api_key=args.api_key)
    chat.run()

if __name__ == "__main__":
    main()
