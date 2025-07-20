#!/usr/bin/env python3
"""
Enhanced Jarvis Commands with AI Brain Integration
Modular command system that falls back to AI for unknown requests
"""

import datetime
import logging
import platform
import random
import sys
import os
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the AI brain module
from ai_brain import create_ai_brain, AIBrainManager

# Try to import system info modules
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system info commands will be limited")

logger = logging.getLogger(__name__)

class JarvisCommands:
    """Enhanced command processor with AI brain integration"""
    
    def __init__(self, tts, assistant, ai_config: Optional[Dict[str, Any]] = None):
        self.tts = tts
        self.assistant = assistant
        
        # Initialize AI brain manager
        self.ai_brain: Optional[AIBrainManager] = None
        self.ai_enabled = True
        
        try:
            self.ai_brain = create_ai_brain(ai_config)
            if self.ai_brain.is_available():
                logger.info("AI brain successfully initialized")
            else:
                logger.warning("AI brain initialized but no providers available")
                self.ai_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize AI brain: {e}")
            self.ai_enabled = False
        
        # Built-in command mappings - these run locally for speed/reliability
        self.command_mappings = {
            # Greetings
            'hello': self._handle_greeting,
            'hi': self._handle_greeting,
            'good morning': self._handle_greeting,
            'good afternoon': self._handle_greeting,
            'good evening': self._handle_greeting,
            
            # Time and date
            'time': self._handle_time,
            'what time is it': self._handle_time,
            'date': self._handle_date,
            'what date is it': self._handle_date,
            "what's the date": self._handle_date,
            
            # Status and system info
            'how are you': self._handle_status,
            'status': self._handle_status,
            'system status': self._handle_system_status,
            'battery': self._handle_battery,
            'memory': self._handle_memory,
            'disk space': self._handle_disk_space,
            
            # Help and identity
            'help': self._handle_help,
            'what can you do': self._handle_help,
            'commands': self._handle_help,
            'who are you': self._handle_identity,
            'introduce yourself': self._handle_identity,
            
            # Control commands
            'stop listening': self._handle_stop_listening,
            'shutdown': self._handle_shutdown,
            'goodbye': self._handle_goodbye,
            'bye': self._handle_goodbye,
            'test': self._handle_test,
            
            # AI brain management
            'ai status': self._handle_ai_status,
            'clear history': self._handle_clear_history,
            'disable ai': self._handle_disable_ai,
            'enable ai': self._handle_enable_ai,
        }
        
        logger.info(f"Enhanced commands initialized - AI enabled: {self.ai_enabled}")
    
    def process_command(self, text: str):
        """Process command with AI fallback"""
        if not text or not text.strip():
            return
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        logger.info(f"Processing command: '{text_clean}'")
        
        # Check for built-in commands first (fast local processing)
        for command, handler in self.command_mappings.items():
            if command in text_lower:
                logger.info(f"Matched built-in command: {command}")
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Built-in command failed: {e}")
                    self._speak("I encountered an error processing that command, sir.")
                return
        
        # If no built-in command matched, try AI brain
        if self.ai_enabled and self.ai_brain and self.ai_brain.is_available():
            logger.info("No built-in command matched, delegating to AI brain")
            self._handle_ai_request(text_clean)
        else:
            # Fallback to unknown command
            logger.info("AI not available, using unknown command handler")
            self._handle_unknown_command(text_clean)
    
    def _handle_ai_request(self, user_input: str):
        """Handle request through AI brain"""
        try:
            # Build context for AI
            context = self._build_context()
            
            # Process through AI brain
            response = self.ai_brain.process_request(user_input, context)
            
            # Speak the response using assistant's feedback prevention
            self._speak(response)
            
        except Exception as e:
            logger.error(f"AI brain processing failed: {e}")
            self._speak("I'm having trouble processing that request at the moment, sir.")
    
    def _build_context(self) -> Dict[str, Any]:
        """Build context information for AI requests"""
        now = datetime.datetime.now()
        
        context = {
            "time": now.strftime("%I:%M %p"),
            "date": now.strftime("%A, %B %d, %Y"),
            "system": platform.system(),
            "available_commands": list(self.command_mappings.keys())
        }
        
        # Add system info if available
        if PSUTIL_AVAILABLE:
            try:
                context["memory_usage"] = f"{psutil.virtual_memory().percent}%"
                context["cpu_usage"] = f"{psutil.cpu_percent()}%"
                
                # Battery info if available
                battery = psutil.sensors_battery()
                if battery:
                    context["battery"] = f"{battery.percent}%"
            except Exception:
                pass  # Skip if system info fails
        
        return context
    
    def _speak(self, text: str):
        """Speak text using assistant's feedback prevention if available"""
        if self.assistant and hasattr(self.assistant, 'speak_without_feedback'):
            self.assistant.speak_without_feedback(text)
        else:
            self._speak(text)
    
    # Built-in command handlers (these run locally for speed)
    def _handle_greeting(self):
        """Handle greeting commands"""
        current_hour = datetime.datetime.now().hour
        
        if 5 <= current_hour < 12:
            greeting = "Good morning, sir. How may I assist you today?"
        elif 12 <= current_hour < 17:
            greeting = "Good afternoon, sir. What can I do for you?"
        elif 17 <= current_hour < 21:
            greeting = "Good evening, sir. How may I be of service?"
        else:
            greeting = "Good evening, sir. Working late, I see. How may I help?"
        
        self._speak(greeting)
    
    def _handle_time(self):
        """Handle time requests"""
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"The current time is {current_time}, sir."
        self._speak(response)
    
    def _handle_date(self):
        """Handle date requests"""
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        response = f"Today is {current_date}, sir."
        self._speak(response)
    
    def _handle_status(self):
        """Handle basic status requests"""
        responses = [
            "All systems are functioning normally, sir.",
            "Operating at full capacity, sir.",
            "Everything is running smoothly, sir.",
            "All systems green, sir."
        ]
        response = random.choice(responses)
        self._speak(response)
    
    def _handle_system_status(self):
        """Handle detailed system status"""
        if not PSUTIL_AVAILABLE:
            self._speak("System monitoring is not available, sir.")
            return
        
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            
            status_parts = [
                f"CPU usage: {cpu}%",
                f"Memory usage: {memory.percent}%"
            ]
            
            # Add battery if available
            battery = psutil.sensors_battery()
            if battery:
                status_parts.append(f"Battery: {battery.percent}%")
            
            response = f"System status: {', '.join(status_parts)}, sir."
            self._speak(response)
            
        except Exception as e:
            logger.error(f"System status failed: {e}")
            self._speak("I cannot access system information at the moment, sir.")
    
    def _handle_battery(self):
        """Handle battery status"""
        if not PSUTIL_AVAILABLE:
            self._speak("Battery monitoring is not available, sir.")
            return
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                plugged = "charging" if battery.power_plugged else "on battery power"
                response = f"Battery level is {percent}% and {plugged}, sir."
            else:
                response = "Battery information is not available on this system, sir."
        except Exception:
            response = "I cannot access battery information at the moment, sir."
        
        self._speak(response)
    
    def _handle_memory(self):
        """Handle memory status"""
        if not PSUTIL_AVAILABLE:
            self._speak("Memory monitoring is not available, sir.")
            return
        
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            response = f"Memory usage is at {percent_used}%, sir."
        except Exception:
            response = "I cannot access memory information at the moment, sir."
        
        self._speak(response)
    
    def _handle_disk_space(self):
        """Handle disk space status"""
        if not PSUTIL_AVAILABLE:
            self._speak("Disk monitoring is not available, sir.")
            return
        
        try:
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            response = f"Disk usage is at {percent_used:.1f}%, sir."
        except Exception:
            response = "I cannot access disk information at the moment, sir."
        
        self._speak(response)
    
    def _handle_help(self):
        """Handle help requests"""
        if self.ai_enabled:
            response = ("I can help with time, date, system status, and much more, sir. "
                       "I have access to advanced AI capabilities for general questions. "
                       "Simply speak naturally and I'll do my best to assist.")
        else:
            response = ("I can help with time, date, system status, and basic commands, sir. "
                       "For a full list of commands, just ask what I can do.")
        
        self._speak(response)
    
    def _handle_identity(self):
        """Handle identity questions"""
        if self.ai_enabled:
            response = ("I am Jarvis, your AI-enhanced personal assistant, sir. "
                       "I combine local system capabilities with advanced AI intelligence "
                       "to provide comprehensive assistance.")
        else:
            response = ("I am Jarvis, your personal assistant, sir. "
                       "I can help with system monitoring, time, date, and basic commands.")
        
        self._speak(response)
    
    def _handle_ai_status(self):
        """Handle AI brain status requests"""
        if not self.ai_brain:
            self._speak("AI brain is not initialized, sir.")
            return
        
        try:
            status = self.ai_brain.get_status()
            if status["available_providers"] > 0:
                primary = status.get("primary", "Unknown")
                fallback = status.get("fallback")
                
                response = f"AI systems online, sir. Primary: {primary}"
                if fallback:
                    response += f", fallback: {fallback}"
                response += "."
            else:
                response = "AI systems are offline, sir. No providers available."
            
            self._speak(response)
            
        except Exception as e:
            logger.error(f"AI status check failed: {e}")
            self._speak("Unable to check AI status, sir.")
    
    def _handle_clear_history(self):
        """Clear AI conversation history"""
        if self.ai_brain:
            try:
                self.ai_brain.clear_all_history()
                self._speak("AI conversation history cleared, sir.")
            except Exception as e:
                logger.error(f"Failed to clear history: {e}")
                self._speak("Failed to clear history, sir.")
        else:
            self._speak("No AI brain to clear, sir.")
    
    def _handle_disable_ai(self):
        """Disable AI brain"""
        self.ai_enabled = False
        self._speak("AI brain disabled, sir. I'll only use built-in commands.")
    
    def _handle_enable_ai(self):
        """Enable AI brain"""
        if self.ai_brain and self.ai_brain.is_available():
            self.ai_enabled = True
            self._speak("AI brain enabled, sir. Full capabilities restored.")
        else:
            self._speak("AI brain is not available, sir. Check your API configuration.")
    
    def _handle_stop_listening(self):
        """Handle stop listening command"""
        self.assistant.is_active = False
        self._speak("I'll stop listening now, sir. Say my name to reactivate me.")
    
    def _handle_shutdown(self):
        """Handle shutdown command"""
        self._speak("Goodbye, sir. Shutting down Jarvis systems.")
        self.assistant.is_listening = False
    
    def _handle_goodbye(self):
        """Handle goodbye"""
        responses = [
            "Until next time, sir.",
            "Have a pleasant day, sir.",
            "Goodbye, sir.",
            "I'll be here when you need me, sir."
        ]
        response = random.choice(responses)
        self._speak(response)
        self.assistant.is_active = False
    
    def _handle_test(self):
        """Handle test command"""
        self._speak("Voice systems are functioning perfectly, sir.")
    
    def _handle_unknown_command(self, command: str):
        """Handle unknown commands when AI isn't available"""
        responses = [
            f"I'm not sure how to handle '{command}', sir. Could you try rephrasing?",
            f"I don't recognize that command, sir. Would you like me to explain what I can do?",
            f"I'm afraid I don't understand '{command}', sir. Please try a different request."
        ]
        response = random.choice(responses)
        self._speak(response)
    
    def get_available_commands(self) -> list:
        """Get list of available built-in commands"""
        return list(self.command_mappings.keys())
    
    def is_ai_enabled(self) -> bool:
        """Check if AI brain is enabled and available"""
        return self.ai_enabled and self.ai_brain and self.ai_brain.is_available()


# Configuration helper
def create_ai_config(
    anthropic_enabled: bool = True,
    anthropic_model: str = "claude-3-haiku-20240307",
    deepseek_enabled: bool = True,
    deepseek_model: str = "deepseek-chat",
    prefer_anthropic: bool = True
) -> Dict[str, Any]:
    """Create AI configuration for the commands module"""
    
    return {
        "providers": {
            "anthropic": {
                "enabled": anthropic_enabled,
                "model": anthropic_model,
                "priority": 1 if prefer_anthropic else 2
            },
            "deepseek": {
                "enabled": deepseek_enabled,
                "model": deepseek_model,
                "priority": 2 if prefer_anthropic else 1
            },
            "local": {
                "enabled": False,
                "priority": 3
            }
        },
        "fallback_enabled": True,
        "health_check_interval": 300
    }


# Example usage
if __name__ == "__main__":
    # Mock TTS and assistant for testing
    class MockTTS:
        def speak_direct(self, text):
            print(f"TTS: {text}")
    
    class MockAssistant:
        def __init__(self):
            self.is_active = True
            self.is_listening = True
            self.prevent_feedback = False
        
        def speak_without_feedback(self, text):
            print(f"ASSISTANT TTS: {text}")
    
    # Create AI config
    ai_config = create_ai_config(
        anthropic_enabled=True,
        deepseek_enabled=True,
        prefer_anthropic=True
    )
    
    # Initialize commands
    mock_tts = MockTTS()
    mock_assistant = MockAssistant()
    
    commands = JarvisCommands(mock_tts, mock_assistant, ai_config)
    
    # Test commands
    test_commands = [
        "hello",
        "what time is it",
        "ai status", 
        "What's the weather like in Tokyo?",  # This should go to AI
        "Tell me a joke about robots",        # This should go to AI
        "system status"
    ]
    
    print("Testing Enhanced Jarvis Commands:")
    print("=" * 50)
    
    for cmd in test_commands:
        print(f"\nUser: {cmd}")
        commands.process_command(cmd)
    
    print("\n" + "=" * 50)
    print(f"AI Enabled: {commands.is_ai_enabled()}")
    print(f"Available built-in commands: {len(commands.get_available_commands())}")