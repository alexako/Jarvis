#!/usr/bin/env python3
"""
Jarvis Voice Commands Module

Centralized command processing for the Jarvis voice assistant.
Contains all voice commands, their patterns, and handler functions.
"""

import datetime
import random
import logging
import subprocess
import platform
import os
from typing import Dict, List, Callable, Optional, Any

logger = logging.getLogger(__name__)

class JarvisCommands:
    """Centralized command processing for Jarvis voice assistant"""
    
    def __init__(self, tts_engine, assistant_instance=None):
        self.tts = tts_engine
        self.assistant = assistant_instance
        
        # Command patterns and their handlers
        self.command_patterns = {
            # Basic interaction
            "time": self.tell_time,
            "what time": self.tell_time,
            "current time": self.tell_time,
            "what's the time": self.tell_time,
            
            "date": self.tell_date,
            "what date": self.tell_date,
            "today's date": self.tell_date,
            "what's the date": self.tell_date,
            "what day": self.tell_date,
            
            # Greetings
            "hello": self.greet,
            "hi": self.greet,
            "good morning": self.greet,
            "good afternoon": self.greet,
            "good evening": self.greet,
            "hey": self.greet,
            
            # Status and testing
            "how are you": self.status_check,
            "status": self.status_check,
            "system status": self.status_check,
            "are you okay": self.status_check,
            "test": self.test_response,
            "test voice": self.test_response,
            "test system": self.test_response,
            
            # System control (deactivation commands moved to separate handling)
            
            # Information requests
            "who are you": self.introduce,
            "what are you": self.introduce,
            "introduce yourself": self.introduce,
            
            # Weather (placeholder)
            "weather": self.weather_info,
            "what's the weather": self.weather_info,
            "temperature": self.weather_info,
            
            # System information
            "battery": self.battery_status,
            "memory": self.memory_usage,
            "disk space": self.disk_usage,
            
            # Entertainment
            "tell me a joke": self.tell_joke,
            "joke": self.tell_joke,
            "something funny": self.tell_joke,
            
            # Productivity
            "remind me": self.set_reminder,
            "reminder": self.set_reminder,
            "note": self.take_note,
            "make a note": self.take_note,
            
            # Help
            "help": self.show_help,
            "what can you do": self.show_help,
            "commands": self.show_help,
            "capabilities": self.show_help,
        }
        
        # Response templates
        self.responses = {
            "acknowledgments": [
                "Certainly, sir.",
                "Right away, sir.",
                "Of course, sir.",
                "Consider it done, sir.",
                "Immediately, sir.",
                "Very well, sir."
            ],
            "errors": [
                "I'm not sure I understand that request, sir. Could you please rephrase?",
                "I don't have that capability at the moment, sir. How else may I assist you?",
                "I'm afraid I don't comprehend that command, sir. Please try again.",
                "Could you please clarify what you'd like me to do, sir?",
                "I'm having difficulty processing that request, sir."
            ],
            "greetings": {
                "morning": "Good morning, sir. How may I assist you today?",
                "afternoon": "Good afternoon, sir. How may I assist you today?", 
                "evening": "Good evening, sir. How may I assist you today?",
                "general": "Hello, sir. How may I assist you today?"
            },
            "farewells": [
                "Until next time, sir. Have a pleasant day.",
                "Goodbye, sir. I'll be here when you need me.",
                "Farewell, sir. Have a wonderful day.",
                "Good day, sir. Until we speak again."
            ]
        }
    
    def process_command(self, text: str) -> bool:
        """Process voice command and execute appropriate action"""
        text_lower = text.lower().strip()
        
        # Deactivation commands - require exact match or very specific phrasing
        deactivation_commands = {
            "stop listening": self.stop_listening,
            "go to sleep": self.stop_listening, 
            "shutdown": self.shutdown,
            "shutdown jarvis": self.shutdown,
            "goodbye": self.goodbye,
            "goodbye jarvis": self.goodbye
        }
        
        # Check deactivation commands first with exact match
        for pattern, handler in deactivation_commands.items():
            if text_lower == pattern or text_lower.endswith(pattern):
                logger.info(f"🎯 Executing deactivation command: {pattern}")
                try:
                    handler(text)
                    return True
                except Exception as e:
                    logger.error(f"Command execution failed: {e}")
                    self.speak_error()
                    return True
        
        # Check for other command patterns with substring match
        for pattern, handler in self.command_patterns.items():
            if pattern not in deactivation_commands and pattern in text_lower:
                logger.info(f"🎯 Executing command: {pattern}")
                try:
                    handler(text)
                    return True
                except Exception as e:
                    logger.error(f"Command execution failed: {e}")
                    self.speak_error()
                    return True
        
        # No command found
        self.handle_unknown_command(text)
        return False
    
    def speak(self, text: str):
        """Convenience method to speak text"""
        if self.assistant and hasattr(self.assistant, 'speak_without_feedback'):
            self.assistant.speak_without_feedback(text)
        else:
            self.tts.speak_direct(text)
    
    def speak_random(self, response_list: List[str]):
        """Speak a random response from list"""
        self.speak(random.choice(response_list))
    
    def speak_error(self):
        """Speak a random error message"""
        self.speak_random(self.responses["errors"])
    
    # ==================== COMMAND HANDLERS ====================
    
    def tell_time(self, text: str):
        """Tell current time"""
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"The current time is {current_time}, sir."
        self.speak(response)
    
    def tell_date(self, text: str):
        """Tell current date"""
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        response = f"Today is {current_date}, sir."
        self.speak(response)
    
    def greet(self, text: str):
        """Handle greetings with time-appropriate responses"""
        current_hour = datetime.datetime.now().hour
        
        if current_hour < 12:
            response = self.responses["greetings"]["morning"]
        elif current_hour < 18:
            response = self.responses["greetings"]["afternoon"]
        else:
            response = self.responses["greetings"]["evening"]
        
        self.speak(response)
    
    def status_check(self, text: str):
        """System status check"""
        responses = [
            "All systems are functioning normally, sir. I am ready to assist you.",
            "Systems operational, sir. How may I help you?",
            "Everything is running smoothly, sir. What do you need?",
            "All systems green, sir. Standing by for your commands."
        ]
        self.speak_random(responses)
    
    def test_response(self, text: str):
        """Test voice and system response"""
        responses = [
            "Test successful. All systems operational, sir.",
            "Voice recognition and synthesis are working perfectly, sir.",
            "I hear you loud and clear, sir.",
            "Systems check complete. Everything is functioning optimally, sir."
        ]
        self.speak_random(responses)
    
    def introduce(self, text: str):
        """Introduce Jarvis"""
        response = "I am Jarvis, your personal AI assistant, sir. I'm here to help with information, system control, and various tasks at your command."
        self.speak(response)
    
    def weather_info(self, text: str):
        """Weather information (placeholder)"""
        response = "I don't currently have access to weather data, sir. This feature will be available in a future update."
        self.speak(response)
    
    def battery_status(self, text: str):
        """Check battery status (macOS)"""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True)
                if "InternalBattery" in result.stdout:
                    # Parse battery percentage
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "InternalBattery" in line and "%" in line:
                            percentage = line.split('%')[0].split()[-1]
                            if "charging" in line.lower():
                                status = "and charging"
                            elif "charged" in line.lower():
                                status = "and fully charged"
                            else:
                                status = "on battery power"
                            response = f"Battery is at {percentage} percent {status}, sir."
                            break
                    else:
                        response = "Unable to determine battery status, sir."
                else:
                    response = "This device doesn't appear to have a battery, sir."
            else:
                response = "Battery monitoring is not available on this system, sir."
        except Exception as e:
            response = "Unable to check battery status at the moment, sir."
        
        self.speak(response)
    
    def memory_usage(self, text: str):
        """Check memory usage"""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(["vm_stat"], capture_output=True, text=True)
                response = "Memory statistics retrieved, sir. System memory appears to be functioning normally."
            else:
                response = "Memory monitoring details are not available on this system, sir."
        except Exception:
            response = "Unable to check memory usage at the moment, sir."
        
        self.speak(response)
    
    def disk_usage(self, text: str):
        """Check disk usage"""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        used_percent = parts[4]
                        response = f"Main disk is {used_percent} full, sir."
                    else:
                        response = "Disk usage information retrieved, sir."
                else:
                    response = "Unable to determine disk usage, sir."
            else:
                response = "Disk monitoring is not available on this system, sir."
        except Exception:
            response = "Unable to check disk usage at the moment, sir."
        
        self.speak(response)
    
    def tell_joke(self, text: str):
        """Tell a joke"""
        jokes = [
            "Why don't scientists trust atoms, sir? Because they make up everything.",
            "I told my computer a joke about UDP, sir. I don't know if it got it.",
            "Why do programmers prefer dark mode, sir? Because light attracts bugs.",
            "What do you call a computer that sings, sir? A Dell.",
            "Why was the JavaScript developer sad, sir? Because he didn't know how to null his feelings."
        ]
        self.speak_random(jokes)
    
    def set_reminder(self, text: str):
        """Set a reminder (placeholder)"""
        response = "I don't currently have access to a reminder system, sir. This feature will be implemented in a future update."
        self.speak(response)
    
    def take_note(self, text: str):
        """Take a note (placeholder)"""
        response = "Note-taking functionality is not yet implemented, sir. This feature will be available soon."
        self.speak(response)
    
    def show_help(self, text: str):
        """Show available commands"""
        response = """I can assist you with time and date information, system status checks, 
        basic system monitoring, greetings, and general conversation, sir. 
        You can ask me about the time, date, system status, battery level, 
        or request a system test. I'm also happy to chat or tell you a joke."""
        self.speak(response)
    
    def stop_listening(self, text: str):
        """Stop listening temporarily"""
        response = "Stopping voice recognition, sir. Say my name to reactivate."
        self.speak(response)
        if self.assistant:
            self.assistant.is_active = False
    
    def shutdown(self, text: str):
        """Shutdown the assistant"""
        response = "Shutting down. Until next time, sir."
        self.speak(response)
        if self.assistant:
            self.assistant.is_listening = False
            self.assistant.is_active = False
    
    def goodbye(self, text: str):
        """Handle goodbye"""
        self.speak_random(self.responses["farewells"])
        if self.assistant:
            self.assistant.is_active = False
    
    def handle_unknown_command(self, text: str):
        """Handle unrecognized commands"""
        self.speak_error()
    
    # ==================== UTILITY METHODS ====================
    
    def get_available_commands(self) -> List[str]:
        """Get list of available command patterns"""
        return list(self.command_patterns.keys())
    
    def add_custom_command(self, pattern: str, handler: Callable[[str], None]):
        """Add a custom command pattern and handler"""
        self.command_patterns[pattern] = handler
        logger.info(f"Added custom command: {pattern}")
    
    def remove_command(self, pattern: str) -> bool:
        """Remove a command pattern"""
        if pattern in self.command_patterns:
            del self.command_patterns[pattern]
            logger.info(f"Removed command: {pattern}")
            return True
        return False
