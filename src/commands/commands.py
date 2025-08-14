#!/usr/bin/env python3
"""
Enhanced Jarvis Commands with AI Brain Integration
Modular command system that falls back to AI for unknown requests
"""

import sys
import os
# Add the src directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import datetime
import logging
import platform
import random
import re
from typing import Optional, Dict, Any

# Import the AI brain module
from ai.ai_brain import create_ai_brain, AIBrainManager

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
    
    def __init__(self, tts, assistant, ai_config: Optional[Dict[str, Any]] = None, context=None):
        self.tts = tts
        self.assistant = assistant
        self.context = context
        
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
        # NOTE: Use specific phrases to avoid false matches in normal conversation
        self.command_mappings = {
            # Greetings - only exact matches or start-of-sentence
            'hello jarvis': self._handle_greeting,
            'hi jarvis': self._handle_greeting,
            'hey jarvis': self._handle_greeting,
            'good morning': self._handle_greeting,
            'good afternoon': self._handle_greeting,
            'good evening': self._handle_greeting,
            
            # Time and date - specific questions only
            'what time is it': self._handle_time,
            'current time': self._handle_time,
            'what date is it': self._handle_date,
            "what's the date": self._handle_date,
            'current date': self._handle_date,
            
            # Status and system info - specific requests
            'how are you': self._handle_status,
            'jarvis status': self._handle_status,
            'system status': self._handle_system_status,
            'battery status': self._handle_battery,
            'check memory usage': self._handle_memory,
            'disk space': self._handle_disk_space,
            
            # Help and identity - specific requests
            'jarvis help me': self._handle_help,
            'what can you do': self._handle_help,
            'list commands': self._handle_help,
            'who are you': self._handle_identity,
            'introduce yourself': self._handle_identity,
            
            # Entertainment - specific requests
            'tell me a joke': self._handle_joke,
            'make me laugh': self._handle_joke,
            
            # Control commands - specific phrases
            'stop listening': self._handle_stop_listening,
            'shutdown system': self._handle_shutdown,
            'goodbye jarvis': self._handle_goodbye,
            'bye jarvis': self._handle_goodbye,
            'run voice test': self._handle_test,
            
            # AI brain management
            'ai status': self._handle_ai_status,
            'clear history': self._handle_clear_history,
            'disable ai': self._handle_disable_ai,
            'enable ai': self._handle_enable_ai,
            
            # Context/Memory commands
            'remember that': self._handle_remember,
            'my name is': self._handle_learn_name,
            'call me': self._handle_learn_name,
            'context status': self._handle_context_status,
            'conversation summary': self._handle_conversation_summary,
            'what do you know about me': self._handle_what_you_know,
            'forget that': self._handle_forget,
            'reset memory': self._handle_reset_memory,
            
            # Multi-user commands
            'switch to user': self._handle_switch_user,
            'i am': self._handle_i_am,
            'who am i': self._handle_who_am_i,
            'list users': self._handle_list_users,
            'current user': self._handle_current_user,
            'create user': self._handle_create_user,
            
            # Alias management commands
            'add alias': self._handle_add_alias,
            'call me also': self._handle_add_alias,
            'my aliases': self._handle_list_aliases,
            'remove alias': self._handle_remove_alias,
            'primary name': self._handle_set_primary,
        }
        
        logger.info(f"Enhanced commands initialized - AI enabled: {self.ai_enabled}")
    
    def process_command(self, text: str):
        """Process command with context awareness and AI fallback"""
        if not text or not text.strip():
            return
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        logger.info(f"Processing command: '{text_clean}'")
        
        # Check for built-in commands first (fast local processing)
        # Use word boundary matching to prevent false matches like "hi" in "this"
        command_matched = False
        for command, handler in self.command_mappings.items():
            # Create regex pattern with word boundaries for better matching
            pattern = r'\b' + re.escape(command) + r'\b'
            if re.search(pattern, text_lower):
                logger.info(f"Matched built-in command: {command}")
                try:
                    # Some commands need the full text for context
                    if command in ['remember that', 'my name is', 'call me', 'forget that', 
                                  'switch to user', 'i am', 'create user', 'add alias', 
                                  'call me also', 'remove alias', 'primary name']:
                        handler(text_clean)
                    else:
                        handler()
                    
                    # For context tracking, we need to add the exchange after response
                    # This is handled in the individual command handlers or _speak method
                    command_matched = True
                    break
                except Exception as e:
                    logger.error(f"Built-in command failed: {e}")
                    self._speak("I encountered an error processing that command, sir.", text_clean, "error")
                    command_matched = True
                    break
        
        if command_matched:
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
        """Handle request through AI brain with context awareness"""
        try:
            # Build context for AI including conversation history
            context = self._build_context()
            
            # Add conversation context if available
            if self.context:
                context_info = self.context.get_context_for_ai(include_persistent=True)
                if context_info:
                    context['conversation_context'] = context_info
            
            # Process through AI brain
            response = self.ai_brain.process_request(user_input, context)
            
            # Store the exchange in context
            if self.context:
                # Determine topic from user input
                topic = self._extract_topic(user_input)
                self.context.add_exchange(
                    user_input=user_input,
                    jarvis_response=response,
                    topic=topic,
                    context_data={'ai_processed': True, 'timestamp': datetime.datetime.now().isoformat()}
                )
            
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
    
    def _speak(self, text: str, user_input: str = None, topic: str = None):
        """Speak text using assistant's feedback prevention if available"""
        if self.assistant and hasattr(self.assistant, 'speak_with_feedback_control'):
            self.assistant.speak_with_feedback_control(text)
        elif self.tts:
            self.tts.speak_direct(text)
        else:
            print(f"Jarvis: {text}")
        
        # Store the exchange in context if available
        if self.context and user_input:
            self.context.add_exchange(
                user_input=user_input,
                jarvis_response=text,
                topic=topic,
                context_data={'command_type': 'builtin', 'timestamp': datetime.datetime.now().isoformat()}
            )
    
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
    
    def _handle_joke(self):
        """Handle joke requests"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Why don't programmers like nature? It has too many bugs.",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "What did the ocean say to the beach? Nothing, it just waved.",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a fake noodle? An impasta!"
        ]
        joke = random.choice(jokes)
        self._speak(f"Here's a joke for you, sir: {joke}")
    
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
    
    # Context/Memory command handlers
    def _handle_remember(self, text: str):
        """Handle remember that commands"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        # Extract what to remember
        if "remember that" in text.lower():
            info = text.lower().split("remember that", 1)[1].strip()
        else:
            info = text.strip()
        
        if info:
            # Store as a general fact
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.context.learn_about_user("remembered_info", f"{timestamp}: {info}")
            self._speak(f"I'll remember that, sir: {info}")
        else:
            self._speak("What would you like me to remember, sir?")
    
    def _handle_learn_name(self, text: str):
        """Handle name learning commands"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        # Extract name from various patterns
        text_lower = text.lower()
        name = None
        
        if "my name is" in text_lower:
            name = text_lower.split("my name is", 1)[1].strip()
        elif "call me" in text_lower:
            name = text_lower.split("call me", 1)[1].strip()
        
        if name:
            # Clean up the name
            name = name.replace(".", "").strip()
            self.context.learn_about_user("name", name, confidence=1.0, source="direct_instruction")
            self._speak(f"Understood, {name}. I'll remember your name.")
        else:
            self._speak("I didn't catch your name, sir. Could you please repeat it?")
    
    def _handle_context_status(self):
        """Handle context status command"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        status = self.context.get_context_status()
        # Convert status to speech-friendly format
        lines = status.split('\n')
        summary = []
        for line in lines[1:6]:  # Get first few important lines
            if '•' in line:
                summary.append(line.replace('•', '').strip())
        
        if summary:
            status_text = "Context system status: " + ". ".join(summary)
            self._speak(status_text)
        else:
            self._speak("Context system is active and functioning, sir.")
    
    def _handle_conversation_summary(self):
        """Handle conversation summary command"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        summary = self.context.get_conversation_summary()
        self._speak(f"Here's your conversation summary, sir: {summary}")
    
    def _handle_what_you_know(self):
        """Handle what do you know about me command"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        info_parts = []
        
        # Add user name if known
        if self.context.user_name:
            info_parts.append(f"Your name is {self.context.user_name}")
        
        # Add preferences
        prefs = self.context.session_preferences
        if prefs:
            pref_count = len(prefs)
            info_parts.append(f"I have {pref_count} preferences stored")
        
        # Add recent topics
        recent_context = self.context.get_recent_context(3)
        if recent_context:
            topics = [ctx.get('topic') for ctx in recent_context if ctx.get('topic')]
            if topics:
                unique_topics = list(set(topics))[:2]  # Limit to 2 topics
                info_parts.append(f"We recently discussed {', '.join(unique_topics)}")
        
        if info_parts:
            response = "Here's what I know about you, sir: " + ". ".join(info_parts) + "."
        else:
            response = "I don't have much personal information stored yet, sir. We can build up my knowledge of your preferences over time."
        
        self._speak(response)
    
    def _handle_forget(self, text: str):
        """Handle forget that command - simplified version"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        # For now, just reset the current topic
        self.context.current_topic = None
        self._speak("I've cleared the current topic from my memory, sir.")
    
    def _handle_reset_memory(self):
        """Handle reset memory command"""
        if not self.context:
            self._speak("Memory system is not available, sir.")
            return
        
        self.context.reset_session()
        self._speak("Session memory has been reset, sir. Persistent preferences and information remain intact.")
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract topic from user input for context tracking"""
        text_lower = text.lower()
        
        # Simple topic extraction based on keywords
        topic_keywords = {
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy'],
            'time': ['time', 'date', 'schedule', 'calendar'],
            'music': ['music', 'song', 'play', 'playlist', 'artist'],
            'news': ['news', 'headlines', 'current events'],
            'programming': ['code', 'programming', 'python', 'javascript', 'debug'],
            'system': ['system', 'battery', 'memory', 'disk', 'performance'],
            'personal': ['remember', 'name', 'preference', 'like', 'favorite']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        # If no specific topic found, use first few words as topic
        words = text.split()[:3]
        if len(words) >= 2:
            return ' '.join(words).lower()
        
        return None
    
    # Multi-user command handlers
    def _handle_switch_user(self, text: str):
        """Handle switch to user command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        # Extract user name from command
        text_lower = text.lower()
        if "switch to user" in text_lower:
            user_name = text_lower.split("switch to user", 1)[1].strip()
        else:
            user_name = text_lower.strip()
        
        if user_name:
            # Clean up the user name and create user_id
            user_name = user_name.replace(".", "").strip()
            user_id = user_name.lower().replace(" ", "_")
            
            if self.context.switch_user(user_id, user_name):
                self._speak(f"Switched to user {user_name}, sir.")
            else:
                self._speak(f"I encountered an error switching to user {user_name}, sir.")
        else:
            self._speak("Please specify which user to switch to, sir.")
    
    def _handle_i_am(self, text: str):
        """Handle 'I am [name]' command for user identification"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        # Extract name
        text_lower = text.lower()
        if "i am" in text_lower:
            user_name = text_lower.split("i am", 1)[1].strip()
        else:
            user_name = text_lower.strip()
        
        if user_name:
            # Clean up the user name
            user_name = user_name.replace(".", "").strip()
            user_id = user_name.lower().replace(" ", "_")
            
            # Switch to this user (creating if necessary)
            if self.context.switch_user(user_id, user_name):
                self._speak(f"Hello {user_name}, sir. I've switched to your profile.")
            else:
                self._speak(f"I encountered an error switching to your profile, sir.")
        else:
            self._speak("Please tell me your name, sir.")
    
    def _handle_who_am_i(self):
        """Handle who am I command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        current_user = self.context.get_current_user()
        user_name = current_user['display_name']
        is_default = current_user['is_default']
        
        if is_default and user_name == "Default User":
            self._speak("You are currently using the default user profile, sir. Say 'I am [your name]' to identify yourself.")
        else:
            self._speak(f"You are {user_name}, sir.")
    
    def _handle_current_user(self):
        """Handle current user status command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        current_user = self.context.get_current_user()
        user_name = current_user['display_name']
        user_id = current_user['user_id']
        
        response = f"Current user: {user_name}"
        if user_id != user_name.lower().replace(" ", "_"):
            response += f" (ID: {user_id})"
        
        self._speak(response + ", sir.")
    
    def _handle_list_users(self):
        """Handle list users command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        users = self.context.list_users()
        if not users:
            self._speak("No users found in the system, sir.")
            return
        
        if len(users) == 1:
            user = users[0]
            current_marker = " (current)" if user['is_current'] else ""
            self._speak(f"There is one user: {user['display_name']}{current_marker}, sir.")
        else:
            user_names = []
            for user in users:
                name = user['display_name']
                if user['is_current']:
                    name += " (current)"
                user_names.append(name)
            
            if len(user_names) <= 3:
                user_list = ", ".join(user_names)
            else:
                user_list = ", ".join(user_names[:3]) + f", and {len(user_names) - 3} others"
            
            self._speak(f"System users: {user_list}, sir.")
    
    def _handle_create_user(self, text: str):
        """Handle create user command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        # Extract user name from command
        text_lower = text.lower()
        if "create user" in text_lower:
            user_name = text_lower.split("create user", 1)[1].strip()
        else:
            user_name = text_lower.strip()
            
        if user_name:
            # Clean up the user name
            user_name = user_name.replace(".", "").strip()
            user_id = user_name.lower().replace(" ", "_")
            
            if self.context.create_user(user_id, user_name):
                self._speak(f"Created user {user_name}, sir.")
            else:
                self._speak(f"User {user_name} already exists or I encountered an error, sir.")
        else:
            self._speak("Please specify the name for the new user, sir.")
    
    # Alias management command handlers
    def _handle_add_alias(self, text: str):
        """Handle add alias commands"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        # Extract alias from command
        text_lower = text.lower()
        alias = None
        
        if "add alias" in text_lower:
            alias = text_lower.split("add alias", 1)[1].strip()
        elif "call me also" in text_lower:
            alias = text_lower.split("call me also", 1)[1].strip()
        
        if alias:
            alias = alias.replace(".", "").strip()
            if self.context.add_user_alias(alias):
                self._speak(f"I'll also call you {alias}, sir.")
            else:
                self._speak(f"I already know that name for you, sir.")
        else:
            self._speak("What additional name would you like me to use, sir?")
    
    def _handle_list_aliases(self):
        """Handle list aliases command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        aliases = self.context.get_user_aliases()
        current_user = self.context.get_current_user()
        
        names = [current_user['display_name']]  # Start with display name
        for alias in aliases:
            if alias['alias'] != current_user['display_name']:  # Avoid duplicates
                names.append(alias['alias'])
        
        if len(names) == 1:
            self._speak(f"I only know you as {names[0]}, sir.")
        elif len(names) == 2:
            self._speak(f"I know you as {names[0]} and {names[1]}, sir.")
        else:
            name_list = ", ".join(names[:-1]) + f", and {names[-1]}"
            self._speak(f"I know you as {name_list}, sir.")
    
    def _handle_remove_alias(self, text: str):
        """Handle remove alias command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        # Extract alias to remove
        text_lower = text.lower()
        if "remove alias" in text_lower:
            alias = text_lower.split("remove alias", 1)[1].strip()
        else:
            alias = text_lower.strip()
        
        if alias:
            alias = alias.replace(".", "").strip()
            if self.context.remove_user_alias(alias):
                self._speak(f"I'll no longer call you {alias}, sir.")
            else:
                self._speak(f"I don't have that name on record, sir.")
        else:
            self._speak("Which name would you like me to remove, sir?")
    
    def _handle_set_primary(self, text: str):
        """Handle set primary name command"""
        if not self.context:
            self._speak("Multi-user system is not available, sir.")
            return
        
        # Extract primary name
        text_lower = text.lower()
        name = None
        
        if "primary name" in text_lower:
            name = text_lower.split("primary name", 1)[1].strip()
            # Handle common phrasings
            if name.startswith("is "):
                name = name[3:].strip()
        
        if name:
            name = name.replace(".", "").strip()
            
            # First add as alias if it doesn't exist
            self.context.add_user_alias(name)
            
            # Then set as primary
            if self.context.set_primary_alias(name):
                self._speak(f"I'll primarily call you {name}, sir.")
            else:
                self._speak(f"I couldn't set {name} as your primary name, sir.")
        else:
            self._speak("What would you like your primary name to be, sir?")
    
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
    local_enabled: bool = True,
    local_model: str = "llama3.2:latest",
    prefer_anthropic: bool = True,
    prefer_local: bool = False
) -> Dict[str, Any]:
    """Create AI configuration for the commands module"""
    
    # Determine priorities based on preferences
    if prefer_local:
        local_priority = 1
        primary_priority = 2
        secondary_priority = 3
    else:
        local_priority = 3
        primary_priority = 1 if prefer_anthropic else 2
        secondary_priority = 2 if prefer_anthropic else 1
    
    return {
        "providers": {
            "anthropic": {
                "enabled": anthropic_enabled,
                "model": anthropic_model,
                "priority": primary_priority if prefer_anthropic else secondary_priority
            },
            "deepseek": {
                "enabled": deepseek_enabled,
                "model": deepseek_model,
                "priority": secondary_priority if prefer_anthropic else primary_priority
            },
            "local": {
                "enabled": local_enabled,
                "model": local_model,
                "priority": local_priority
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