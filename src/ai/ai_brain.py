#!/usr/bin/env python3
"""
Jarvis AI Brain Module - Modular intelligence provider
Supports multiple AI providers with fallbacks and cost optimization
"""

import sys
import os
# Add the src directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum

class BrainProvider(Enum):
    """Available AI brain providers"""
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    LOCAL = "local"

class ComplexityLevel(Enum):
    """Request complexity levels for cost optimization"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class BaseBrain(ABC):
    """Abstract base class for AI brain providers"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.available = False
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 8
    
    @abstractmethod
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process a user request and return response"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the brain provider is healthy and responsive"""
        pass
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info(f"{self.provider_name} brain history cleared")
    
    def assess_complexity(self, user_input: str) -> ComplexityLevel:
        """Assess the complexity of a user request"""
        input_lower = user_input.lower()
        
        complex_keywords = [
            "analyze", "compare", "write", "create", "design", "strategy", 
            "detailed", "explain in detail", "comprehensive", "research"
        ]
        
        medium_keywords = [
            "explain", "how", "what", "why", "help", "calculate", 
            "describe", "tell me about"
        ]
        
        simple_keywords = [
            "time", "date", "weather", "hello", "hi", "status", 
            "yes", "no", "thanks"
        ]
        
        # Check for complexity indicators
        if any(keyword in input_lower for keyword in complex_keywords):
            return ComplexityLevel.COMPLEX
        elif any(keyword in input_lower for keyword in medium_keywords):
            return ComplexityLevel.MEDIUM
        elif any(keyword in input_lower for keyword in simple_keywords):
            return ComplexityLevel.SIMPLE
        
        # Fallback to length-based assessment
        word_count = len(user_input.split())
        if word_count > 20:
            return ComplexityLevel.COMPLEX
        elif word_count > 8:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE

class AnthropicBrain(BaseBrain):
    """Claude-powered AI brain"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        super().__init__("Anthropic Claude")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.model_name = model  # Alias for compatibility with tests
        self.provider = BrainProvider.ANTHROPIC  # Add provider attribute
        
        if not self.api_key:
            logger.warning("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.available = True
            logger.info(f"Anthropic brain initialized with {model}")
        except ImportError:
            logger.error("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic brain: {e}")
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process request through Claude"""
        if not self.available:
            raise Exception("Anthropic brain not available")
        
        try:
            # Build system prompt with Jarvis personality
            system_prompt = """You are Jarvis, Tony Stark's sophisticated AI assistant. Your characteristics:

- Speak with refined British wit and intelligence
- Address the user as "sir" when appropriate  
- Maintain dry humor and slight sarcasm
- Be helpful while keeping responses concise (under 100 words unless detail is requested)
- You're incredibly capable but personable
- Professional butler aesthetic with cutting-edge AI capabilities

Use any provided context naturally in your responses."""

            # Prepare the user message with context
            if context:
                context_parts = []
                for key, value in context.items():
                    if value:
                        context_parts.append(f"{key}: {value}")
                
                if context_parts:
                    full_message = f"Context: {', '.join(context_parts)}\n\nRequest: {user_input}"
                else:
                    full_message = user_input
            else:
                full_message = user_input
            
            # Build conversation messages
            messages = []
            
            # Add recent history
            for exchange in self.conversation_history[-self.max_history:]:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
            
            # Add current request
            messages.append({"role": "user", "content": full_message})
            
            # Get response from Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.7,
                system=system_prompt,
                messages=messages
            )
            
            assistant_response = response.content[0].text.strip()
            
            # Store in history
            self.conversation_history.append({
                "user": user_input,
                "assistant": assistant_response
            })
            
            # Trim history
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise
    
    def process_query(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Alias for process_request for backward compatibility"""
        return self.process_request(user_input, context)
    
    def is_healthy(self) -> bool:
        """Test if Anthropic API is responsive"""
        if not self.available:
            return False
        
        try:
            # Simple test request
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
            return bool(response.content)
        except Exception:
            return False

class DeepSeekBrain(BaseBrain):
    """DeepSeek AI-powered brain"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        super().__init__("DeepSeek")
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") 
        self.model = model
        self.model_name = model  # Alias for compatibility with tests
        self.provider = BrainProvider.DEEPSEEK  # Add provider attribute
        self.base_url = "https://api.deepseek.com"
        
        if not self.api_key:
            logger.warning("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
            return
        
        try:
            # DeepSeek uses OpenAI-compatible API
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.available = True
            logger.info(f"DeepSeek brain initialized with {model}")
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek brain: {e}")
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process request through DeepSeek"""
        if not self.available:
            raise Exception("DeepSeek brain not available")
        
        try:
            # Build messages
            messages = [
                {
                    "role": "system", 
                    "content": "You are Jarvis, Tony Stark's sophisticated AI assistant. Be witty, helpful, and address the user as 'sir' when appropriate. Keep responses under 100 words unless detail is requested."
                }
            ]
            
            # Add history
            for exchange in self.conversation_history[-self.max_history:]:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
            
            # Add current request with context
            if context:
                context_str = json.dumps(context, indent=2)
                full_message = f"Context: {context_str}\n\nRequest: {user_input}"
            else:
                full_message = user_input
            
            messages.append({"role": "user", "content": full_message})
            
            # Get response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Store in history
            self.conversation_history.append({
                "user": user_input,
                "assistant": assistant_response
            })
            
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            raise
    
    def process_query(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Alias for process_request for backward compatibility"""
        return self.process_request(user_input, context)
    
    def is_healthy(self) -> bool:
        """Test if DeepSeek API is responsive"""
        if not self.available:
            return False
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return bool(response.choices)
        except Exception:
            return False

class LocalBrain(BaseBrain):
    """Local Ollama-powered brain for private, offline AI responses"""
    
    def __init__(self, model_name: str = "phi3.5:3.8b"):
        super().__init__("Local Phi-3.5")
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
        # Test Ollama connection
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0 and model_name in result.stdout:
                self.available = True
                logger.info(f"Local brain initialized with {model_name}")
            else:
                logger.warning(f"Model {model_name} not found in Ollama")
                self.available = False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Ollama not available: {e}")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize local brain: {e}")
            self.available = False
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process request using local Ollama model"""
        if not self.available:
            return "Local AI is not available, sir."
        
        try:
            import subprocess
            import json
            
            # Debug: Log what context is being received
            if context:
                logger.debug(f"LocalBrain received context with keys: {list(context.keys())}")
                if 'conversation_context' in context:
                    logger.debug(f"Conversation context length: {len(context.get('conversation_context', ''))}")
            else:
                logger.debug("LocalBrain received no context")
            
            # Create Jarvis personality prompt with context awareness
            system_prompt = """You are Jarvis, an AI assistant with a formal, helpful personality. 
Address the user as 'sir' and maintain a professional, respectful tone. 
Keep responses concise but informative. You are knowledgeable but acknowledge when you don't have current information.

IMPORTANT: You have access to conversation context and user information. Use this information to provide personalized, contextually aware responses. Remember previous conversations and user details."""
            
            # Build context-aware prompt
            full_prompt = system_prompt
            
            # Add context information if available
            if context:
                # Add conversation context
                if 'conversation_context' in context and context['conversation_context']:
                    full_prompt += f"\n\nCONTEXT INFORMATION:\n{context['conversation_context']}"
                
                # Add any other relevant context
                for key, value in context.items():
                    if key != 'conversation_context' and value:
                        full_prompt += f"\n{key.replace('_', ' ').title()}: {value}"
            
            # Add the user input
            full_prompt += f"\n\nUser: {user_input}\nJarvis:"
            
            # Call Ollama via subprocess for reliability
            result = subprocess.run(
                ["ollama", "run", self.model_name, full_prompt],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Clean up the response
                if response:
                    # Remove any unwanted prefixes or suffixes
                    response = response.replace("Jarvis:", "").strip()
                    
                    # Add conversation to history
                    self.conversation_history.append({
                        "user": user_input,
                        "assistant": response
                    })
                    
                    # Keep history manageable
                    if len(self.conversation_history) > self.max_history:
                        self.conversation_history = self.conversation_history[-self.max_history:]
                    
                    return response
                else:
                    return "I'm sorry sir, I couldn't generate a proper response."
            else:
                logger.error(f"Ollama error: {result.stderr}")
                return "I encountered an error processing your request, sir."
                
        except subprocess.TimeoutExpired:
            logger.warning("Local AI request timed out")
            return "I'm sorry sir, that request is taking too long to process."
        except Exception as e:
            logger.error(f"Local brain processing error: {e}")
            return "I encountered an error processing your request, sir."
    
    def is_healthy(self) -> bool:
        """Check if Ollama is running and responsive"""
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "run", self.model_name, "Hello"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False

class AIBrainManager:
    """Main AI brain manager with provider orchestration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.brains: Dict[BrainProvider, BaseBrain] = {}
        self.primary_brain: Optional[BaseBrain] = None
        self.fallback_brain: Optional[BaseBrain] = None
        
        self._initialize_brains()
        self._setup_provider_hierarchy()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "providers": {
                "anthropic": {
                    "enabled": True,
                    "model": "claude-3-haiku-20240307",
                    "priority": 1
                },
                "openai": {
                    "enabled": True, 
                    "model": "gpt-3.5-turbo",
                    "priority": 2
                },
                "local": {
                    "enabled": False,
                    "priority": 3
                }
            },
            "fallback_enabled": True,
            "health_check_interval": 300  # 5 minutes
        }
    
    def _initialize_brains(self):
        """Initialize available brain providers"""
        providers_config = self.config.get("providers", {})
        
        # Initialize Anthropic
        if providers_config.get("anthropic", {}).get("enabled", False):
            model = providers_config["anthropic"].get("model", "claude-3-haiku-20240307")
            brain = AnthropicBrain(model=model)
            if brain.available:
                self.brains[BrainProvider.ANTHROPIC] = brain
        
        # Initialize DeepSeek
        if providers_config.get("deepseek", {}).get("enabled", False):
            model = providers_config["deepseek"].get("model", "deepseek-chat")
            brain = DeepSeekBrain(model=model)
            if brain.available:
                self.brains[BrainProvider.DEEPSEEK] = brain
        
        # Initialize Local (when implemented)
        if providers_config.get("local", {}).get("enabled", False):
            brain = LocalBrain()
            if brain.available:
                self.brains[BrainProvider.LOCAL] = brain
        
        logger.info(f"Initialized {len(self.brains)} brain providers: {list(self.brains.keys())}")
    
    def _setup_provider_hierarchy(self):
        """Setup primary and fallback providers based on priority"""
        if not self.brains:
            logger.warning("No AI brain providers available")
            return
        
        # Sort providers by priority
        providers_config = self.config.get("providers", {})
        available_providers = []
        
        for provider_enum, brain in self.brains.items():
            provider_name = provider_enum.value
            priority = providers_config.get(provider_name, {}).get("priority", 999)
            available_providers.append((priority, provider_enum, brain))
        
        available_providers.sort(key=lambda x: x[0])  # Sort by priority
        
        # Set primary and fallback
        if available_providers:
            self.primary_brain = available_providers[0][2]
            logger.info(f"Primary brain: {self.primary_brain.provider_name}")
            
            if len(available_providers) > 1 and self.config.get("fallback_enabled", True):
                self.fallback_brain = available_providers[1][2]
                logger.info(f"Fallback brain: {self.fallback_brain.provider_name}")
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process a request through available brain providers"""
        if not self.primary_brain:
            return "I'm sorry sir, my intelligence systems are currently offline."
        
        # Try primary brain first
        try:
            response = self.primary_brain.process_request(user_input, context)
            
            # Check if response indicates failure
            failure_indicators = ["error", "sorry", "trouble", "difficulties", "offline", "not available"]
            response_lower = response.lower()
            
            if any(indicator in response_lower for indicator in failure_indicators):
                logger.warning("Primary brain response indicates failure")
                if self.fallback_brain:
                    return self._try_fallback(user_input, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Primary brain failed: {e}")
            if self.fallback_brain:
                return self._try_fallback(user_input, context)
            else:
                return "I'm experiencing technical difficulties, sir."
    
    def _try_fallback(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Try fallback brain provider"""
        try:
            logger.info("Attempting fallback brain")
            response = self.fallback_brain.process_request(user_input, context)
            return response
        except Exception as e:
            logger.error(f"Fallback brain also failed: {e}")
            return "I'm experiencing difficulties across all my intelligence systems, sir."
    
    def is_available(self) -> bool:
        """Check if any brain provider is available"""
        return self.primary_brain is not None or self.fallback_brain is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all brain providers"""
        status = {
            "available_providers": len(self.brains),
            "primary": self.primary_brain.provider_name if self.primary_brain else None,
            "fallback": self.fallback_brain.provider_name if self.fallback_brain else None,
            "providers": {}
        }
        
        for provider_enum, brain in self.brains.items():
            status["providers"][provider_enum.value] = {
                "available": brain.available,
                "healthy": brain.is_healthy()
            }
        
        return status
    
    def clear_all_history(self):
        """Clear conversation history for all providers"""
        for brain in self.brains.values():
            brain.clear_history()
        logger.info("All brain provider histories cleared")

# Convenience function for easy integration
def create_ai_brain(config: Optional[Dict[str, Any]] = None) -> AIBrainManager:
    """Create and return an AI brain manager instance"""
    return AIBrainManager(config)

# Example usage and testing
if __name__ == "__main__":
    # Test the AI brain manager
    brain_manager = create_ai_brain()
    
    if brain_manager.is_available():
        print("AI Brain Manager initialized successfully!")
        print(f"Status: {json.dumps(brain_manager.get_status(), indent=2)}")
        
        # Test some requests
        test_requests = [
            "What time is it?",
            "Tell me a joke about AI assistants", 
            "Explain quantum computing in simple terms",
            "What's the weather like on Mars?"
        ]
        
        context = {
            "time": "3:30 PM",
            "date": "Saturday, July 19, 2025",
            "location": "Lab"
        }
        
        for request in test_requests:
            print(f"\nUser: {request}")
            try:
                response = brain_manager.process_request(request, context)
                print(f"Jarvis: {response}")
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        print("No AI brain providers available.")
        print("Set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY environment variables.")
