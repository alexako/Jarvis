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
import os
import sys
import json
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

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
        # Generate a more descriptive name based on the actual model
        model_display_name = self._get_model_display_name(model_name)
        super().__init__(f"Local {model_display_name}")
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
        logger.info(f"LocalBrain constructor called with model_name: {model_name}")
        
        # Test Ollama connection
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info(f"Ollama models available:\n{result.stdout}")
                if model_name in result.stdout:
                    self.available = True
                    logger.info(f"Local brain initialized with {model_name}")
                else:
                    logger.warning(f"Model {model_name} not found in Ollama")
                    # Let's try to pull the model
                    logger.info(f"Attempting to pull model {model_name}")
                    pull_result = subprocess.run(
                        ["ollama", "pull", model_name],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout for pulling
                    )
                    if pull_result.returncode == 0:
                        self.available = True
                        logger.info(f"Successfully pulled and initialized {model_name}")
                    else:
                        logger.error(f"Failed to pull {model_name}: {pull_result.stderr}")
                        self.available = False
            else:
                logger.error(f"Ollama list failed: {result.stderr}")
                self.available = False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Ollama not available: {e}")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize local brain: {e}")
            self.available = False
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process request using local Ollama model"""
        logger.info("=== LocalBrain.process_request START ===")
        logger.info(f"user_input: {user_input}")
        logger.info(f"context: {context}")
        logger.info(f"self.available: {self.available}")
        
        if not self.available:
            logger.warning("LocalBrain not available")
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
            
            logger.info(f"Calling ollama with model: {self.model_name}")
            logger.info(f"Full prompt: {full_prompt}")
            
            # Call Ollama via subprocess for reliability
            result = subprocess.run(
                ["ollama", "run", self.model_name, full_prompt],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            logger.info(f"Ollama result returncode: {result.returncode}")
            logger.info(f"Ollama result stdout: {result.stdout}")
            logger.info(f"Ollama result stderr: {result.stderr}")
            
            if result.returncode == 0:
                response = result.stdout.strip()
                logger.info(f"Raw response from ollama: {response}")
                
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
                    
                    logger.info("=== LocalBrain.process_request END (success) ===")
                    return response
                else:
                    logger.warning("Empty response from ollama")
                    logger.info("=== LocalBrain.process_request END (empty response) ===")
                    return "I'm sorry sir, I couldn't generate a proper response."
            else:
                logger.error(f"Ollama error: {result.stderr}")
                logger.info("=== LocalBrain.process_request END (ollama error) ===")
                return "I encountered an error processing your request, sir."
                
        except subprocess.TimeoutExpired:
            logger.warning("Local AI request timed out")
            logger.info("=== LocalBrain.process_request END (timeout) ===")
            return "I'm sorry sir, that request is taking too long to process."
        except Exception as e:
            logger.error(f"Local brain processing error: {e}")
            logger.info("=== LocalBrain.process_request END (exception) ===")
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

    def _get_model_display_name(self, model_name: str) -> str:
        """Convert model name to a user-friendly display name"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            # Extract version if present
            if "3.2" in model_name:
                return "Llama 3.2"
            elif "3.1" in model_name:
                return "Llama 3.1"
            elif "3" in model_name:
                return "Llama 3"
            else:
                return "Llama"
        elif "phi" in model_name_lower:
            if "3.5" in model_name:
                return "Phi-3.5"
            else:
                return "Phi"
        elif "qwen" in model_name_lower:
            return "Qwen"
        elif "gemma" in model_name_lower:
            return "Gemma"
        else:
            # For unknown models, just capitalize the first part before the colon
            base_model = model_name.split(":")[0]
            return base_model.capitalize()

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
        """Initialize available brain providers with lazy loading"""
        providers_config = self.config.get("providers", {})
        
        logger.info(f"AIBrainManager config: {self.config}")
        logger.info(f"Providers config: {providers_config}")
        
        # Store configuration for lazy initialization
        self._provider_configs = {}
        
        # Only store configs for enabled providers, don't initialize yet
        for provider_name, provider_config in providers_config.items():
            logger.info(f"Checking provider {provider_name}: {provider_config}")
            if provider_config.get("enabled", False):
                self._provider_configs[provider_name] = provider_config
                logger.info(f"Enabled provider {provider_name}")
            else:
                logger.info(f"Provider {provider_name} is disabled")
        
        logger.info(f"Deferred initialization of {len(self._provider_configs)} brain providers: {list(self._provider_configs.keys())}")
    
    def _get_or_create_brain(self, provider_enum: BrainProvider):
        """Get or create a brain instance with lazy initialization"""
        if provider_enum not in self.brains:
            # Initialize the brain on first access
            provider_name = provider_enum.value
            if provider_name in self._provider_configs:
                provider_config = self._provider_configs[provider_name]
                
                if provider_enum == BrainProvider.ANTHROPIC:
                    model = provider_config.get("model", "claude-3-haiku-20240307")
                    brain = AnthropicBrain(model=model)
                elif provider_enum == BrainProvider.DEEPSEEK:
                    model = provider_config.get("model", "deepseek-chat")
                    brain = DeepSeekBrain(model=model)
                elif provider_enum == BrainProvider.LOCAL:
                    model = provider_config.get("model", "llama3.2:latest")
                    logger.info(f"Initializing LocalBrain with model: {model}")
                    brain = LocalBrain(model_name=model)
                else:
                    return None
                
                if brain.available:
                    self.brains[provider_enum] = brain
                    logger.info(f"Initialized {brain.provider_name} brain on demand")
                else:
                    logger.warning(f"Failed to initialize {provider_name} brain")
                    return None
        
        return self.brains.get(provider_enum)
    
    def get_primary_brain(self) -> Optional[BaseBrain]:
        """Get the primary brain provider (initializing it if needed)"""
        logger.info(f"Getting primary brain, _primary_provider: {getattr(self, '_primary_provider', None)}")
        if hasattr(self, '_primary_provider'):
            brain = self._get_or_create_brain(self._primary_provider)
            logger.info(f"Primary brain retrieved: {brain.provider_name if brain else None}")
            return brain
        logger.info("No primary provider found")
        return None
    
    def get_fallback_brain(self) -> Optional[BaseBrain]:
        """Get the fallback brain provider (initializing it if needed)"""
        logger.info(f"Getting fallback brain, _fallback_provider: {getattr(self, '_fallback_provider', None)}")
        if hasattr(self, '_fallback_provider'):
            brain = self._get_or_create_brain(self._fallback_provider)
            logger.info(f"Fallback brain retrieved: {brain.provider_name if brain else None}")
            return brain
        logger.info("No fallback provider found")
        return None
    
    def _setup_provider_hierarchy(self):
        """Setup primary and fallback providers based on priority with lazy loading"""
        if not self._provider_configs:
            logger.warning("No AI brain providers configured")
            return
        
        # Sort providers by priority
        available_providers = []
        
        for provider_name, provider_config in self._provider_configs.items():
            try:
                provider_enum = BrainProvider(provider_name)
                priority = provider_config.get("priority", 999)
                available_providers.append((priority, provider_enum, provider_config))
            except ValueError:
                logger.warning(f"Unknown provider name: {provider_name}")
        
        available_providers.sort(key=lambda x: x[0])  # Sort by priority
        
        # Set primary and fallback (initialize on demand)
        if available_providers:
            self._primary_provider = available_providers[0][1]
            self.primary_brain = None  # Will be initialized on first use
            
            logger.info(f"Primary brain provider: {self._primary_provider.value}")
            
            if len(available_providers) > 1 and self.config.get("fallback_enabled", True):
                self._fallback_provider = available_providers[1][1]
                self.fallback_brain = None  # Will be initialized on first use
                logger.info(f"Fallback brain provider: {self._fallback_provider.value}")
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process a request through available brain providers"""
        logger.info("=== AIBrainManager.process_request START ===")
        logger.info(f"Processing request with user_input: {user_input}")
        logger.info(f"self.primary_brain: {self.primary_brain}")
        logger.info(f"hasattr(self, '_primary_provider'): {hasattr(self, '_primary_provider')}")
        
        # Initialize primary brain on first use
        if self.primary_brain is None and hasattr(self, '_primary_provider'):
            logger.info("Initializing primary brain on first use")
            self.primary_brain = self._get_or_create_brain(self._primary_provider)
            logger.info(f"Primary brain initialized: {self.primary_brain}")
        
        logger.info(f"After initialization - self.primary_brain: {self.primary_brain}")
        
        if not self.primary_brain:
            logger.warning("No primary brain available")
            logger.info("=== AIBrainManager.process_request END (no primary brain) ===")
            return "I'm sorry sir, my intelligence systems are currently offline."
        
        logger.info(f"Using primary brain: {self.primary_brain.provider_name}")
        logger.info(f"Primary brain available: {self.primary_brain.available}")
        
        # Try primary brain first
        try:
            logger.info("Calling primary brain process_request")
            response = self.primary_brain.process_request(user_input, context)
            logger.info(f"Primary brain response: {response}")
            
            # Check if response indicates failure
            failure_indicators = ["error", "sorry", "trouble", "difficulties", "offline", "not available"]
            response_lower = response.lower()
            
            if any(indicator in response_lower for indicator in failure_indicators):
                logger.warning("Primary brain response indicates failure")
                # Initialize fallback brain on first use
                if self.fallback_brain is None and hasattr(self, '_fallback_provider'):
                    self.fallback_brain = self._get_or_create_brain(self._fallback_provider)
                
                if self.fallback_brain:
                    logger.info("Using fallback brain")
                    result = self._try_fallback(user_input, context)
                    logger.info("=== AIBrainManager.process_request END (with fallback) ===")
                    return result
            
            logger.info("=== AIBrainManager.process_request END (success) ===")
            return response
            
        except Exception as e:
            logger.error(f"Primary brain failed: {e}")
            # Initialize fallback brain on first use
            if self.fallback_brain is None and hasattr(self, '_fallback_provider'):
                self.fallback_brain = self._get_or_create_brain(self._fallback_provider)
            
            if self.fallback_brain:
                logger.info("Using fallback brain after exception")
                result = self._try_fallback(user_input, context)
                logger.info("=== AIBrainManager.process_request END (with fallback after exception) ===")
                return result
            else:
                logger.info("=== AIBrainManager.process_request END (technical difficulties) ===")
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
        # Check if we have any configured providers
        return bool(self._provider_configs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all brain providers"""
        logger.info("get_status called")
        logger.info(f"self._provider_configs: {self._provider_configs}")
        
        # Initialize primary brain if not already done
        if self.primary_brain is None and hasattr(self, '_primary_provider'):
            logger.info("Initializing primary brain for status check")
            self.primary_brain = self._get_or_create_brain(self._primary_provider)
            logger.info(f"Primary brain after initialization: {self.primary_brain}")
        
        # Initialize fallback brain if not already done
        if self.fallback_brain is None and hasattr(self, '_fallback_provider'):
            logger.info("Initializing fallback brain for status check")
            self.fallback_brain = self._get_or_create_brain(self._fallback_provider)
            logger.info(f"Fallback brain after initialization: {self.fallback_brain}")
        
        status = {
            "available_providers": len(self._provider_configs),
            "primary": self.primary_brain.provider_name if self.primary_brain else None,
            "fallback": self.fallback_brain.provider_name if self.fallback_brain else None,
            "providers": {}
        }
        
        logger.info(f"Initial status: {status}")
        
        # Add status for all configured providers
        for provider_name, provider_config in self._provider_configs.items():
            logger.info(f"Checking status for provider: {provider_name}")
            try:
                provider_enum = BrainProvider(provider_name)
                logger.info(f"Provider enum: {provider_enum}")
                # Get or create brain for status check
                brain = self._get_or_create_brain(provider_enum)
                logger.info(f"Brain for {provider_name}: {brain}")
                status["providers"][provider_name] = {
                    "available": brain.available if brain else False,
                    "healthy": brain.is_healthy() if brain and brain.available else False
                }
                logger.info(f"Status for {provider_name}: {status['providers'][provider_name]}")
            except ValueError as e:
                logger.error(f"Error processing provider {provider_name}: {e}")
                status["providers"][provider_name] = {
                    "available": False,
                    "healthy": False
                }
        
        logger.info(f"Final status: {status}")
        return status
    
    def clear_all_history(self):
        """Clear conversation history for all providers"""
        # Initialize all brains to clear their history
        for provider_name in self._provider_configs.keys():
            try:
                provider_enum = BrainProvider(provider_name)
                brain = self._get_or_create_brain(provider_enum)
                if brain:
                    brain.clear_history()
            except ValueError:
                pass
        
        logger.info("All brain provider histories cleared")
    
    def is_available(self) -> bool:
        """Check if any brain provider is available"""
        # Check if we have any configured providers
        return bool(self._provider_configs)
    
    def clear_all_history(self):
        """Clear conversation history for all providers"""
        # Initialize all brains to clear their history
        for provider_name in self._provider_configs.keys():
            try:
                provider_enum = BrainProvider(provider_name)
                brain = self._get_or_create_brain(provider_enum)
                if brain:
                    brain.clear_history()
            except ValueError:
                pass
        
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
