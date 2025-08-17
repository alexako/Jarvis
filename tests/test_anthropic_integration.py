#!/usr/bin/env python3
"""
Integration tests for Anthropic (Claude) AI functionality
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.ai_brain import AnthropicBrain, BrainProvider, AIBrainManager
from src.commands.commands import create_ai_config


class TestAnthropicIntegration(unittest.TestCase):
    """Integration tests for Anthropic AI brain functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Check if Anthropic API key is available
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not found in environment")
    
    def test_anthropic_brain_initialization(self):
        """Test that Anthropic brain can be initialized"""
        brain = AnthropicBrain(api_key=self.api_key)
        self.assertIsNotNone(brain)
        self.assertEqual(brain.provider, BrainProvider.ANTHROPIC)
    
    def test_anthropic_api_configuration(self):
        """Test that Anthropic brain is configured with correct API settings"""
        brain = AnthropicBrain(api_key=self.api_key)
        
        # Check that the client is initialized
        self.assertIsNotNone(brain.client)
        
        # Check API key is set (without exposing it)
        self.assertTrue(len(self.api_key) > 10)
    
    def test_anthropic_model_setting(self):
        """Test that Anthropic brain uses the correct model"""
        brain = AnthropicBrain(api_key=self.api_key)
        
        # Default model should be claude-3-haiku-20240307
        self.assertEqual(brain.model_name, "claude-3-haiku-20240307")
        
        # Test custom model setting
        custom_brain = AnthropicBrain(api_key=self.api_key, model="claude-3-sonnet-20240229")
        self.assertEqual(custom_brain.model_name, "claude-3-sonnet-20240229")
    
    @patch('src.ai.ai_brain.AnthropicBrain.process_request')
    def test_anthropic_simple_query(self, mock_process_request):
        """Test that Anthropic brain can handle a simple query"""
        # Mock the API response
        mock_process_request.return_value = "Good day, sir! How may I assist you?"
        
        brain = AnthropicBrain(api_key=self.api_key)
        response = brain.process_request("Hello")
        
        self.assertEqual(response, "Good day, sir! How may I assist you?")
        mock_process_request.assert_called_once_with("Hello")
    
    @patch('src.ai.ai_brain.AnthropicBrain.process_request')
    def test_anthropic_error_handling(self, mock_process_request):
        """Test that Anthropic brain handles API errors gracefully"""
        # Mock an API error
        mock_process_request.side_effect = Exception("API Error")
        
        brain = AnthropicBrain(api_key=self.api_key)
        
        # Should raise the exception (let the caller handle it)
        with self.assertRaises(Exception):
            brain.process_request("Hello")
    
    def test_ai_config_with_anthropic_preference(self):
        """Test that AI config correctly prioritizes Anthropic when requested"""
        config = create_ai_config(
            prefer_anthropic=True
        )
        
        # Check that Anthropic has priority 1 (primary)
        anthropic_config = config['providers']['anthropic']
        deepseek_config = config['providers']['deepseek']
        
        self.assertIsNotNone(anthropic_config)
        self.assertIsNotNone(deepseek_config)
        self.assertEqual(anthropic_config['priority'], 1)
        self.assertEqual(deepseek_config['priority'], 2)
    
    def test_ai_config_anthropic_default(self):
        """Test that Anthropic is prioritized by default"""
        config = create_ai_config()  # No preference specified
        
        # Check that Anthropic has priority 1 by default
        anthropic_config = config['providers']['anthropic']
        deepseek_config = config['providers']['deepseek']
        
        self.assertEqual(anthropic_config['priority'], 1)
        self.assertEqual(deepseek_config['priority'], 2)
    
    def test_ai_brain_manager_with_anthropic(self):
        """Test that AIBrainManager correctly handles Anthropic priority"""
        config = create_ai_config(
            prefer_anthropic=True
        )
        
        # Mock environment variables
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.api_key}):
            manager = AIBrainManager(config)
            
            # Check that manager was initialized
            self.assertIsNotNone(manager)
            
            # Trigger initialization of the Anthropic brain by accessing it
            # This is needed because of lazy loading
            anthropic_brain = manager._get_or_create_brain(BrainProvider.ANTHROPIC)
            
            # Check that Anthropic is available in the brains dictionary
            self.assertIn(BrainProvider.ANTHROPIC, manager.brains)
            
            # Check that the Anthropic brain is available
            self.assertTrue(anthropic_brain.available)


class TestAnthropicLiveAPI(unittest.TestCase):
    """Live API tests for Anthropic (requires actual API key)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not found - skipping live API tests")
    
    def test_anthropic_live_simple_query(self):
        """Test a simple live query to Anthropic API"""
        brain = AnthropicBrain(api_key=self.api_key)
        
        try:
            response = brain.process_query("Say 'Hello World' and nothing else.")
            
            # Response should contain "Hello World" and be reasonably short
            self.assertIsNotNone(response)
            self.assertIn("Hello", response)
            self.assertLess(len(response), 200)  # Should be a short response
            
        except Exception as e:
            self.fail(f"Live API test failed: {e}")
    
    def test_anthropic_live_math_query(self):
        """Test a math query to verify Anthropic reasoning"""
        brain = AnthropicBrain(api_key=self.api_key)
        
        try:
            response = brain.process_query("What is 7 * 8? Answer with just the number.")
            
            # Response should contain "56"
            self.assertIsNotNone(response)
            self.assertIn("56", response)
            
        except Exception as e:
            self.fail(f"Live API math test failed: {e}")
    
    def test_anthropic_live_jarvis_personality(self):
        """Test that Anthropic responds with Jarvis personality"""
        brain = AnthropicBrain(api_key=self.api_key)
        
        try:
            response = brain.process_query("Who are you?")
            
            # Response should indicate Jarvis personality
            self.assertIsNotNone(response)
            response_lower = response.lower()
            # Should contain jarvis or reference to being an AI assistant
            jarvis_indicators = ["jarvis", "ai assistant", "assistant", "sir"]
            self.assertTrue(any(indicator in response_lower for indicator in jarvis_indicators))
            
        except Exception as e:
            self.fail(f"Live API personality test failed: {e}")
    
    def test_anthropic_live_context_handling(self):
        """Test that Anthropic handles context properly"""
        brain = AnthropicBrain(api_key=self.api_key)
        
        try:
            context = {"user_name": "Tony", "location": "Malibu"}
            response = brain.process_query("Where am I?", context=context)
            
            # Response should reference the location from context
            self.assertIsNotNone(response)
            self.assertIn("Malibu", response)
            
        except Exception as e:
            self.fail(f"Live API context test failed: {e}")


class TestAnthropicVsDeepSeek(unittest.TestCase):
    """Test Anthropic vs DeepSeek provider priority behavior"""
    
    def test_anthropic_primary_deepseek_fallback(self):
        """Test that when Anthropic is primary, DeepSeek is fallback"""
        config = create_ai_config(prefer_anthropic=True)
        
        anthropic_config = config['providers']['anthropic']
        deepseek_config = config['providers']['deepseek']
        
        self.assertEqual(anthropic_config['priority'], 1)
        self.assertEqual(deepseek_config['priority'], 2)
        self.assertTrue(anthropic_config['enabled'])
        self.assertTrue(deepseek_config['enabled'])
    
    def test_deepseek_primary_anthropic_fallback(self):
        """Test that when DeepSeek is primary, Anthropic is fallback"""
        config = create_ai_config(prefer_anthropic=False)
        
        anthropic_config = config['providers']['anthropic']
        deepseek_config = config['providers']['deepseek']
        
        self.assertEqual(anthropic_config['priority'], 2)
        self.assertEqual(deepseek_config['priority'], 1)
        self.assertTrue(anthropic_config['enabled'])
        self.assertTrue(deepseek_config['enabled'])


if __name__ == '__main__':
    # Run the integration tests
    print("Running Anthropic Integration Tests...")
    print(f"Anthropic API Key: {'✓ Present' if os.getenv('ANTHROPIC_API_KEY') else '✗ Missing'}")
    print("=" * 60)
    
    unittest.main(verbosity=2)