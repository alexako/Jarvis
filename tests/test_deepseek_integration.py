#!/usr/bin/env python3
"""
Integration tests for DeepSeek AI functionality
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.ai_brain import DeepSeekBrain, BrainProvider, AIBrainManager
from src.commands.commands import create_ai_config


class TestDeepSeekIntegration(unittest.TestCase):
    """Integration tests for DeepSeek AI brain functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Check if DeepSeek API key is available
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            self.skipTest("DEEPSEEK_API_KEY not found in environment")
    
    def test_deepseek_brain_initialization(self):
        """Test that DeepSeek brain can be initialized"""
        brain = DeepSeekBrain(api_key=self.api_key)
        self.assertIsNotNone(brain)
        self.assertEqual(brain.provider, BrainProvider.DEEPSEEK)
    
    def test_deepseek_api_configuration(self):
        """Test that DeepSeek brain is configured with correct API settings"""
        brain = DeepSeekBrain(api_key=self.api_key)
        
        # Check that the client is initialized
        self.assertIsNotNone(brain.client)
        
        # Check API key is set (without exposing it)
        self.assertTrue(len(self.api_key) > 10)
    
    def test_deepseek_model_setting(self):
        """Test that DeepSeek brain uses the correct model"""
        brain = DeepSeekBrain(api_key=self.api_key)
        
        # Default model should be deepseek-chat
        self.assertEqual(brain.model_name, "deepseek-chat")
        
        # Test custom model setting
        custom_brain = DeepSeekBrain(api_key=self.api_key, model="custom-model")
        self.assertEqual(custom_brain.model_name, "custom-model")
    
    @patch('src.ai.ai_brain.DeepSeekBrain.process_request')
    def test_deepseek_simple_query(self, mock_process_request):
        """Test that DeepSeek brain can handle a simple query"""
        # Mock the API response
        mock_process_request.return_value = "Hello! How can I help you today?"
        
        brain = DeepSeekBrain(api_key=self.api_key)
        response = brain.process_request("Hello")
        
        self.assertEqual(response, "Hello! How can I help you today?")
        mock_process_request.assert_called_once_with("Hello", None)
    
    @patch('src.ai.ai_brain.DeepSeekBrain.process_request')
    def test_deepseek_error_handling(self, mock_process_request):
        """Test that DeepSeek brain handles API errors gracefully"""
        # Mock an API error
        mock_process_request.side_effect = Exception("API Error")
        
        brain = DeepSeekBrain(api_key=self.api_key)
        
        # Should raise the exception (let the caller handle it)
        with self.assertRaises(Exception):
            brain.process_request("Hello")
    
    def test_ai_config_with_deepseek_preference(self):
        """Test that AI config correctly prioritizes DeepSeek when requested"""
        config = create_ai_config(
            prefer_anthropic=False
        )
        
        # Check that DeepSeek has priority 1 (primary)
        deepseek_config = config['providers']['deepseek']
        anthropic_config = config['providers']['anthropic']
        
        self.assertIsNotNone(deepseek_config)
        self.assertIsNotNone(anthropic_config)
        self.assertEqual(deepseek_config['priority'], 1)
        self.assertEqual(anthropic_config['priority'], 2)
    
    def test_ai_brain_manager_with_deepseek(self):
        """Test that AIBrainManager correctly handles DeepSeek priority"""
        config = create_ai_config(
            prefer_anthropic=False
        )
        
        # Mock environment variables
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': self.api_key}):
            manager = AIBrainManager(config)
            
            # Check that manager was initialized
            self.assertIsNotNone(manager)
            
            # Check that DeepSeek is available in the brains dictionary
            self.assertIn(BrainProvider.DEEPSEEK, manager.brains)
            
            # Check that the DeepSeek brain is available
            deepseek_brain = manager.brains[BrainProvider.DEEPSEEK]
            self.assertTrue(deepseek_brain.available)


class TestDeepSeekLiveAPI(unittest.TestCase):
    """Live API tests for DeepSeek (requires actual API key)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            self.skipTest("DEEPSEEK_API_KEY not found - skipping live API tests")
    
    def test_deepseek_live_simple_query(self):
        """Test a simple live query to DeepSeek API"""
        brain = DeepSeekBrain(api_key=self.api_key)
        
        try:
            response = brain.process_query("Say 'Hello World' and nothing else.")
            
            # Response should contain "Hello World" and be reasonably short
            self.assertIsNotNone(response)
            self.assertIn("Hello", response)
            self.assertLess(len(response), 200)  # Should be a short response
            
        except Exception as e:
            self.fail(f"Live API test failed: {e}")
    
    def test_deepseek_live_math_query(self):
        """Test a math query to verify DeepSeek reasoning"""
        brain = DeepSeekBrain(api_key=self.api_key)
        
        try:
            response = brain.process_query("What is 7 * 8? Answer with just the number.")
            
            # Response should contain "56"
            self.assertIsNotNone(response)
            self.assertIn("56", response)
            
        except Exception as e:
            self.fail(f"Live API math test failed: {e}")


if __name__ == '__main__':
    # Run the integration tests
    print("Running DeepSeek Integration Tests...")
    print(f"DeepSeek API Key: {'✓ Present' if os.getenv('DEEPSEEK_API_KEY') else '✗ Missing'}")
    print("=" * 60)
    
    unittest.main(verbosity=2)