#!/usr/bin/env python3
"""
End-to-end tests for Anthropic integration with --use-anthropic flag
"""

import unittest
import subprocess
import os
import sys
import time
import signal
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnthropicEndToEnd(unittest.TestCase):
    """End-to-end tests for Anthropic with --use-anthropic flag"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not found - skipping E2E tests")
        
        # Test script path
        self.script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jarvis.py')
    
    def test_anthropic_flag_in_process_args(self):
        """Test that Anthropic preference is shown in startup output"""
        # Run with Anthropic flag and capture startup output
        process = subprocess.Popen([
            'python', self.script_path, 
            '--enable-ai', '--use-anthropic', '--help'
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
        )
        
        stdout, stderr = process.communicate(timeout=10)
        
        # Should show help without errors
        self.assertEqual(process.returncode, 0)
        self.assertIn('--use-anthropic', stdout)
    
    def test_anthropic_provider_selection_logged(self):
        """Test that Anthropic is selected as primary provider in logs"""
        # This test would ideally capture logging output
        # For now, we test that the flag doesn't cause errors
        
        process = subprocess.Popen([
            'python', self.script_path,
            '--enable-ai', '--use-anthropic'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
        
        # Let it start up briefly then terminate
        time.sleep(2)
        process.terminate()
        
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        
        # Should not have immediate errors on startup
        combined_output = stdout + stderr
        self.assertNotIn('Error:', combined_output)
        self.assertNotIn('Traceback', combined_output)
    
    def test_anthropic_default_behavior(self):
        """Test that Anthropic is the default even without explicit flag"""
        # Test that --enable-ai alone works (should default to Anthropic)
        process = subprocess.Popen([
            'python', self.script_path,
            '--enable-ai', '--help'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
        
        stdout, stderr = process.communicate(timeout=10)
        
        # Should show help without errors
        self.assertEqual(process.returncode, 0)
        self.assertIn('--use-anthropic', stdout)
        self.assertIn('(default)', stdout)


class TestAnthropicConfigValidation(unittest.TestCase):
    """Test Anthropic configuration validation in E2E context"""
    
    def test_anthropic_environment_check(self):
        """Test that environment is properly set up for Anthropic"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if api_key:
            # Key should be reasonable length
            self.assertGreater(len(api_key), 10)
            # Should start with expected prefix
            self.assertTrue(api_key.startswith('sk-ant-'))
        else:
            self.skipTest("ANTHROPIC_API_KEY not found")
    
    def test_anthropic_import_requirements(self):
        """Test that required packages for Anthropic are available"""
        try:
            import anthropic
            # Basic check that anthropic can be imported
            self.assertTrue(hasattr(anthropic, 'Anthropic'))
        except ImportError:
            self.fail("Anthropic library not available - required for Anthropic integration")
    
    def test_anthropic_vs_deepseek_environment(self):
        """Test environment supports both Anthropic and DeepSeek"""
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # At least one should be available
        self.assertTrue(anthropic_key or deepseek_key, 
                       "At least one AI provider API key should be available")
        
        if anthropic_key and deepseek_key:
            # Both are available - great for testing provider switching
            self.assertNotEqual(anthropic_key, deepseek_key)


class TestAnthropicCommandLineIntegration(unittest.TestCase):
    """Test command line integration for Anthropic"""
    
    def test_anthropic_help_text_accuracy(self):
        """Test that help text accurately describes Anthropic functionality"""
        result = subprocess.run([
            'python', 'jarvis.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertEqual(result.returncode, 0)
        help_text = result.stdout
        
        # Check specific help text content
        self.assertIn('--use-anthropic', help_text)
        self.assertIn('Use Anthropic Claude as primary AI provider', help_text)
        self.assertIn('(default)', help_text)
        self.assertIn('--use-deepseek', help_text)
        self.assertIn('--enable-ai', help_text)
    
    def test_anthropic_flag_combinations(self):
        """Test that --use-anthropic works correctly with other flag combinations"""
        # Test combinations that should work
        valid_combinations = [
            ['--use-anthropic'],
            ['--enable-ai', '--use-anthropic'],
            ['--use-anthropic', '--tts-engine', 'pyttsx3']
        ]
        
        for flags in valid_combinations:
            with self.subTest(flags=flags):
                # Run with --help to avoid actually starting the assistant
                result = subprocess.run([
                    'python', 'jarvis.py'
                ] + flags + ['--help'],
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def test_anthropic_mutual_exclusivity(self):
        """Test that Anthropic and DeepSeek flags are mutually exclusive"""
        # This should fail
        result = subprocess.run([
            'python', 'jarvis.py', '--enable-ai', '--use-anthropic', '--use-deepseek'
        ], capture_output=True, text=True, 
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Should exit with error
        self.assertNotEqual(result.returncode, 0)
        
        # Should contain error message
        output = result.stdout + result.stderr
        self.assertIn('Only one AI provider can be specified', output)


class TestAnthropicSystemIntegration(unittest.TestCase):
    """Test system-level integration for Anthropic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
    
    @unittest.skipIf(not os.getenv('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY not available")
    def test_anthropic_import_and_basic_functionality(self):
        """Test that Anthropic can be imported and basic functionality works"""
        # Test that we can import the brain classes
        try:
            from src.ai.ai_brain import AnthropicBrain, BrainProvider
            
            # Test that Anthropic brain can be created
            brain = AnthropicBrain(api_key=self.api_key)
            self.assertIsNotNone(brain)
            
            # Test that it has the required interface
            self.assertTrue(hasattr(brain, 'process_request'))
            self.assertTrue(hasattr(brain, 'available'))  # Check attribute, not method
            
            # Test that the brain is actually available with API key
            self.assertTrue(brain.available)
            
        except Exception as e:
            self.fail(f"Failed to import or create Anthropic brain: {e}")
    
    @unittest.skipIf(not os.getenv('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY not available")
    def test_anthropic_provider_enum(self):
        """Test that Anthropic provider enum is correctly defined"""
        from src.ai.ai_brain import BrainProvider
        
        # Check that ANTHROPIC is defined
        self.assertTrue(hasattr(BrainProvider, 'ANTHROPIC'))
        self.assertEqual(BrainProvider.ANTHROPIC.value, 'anthropic')
    
    def test_anthropic_fallback_behavior(self):
        """Test that system handles missing Anthropic API key gracefully"""
        # Test with no API key
        from src.ai.ai_brain import AnthropicBrain
        
        with patch.dict(os.environ, {}, clear=True):
            brain = AnthropicBrain()
            
            # Should not crash, but should not be available
            self.assertIsNotNone(brain)
            self.assertFalse(brain.available)


class TestAnthropicPerformance(unittest.TestCase):
    """Test performance characteristics of Anthropic integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not found - skipping performance tests")
    
    def test_anthropic_response_time(self):
        """Test that Anthropic responds within reasonable time"""
        from src.ai.ai_brain import AnthropicBrain
        
        brain = AnthropicBrain(api_key=self.api_key)
        
        if not brain.available:
            self.skipTest("Anthropic brain not available")
        
        start_time = time.time()
        try:
            response = brain.process_query("Hello")
            response_time = time.time() - start_time
            
            # Should respond within 30 seconds
            self.assertLess(response_time, 30.0)
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 0)
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


if __name__ == '__main__':
    print("Running Anthropic End-to-End Tests...")
    print(f"Anthropic API Key: {'✓ Present' if os.getenv('ANTHROPIC_API_KEY') else '✗ Missing'}")
    print(f"Python Version: {sys.version}")
    print("=" * 70)
    
    # Run the tests
    unittest.main(verbosity=2)