#!/usr/bin/env python3
"""
End-to-end tests for DeepSeek integration with --use-deepseek flag
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


class TestDeepSeekEndToEnd(unittest.TestCase):
    """End-to-end tests for DeepSeek with --use-deepseek flag"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            self.skipTest("DEEPSEEK_API_KEY not found - skipping E2E tests")
        
        # Test script path
        self.script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jarvis_assistant.py')
    
    def test_deepseek_flag_in_process_args(self):
        """Test that DeepSeek preference is shown in startup output"""
        # Run with DeepSeek flag and capture startup output
        process = subprocess.Popen([
            'python', self.script_path, 
            '--enable-ai', '--use-deepseek', '--help'
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
        )
        
        stdout, stderr = process.communicate(timeout=10)
        
        # Should show help without errors
        self.assertEqual(process.returncode, 0)
        self.assertIn('--use-deepseek', stdout)
    
    def test_deepseek_provider_selection_logged(self):
        """Test that DeepSeek is selected as primary provider in logs"""
        # This test would ideally capture logging output
        # For now, we test that the flag doesn't cause errors
        
        process = subprocess.Popen([
            'python', self.script_path,
            '--enable-ai', '--use-deepseek'
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


class TestDeepSeekConfigValidation(unittest.TestCase):
    """Test DeepSeek configuration validation in E2E context"""
    
    def test_deepseek_environment_check(self):
        """Test that environment is properly set up for DeepSeek"""
        api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if api_key:
            # Key should be reasonable length
            self.assertGreater(len(api_key), 10)
            # Should start with expected prefix
            self.assertTrue(api_key.startswith('sk-'))
        else:
            self.skipTest("DEEPSEEK_API_KEY not found")
    
    def test_deepseek_import_requirements(self):
        """Test that required packages for DeepSeek are available"""
        try:
            import openai
            # Basic check that openai can be imported
            self.assertTrue(hasattr(openai, 'OpenAI'))
        except ImportError:
            self.fail("OpenAI library not available - required for DeepSeek")
    
    def test_python_environment_compatibility(self):
        """Test that Python environment supports DeepSeek functionality"""
        # Check Python version
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        self.assertGreaterEqual(version_info.minor, 8)  # Python 3.8+
        
        # Check virtual environment is active
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # Virtual environment is active
            pass
        else:
            # May not be in venv, that's ok for testing
            pass


class TestDeepSeekCommandLineIntegration(unittest.TestCase):
    """Test command line integration for DeepSeek"""
    
    def test_help_text_accuracy(self):
        """Test that help text accurately describes DeepSeek functionality"""
        result = subprocess.run([
            'python', 'jarvis_assistant.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertEqual(result.returncode, 0)
        help_text = result.stdout
        
        # Check specific help text content
        self.assertIn('--use-deepseek', help_text)
        self.assertIn('Use DeepSeek as primary AI provider', help_text)
        self.assertIn('--use-anthropic', help_text)
        self.assertIn('--enable-ai', help_text)
    
    def test_invalid_flag_combinations(self):
        """Test various invalid flag combinations"""
        invalid_combinations = [
            ['--use-anthropic', '--use-deepseek'],
            ['--enable-ai', '--use-anthropic', '--use-deepseek'],
        ]
        
        for flags in invalid_combinations:
            with self.subTest(flags=flags):
                result = subprocess.run([
                    'python', 'jarvis_assistant.py'
                ] + flags, 
                capture_output=True, text=True, 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # Should exit with error
                self.assertNotEqual(result.returncode, 0)
                
                # Should contain error message
                output = result.stdout + result.stderr
                self.assertIn('Only one AI provider can be specified', output)
    
    def test_valid_flag_combinations(self):
        """Test valid flag combinations for help output"""
        valid_combinations = [
            ['--use-deepseek', '--help'],
            ['--enable-ai', '--use-deepseek', '--help'],
            ['--use-anthropic', '--help'],
            ['--enable-ai', '--help'],
        ]
        
        for flags in valid_combinations:
            with self.subTest(flags=flags):
                result = subprocess.run([
                    'python', 'jarvis_assistant.py'
                ] + flags,
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # Should succeed when asking for help
                self.assertEqual(result.returncode, 0)
                self.assertIn('usage:', result.stdout)


class TestDeepSeekSystemIntegration(unittest.TestCase):
    """Test system-level integration for DeepSeek"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
    
    @unittest.skipIf(not os.getenv('DEEPSEEK_API_KEY'), "DEEPSEEK_API_KEY not available")
    def test_deepseek_import_and_basic_functionality(self):
        """Test that DeepSeek can be imported and basic functionality works"""
        # Test that we can import the brain classes
        try:
            from ai_brain import DeepSeekBrain, BrainProvider
            
            # Test that DeepSeek brain can be created
            brain = DeepSeekBrain(api_key=self.api_key)
            self.assertIsNotNone(brain)
            
            # Test that it has the required interface
            self.assertTrue(hasattr(brain, 'process_request'))
            self.assertTrue(hasattr(brain, 'available'))  # Check attribute, not method
            
            # Test that the brain is actually available with API key
            self.assertTrue(brain.available)
            
        except Exception as e:
            self.fail(f"Failed to import or create DeepSeek brain: {e}")


if __name__ == '__main__':
    print("Running DeepSeek End-to-End Tests...")
    print(f"DeepSeek API Key: {'✓ Present' if os.getenv('DEEPSEEK_API_KEY') else '✗ Missing'}")
    print(f"Python Version: {sys.version}")
    print("=" * 70)
    
    # Run the tests
    unittest.main(verbosity=2)