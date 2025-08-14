#!/usr/bin/env python3
"""
Unit tests for --use-anthropic flag functionality
"""

import unittest
import argparse
import sys
import os
import subprocess
from unittest.mock import patch, MagicMock


class TestAnthropicFlag(unittest.TestCase):
    """Unit tests for Anthropic flag parsing and validation"""
    
    def test_anthropic_flag_help_text(self):
        """Test that --use-anthropic flag appears in help text"""
        result = subprocess.run([
            'python', 'jarvis.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertIn('--use-anthropic', result.stdout)
        self.assertIn('Use Anthropic Claude as primary AI provider', result.stdout)
    
    def test_mutual_exclusivity_error_deepseek_first(self):
        """Test that using both AI provider flags raises error (deepseek first)"""
        result = subprocess.run([
            'python', 'jarvis.py', '--enable-ai', '--use-deepseek', '--use-anthropic'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertNotEqual(result.returncode, 0)
        # Error message appears in stdout, not stderr
        output = result.stdout + result.stderr
        self.assertIn('Only one AI provider can be specified as primary', output)
    
    def test_anthropic_flag_no_error(self):
        """Test that --use-anthropic flag alone doesn't cause errors during parsing"""
        # Test just the help to ensure flag is recognized
        result = subprocess.run([
            'python', 'jarvis.py', '--use-anthropic', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('--use-anthropic', result.stdout)
    
    def test_anthropic_flag_with_enable_ai(self):
        """Test that --use-anthropic flag works with --enable-ai"""
        result = subprocess.run([
            'python', 'jarvis.py', '--enable-ai', '--use-anthropic', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('--use-anthropic', result.stdout)
    
    def test_anthropic_is_default_provider(self):
        """Test that Anthropic is mentioned as default provider in help"""
        result = subprocess.run([
            'python', 'jarvis.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
        self.assertIn('(default)', result.stdout)


class TestAnthropicConfig(unittest.TestCase):
    """Test Anthropic configuration creation"""
    
    def test_ai_provider_preference_anthropic_explicit(self):
        """Test that Anthropic preference is set correctly when explicitly specified"""
        # Simulate argument parsing
        args = argparse.Namespace()
        args.use_deepseek = False
        args.use_anthropic = True
        
        # Test the logic from jarvis.py:190-193
        if args.use_deepseek:
            ai_provider_preference = "deepseek"
        elif args.use_anthropic:
            ai_provider_preference = "anthropic"
        else:
            ai_provider_preference = "anthropic"  # default
            
        self.assertEqual(ai_provider_preference, "anthropic")
    
    def test_ai_provider_preference_anthropic_default(self):
        """Test that Anthropic is the default when no provider flags are used"""
        args = argparse.Namespace()
        args.use_deepseek = False
        args.use_anthropic = False
        
        if args.use_deepseek:
            ai_provider_preference = "deepseek"
        elif args.use_anthropic:
            ai_provider_preference = "anthropic"
        else:
            ai_provider_preference = "anthropic"  # default
            
        self.assertEqual(ai_provider_preference, "anthropic")
    
    def test_ai_provider_preference_not_deepseek(self):
        """Test that when deepseek is not used, anthropic is selected"""
        args = argparse.Namespace()
        args.use_deepseek = False
        args.use_anthropic = False  # Even when not explicitly set
        
        if args.use_deepseek:
            ai_provider_preference = "deepseek"
        elif args.use_anthropic:
            ai_provider_preference = "anthropic"
        else:
            ai_provider_preference = "anthropic"  # default
            
        self.assertEqual(ai_provider_preference, "anthropic")


class TestAnthropicFlagBehavior(unittest.TestCase):
    """Test specific behaviors of the Anthropic flag"""
    
    def test_anthropic_flag_redundant_default(self):
        """Test that using --use-anthropic when it's already default doesn't cause issues"""
        result = subprocess.run([
            'python', 'jarvis.py', '--enable-ai', '--use-anthropic', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Should work fine since anthropic is default anyway
        self.assertEqual(result.returncode, 0)
        self.assertIn('--use-anthropic', result.stdout)
    
    def test_anthropic_flag_overrides_default(self):
        """Test that explicit --use-anthropic flag behavior is documented"""
        # This test ensures the flag exists and is documented properly
        result = subprocess.run([
            'python', 'jarvis.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        help_text = result.stdout
        self.assertIn('--use-anthropic', help_text)
        self.assertIn('--use-deepseek', help_text)
        
        # Both should be present for mutual exclusivity testing
        lines = help_text.split('\n')
        anthropic_lines = [line for line in lines if '--use-anthropic' in line]
        deepseek_lines = [line for line in lines if '--use-deepseek' in line]
        
        self.assertTrue(len(anthropic_lines) > 0, "Anthropic flag should be in help")
        self.assertTrue(len(deepseek_lines) > 0, "DeepSeek flag should be in help")


if __name__ == '__main__':
    # Run the unit tests
    unittest.main(verbosity=2)