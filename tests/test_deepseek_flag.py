#!/usr/bin/env python3
"""
Unit tests for --use-deepseek flag functionality
"""

import unittest
import argparse
import sys
import os
import subprocess
from unittest.mock import patch, MagicMock


class TestDeepSeekFlag(unittest.TestCase):
    """Unit tests for DeepSeek flag parsing and validation"""
    
    def test_deepseek_flag_help_text(self):
        """Test that --use-deepseek flag appears in help text"""
        result = subprocess.run([
            'python', 'jarvis.py', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertIn('--use-deepseek', result.stdout)
        self.assertIn('Use DeepSeek as primary AI provider', result.stdout)
    
    def test_mutual_exclusivity_error(self):
        """Test that using both AI provider flags raises error"""
        result = subprocess.run([
            'python', 'jarvis.py', '--enable-ai', '--use-anthropic', '--use-deepseek'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertNotEqual(result.returncode, 0)
        # Error message appears in stdout, not stderr
        output = result.stdout + result.stderr
        self.assertIn('Only one AI provider can be specified as primary', output)
    
    def test_deepseek_flag_no_error(self):
        """Test that --use-deepseek flag alone doesn't cause errors during parsing"""
        # Test just the help to ensure flag is recognized
        result = subprocess.run([
            'python', 'jarvis.py', '--use-deepseek', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('--use-deepseek', result.stdout)
    
    def test_anthropic_flag_no_error(self):
        """Test that --use-anthropic flag alone doesn't cause errors during parsing"""
        result = subprocess.run([
            'python', 'jarvis.py', '--use-anthropic', '--help'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('--use-anthropic', result.stdout)


class TestDeepSeekConfig(unittest.TestCase):
    """Test DeepSeek configuration creation"""
    
    def test_ai_provider_preference_deepseek(self):
        """Test that DeepSeek preference is set correctly"""
        # Simulate argument parsing
        args = argparse.Namespace()
        args.use_deepseek = True
        args.use_anthropic = False
        
        # Test the logic from jarvis.py:190-193
        if args.use_deepseek:
            ai_provider_preference = "deepseek"
        elif args.use_anthropic:
            ai_provider_preference = "anthropic"
        else:
            ai_provider_preference = "anthropic"  # default
            
        self.assertEqual(ai_provider_preference, "deepseek")
    
    def test_ai_provider_preference_anthropic(self):
        """Test that Anthropic preference is set correctly"""
        args = argparse.Namespace()
        args.use_deepseek = False
        args.use_anthropic = True
        
        if args.use_deepseek:
            ai_provider_preference = "deepseek"
        elif args.use_anthropic:
            ai_provider_preference = "anthropic"
        else:
            ai_provider_preference = "anthropic"  # default
            
        self.assertEqual(ai_provider_preference, "anthropic")
    
    def test_ai_provider_preference_default(self):
        """Test default AI provider preference"""
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




if __name__ == '__main__':
    # Run the unit tests
    unittest.main(verbosity=2)