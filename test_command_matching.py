#!/usr/bin/env python3
"""
Test script to verify command matching works correctly
"""

import re

def test_command_matching():
    """Test the improved command matching logic with specific phrases"""
    
    # Commands that should be matched (updated to new specific commands)
    commands = ['hi jarvis', 'hello jarvis', 'what time is it', 'current date', 'jarvis status', 
                'help me', 'tell me a joke', 'run voice test', 'memory usage']
    
    # Test cases: (input_text, expected_matches)
    test_cases = [
        ("Hi Jarvis, how are you?", ['hi jarvis']),
        ("Hello Jarvis", ['hello jarvis']),
        ("What time is it?", ['what time is it']),
        ("This is a test", []),  # Should NOT match anything
        ("I'm going with PushOver to test notifications", []),  # Should NOT match 'test'
        ("Can you help me with this?", ['help me']),
        ("I want to test this feature", []),  # Should NOT match old 'test' command
        ("Run voice test please", ['run voice test']),  # Should match specific test command
        ("The memory usage is high", ['memory usage']),  # Should match specific memory command
        ("I need help", []),  # Should NOT match 'help me' exactly
        ("Tell me a joke", ['tell me a joke']),
        ("What's the current date?", ['current date']),  # Should match 'current date'
    ]
    
    print("Testing improved command matching...")
    print()
    
    for text, expected in test_cases:
        text_lower = text.lower()
        matches = []
        
        for command in commands:
            pattern = r'\b' + re.escape(command) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(command)
        
        status = "✅ PASS" if matches == expected else "❌ FAIL"
        print(f"{status} '{text}' -> {matches} (expected: {expected})")
        
        if matches != expected:
            print(f"   Expected: {expected}")
            print(f"   Got: {matches}")
    
    print()
    print("Test complete!")

if __name__ == "__main__":
    test_command_matching()