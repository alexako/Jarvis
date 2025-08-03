#!/usr/bin/env python3
"""
Test script to verify command matching works correctly
"""

import re

def test_command_matching():
    """Test the improved command matching logic"""
    
    # Commands that should be matched
    commands = ['hi', 'hello', 'time', 'date', 'status']
    
    # Test cases: (input_text, expected_matches)
    test_cases = [
        ("Hi there", ['hi']),
        ("Hello Jarvis", ['hello']),
        ("What time is it", ['time']),
        ("This is a test", []),  # Should NOT match 'hi' in 'this'
        ("I'm going with PushOver", []),  # Should NOT match 'hi' in 'with'
        ("Hi", ['hi']),
        ("time for lunch", ['time']),
        ("Something within it", []),  # Should NOT match 'hi' as substring
        ("Say hi to everyone", ['hi']),  # Should match 'hi' as word
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