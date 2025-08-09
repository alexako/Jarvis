#!/usr/bin/env python3
"""
Test the specific PushOver conversation that was causing issues
"""

import re

def test_pushover_conversation():
    """Test the specific conversation patterns that were problematic"""
    
    # Updated command mappings (the specific ones we're using now)
    commands = [
        'hello jarvis', 'hi jarvis', 'hey jarvis', 'good morning', 'good afternoon', 'good evening',
        'what time is it', 'current time', 'what date is it', "what's the date", 'current date',
        'how are you', 'jarvis status', 'system status', 'battery status', 'check memory usage', 'disk space',
        'jarvis help me', 'what can you do', 'list commands', 'who are you', 'introduce yourself',
        'tell me a joke', 'make me laugh', 'stop listening', 'shutdown system', 
        'goodbye jarvis', 'bye jarvis', 'run voice test', 'ai status', 'clear history'
    ]
    
    # Test the actual problematic conversation
    problematic_cases = [
        # The original problematic messages
        ("Yes... I'm chatting with you to test the notification system I implemented with PushOver", []),
        ("I'm going with PushOver since I've used it for so long. Any input?", []),
        ("What?", []),
        ("We were just having a conversation about push notification services", []),
        
        # Other potential issues
        ("This is important information", []),
        ("I need help with this task", []),
        ("The memory usage seems fine", []),
        ("What time zone are you in", []),
        ("I want to test this feature", []),
        ("Can you help me understand", []),
        ("That's a good joke", []),
        ("The current status is unclear", []),
        ("Update the date field", []),
        
        # Commands that SHOULD still work
        ("Hi Jarvis", ['hi jarvis']),
        ("What time is it?", ['what time is it']),
        ("Tell me a joke", ['tell me a joke']),
        ("Jarvis help me please", ['jarvis help me']),
        ("Run voice test", ['run voice test']),
    ]
    
    print("Testing PushOver conversation patterns...")
    print()
    
    all_passed = True
    for text, expected in problematic_cases:
        text_lower = text.lower()
        matches = []
        
        for command in commands:
            pattern = r'\b' + re.escape(command) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(command)
        
        status = "✅ PASS" if matches == expected else "❌ FAIL"
        if matches != expected:
            all_passed = False
        
        print(f"{status} '{text}'")
        if matches != expected:
            print(f"      Expected: {expected}")
            print(f"      Got: {matches}")
        elif matches:
            print(f"      Matched: {matches}")
        else:
            print(f"      → Goes to AI brain (correct!)")
        print()
    
    if all_passed:
        print("🎉 All tests passed! PushOver conversations should work correctly now.")
    else:
        print("❌ Some tests failed. Need more adjustments.")

if __name__ == "__main__":
    test_pushover_conversation()