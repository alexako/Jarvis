#!/usr/bin/env python3
"""
Test script for Jarvis Context/Memory System
"""

import sys
import os
from jarvis_context import create_jarvis_context

def test_context_system():
    """Test the context system functionality"""
    print("🧠 Testing Jarvis Context System")
    print("=" * 50)
    
    # Initialize context
    context = create_jarvis_context(db_path="test_jarvis_memory.db")
    
    print("✅ Context system initialized")
    print(f"Database: {context.db_path}")
    print()
    
    # Test 1: Learn user name
    print("Test 1: Learning user name")
    context.learn_about_user("name", "Alex", confidence=1.0, source="test")
    print(f"User name: {context.user_name}")
    print()
    
    # Test 2: Set preferences
    print("Test 2: Setting preferences")
    context.set_preference("favorite_color", "blue")
    context.set_preference("wake_time", "7:00 AM")
    context.set_preference("notifications", True, "bool")
    print(f"Preferences: {context.session_preferences}")
    print()
    
    # Test 3: Add conversation exchanges
    print("Test 3: Adding conversation exchanges")
    exchanges = [
        ("Hello Jarvis", "Good morning, Alex. How may I assist you today?", "greeting"),
        ("What's the weather like?", "I'll check the weather for you.", "weather"),
        ("Remember that I have a meeting at 3 PM", "I'll remember that you have a meeting at 3 PM.", "personal"),
        ("What did I just tell you to remember?", "You told me to remember that you have a meeting at 3 PM.", "personal")
    ]
    
    for user_input, jarvis_response, topic in exchanges:
        context.add_exchange(user_input, jarvis_response, topic)
        print(f"Added: {user_input} -> {jarvis_response} (topic: {topic})")
    print()
    
    # Test 4: Get recent context
    print("Test 4: Recent context")
    recent = context.get_recent_context(3)
    for i, exchange in enumerate(recent, 1):
        print(f"{i}. {exchange['user_input']} -> {exchange['jarvis_response']}")
    print()
    
    # Test 5: Context for AI
    print("Test 5: Context for AI processing")
    ai_context = context.get_context_for_ai()
    print("AI Context:")
    print(ai_context)
    print()
    
    # Test 6: Pronoun handling
    print("Test 6: Pronoun and reference handling")
    test_inputs = [
        "Tell me more about it",
        "Can you explain that?",
        "What time is my meeting?"
    ]
    
    for test_input in test_inputs:
        enhanced = context.handle_pronouns_and_references(test_input)
        print(f"'{test_input}' -> '{enhanced}'")
    print()
    
    # Test 7: Context status
    print("Test 7: Context status")
    status = context.get_context_status()
    print(status)
    print()
    
    # Test 8: Conversation summary
    print("Test 8: Conversation summary")
    summary = context.get_conversation_summary(days=1)
    print(f"Summary: {summary}")
    print()
    
    # Test 9: Export data
    print("Test 9: Export context data")
    exported = context.export_context_data()
    print("Exported data keys:", list(exported.keys()))
    if 'session_info' in exported:
        print("Session info:", exported['session_info'])
    print()
    
    print("✅ All tests completed successfully!")
    print("🗑️  Cleaning up test database...")
    
    # Clean up test database
    try:
        os.remove("test_jarvis_memory.db")
        print("✅ Test database removed")
    except FileNotFoundError:
        print("⚠️  Test database not found")

if __name__ == "__main__":
    test_context_system()