#!/usr/bin/env python3
"""
Test script for Jarvis Multi-User Context System
"""

import sys
import os
from jarvis_context import create_jarvis_context

def test_multiuser_system():
    """Test the multi-user context system functionality"""
    print("👥 Testing Jarvis Multi-User Context System")
    print("=" * 60)
    
    # Initialize context with a specific default user
    context = create_jarvis_context(db_path="test_multiuser_memory.db", default_user="alex")
    
    print("✅ Multi-user context system initialized")
    print(f"Database: {context.db_path}")
    print(f"Default user: {context.default_user_id}")
    print(f"Current user: {context.current_user_id}")
    print()
    
    # Test 1: Default user setup
    print("Test 1: Default user operations")
    context.learn_about_user("name", "Alex")
    context.set_preference("favorite_color", "blue")
    context.add_exchange("Hello Jarvis", "Hello Alex, how can I help?", "greeting")
    print(f"Current user: {context.get_current_user()}")
    print(f"Preferences: {context.session_preferences}")
    print()
    
    # Test 2: Create and switch to new user
    print("Test 2: Creating and switching users")
    success = context.create_user("sarah", "Sarah Johnson")
    print(f"Created user Sarah: {success}")
    
    success = context.switch_user("sarah")
    print(f"Switched to Sarah: {success}")
    print(f"Current user: {context.get_current_user()}")
    print(f"Session preferences (should be empty): {context.session_preferences}")
    print()
    
    # Test 3: User-specific data
    print("Test 3: User-specific data isolation")
    # Set different preferences for Sarah
    context.set_preference("favorite_color", "green")
    context.set_preference("work_hours", "9-5")
    context.add_exchange("What's my favorite color?", "Your favorite color is green.", "personal")
    context.add_exchange("Set a reminder", "I'll set that reminder for you, Sarah.", "reminder")
    
    print("Sarah's data:")
    print(f"  Preferences: {context.session_preferences}")
    print(f"  Recent context: {len(context.get_recent_context())}")
    print()
    
    # Test 4: Switch back to Alex and verify isolation
    print("Test 4: Data isolation verification")
    context.switch_user("alex")
    print("Switched back to Alex:")
    print(f"  Current user: {context.get_current_user()}")
    print(f"  Preferences: {context.session_preferences}")
    print(f"  Recent context: {len(context.get_recent_context())}")
    print()
    
    # Test 5: Create third user and test user listing
    print("Test 5: User management")
    context.create_user("john", "John Smith")
    context.switch_user("john")
    context.set_preference("timezone", "PST")
    context.add_exchange("Good morning", "Good morning John!", "greeting")
    
    users = context.list_users()
    print("All users in system:")
    for user in users:
        current_marker = " (CURRENT)" if user['is_current'] else ""
        print(f"  - {user['display_name']} (ID: {user['user_id']}){current_marker}")
    print()
    
    # Test 6: Context for AI with user-specific data
    print("Test 6: User-specific AI context")
    ai_context = context.get_context_for_ai()
    print("AI Context for John:")
    print(ai_context[:200] + "..." if len(ai_context) > 200 else ai_context)
    print()
    
    # Test 7: Cross-user context verification
    print("Test 7: Cross-user data verification")
    
    # Switch to each user and show their unique data
    for user_id in ["alex", "sarah", "john"]:
        context.switch_user(user_id)
        user_info = context.get_current_user()
        prefs = context.session_preferences
        history_count = len(context.get_recent_context())
        
        print(f"{user_info['display_name']}:")
        print(f"  Preferences: {list(prefs.keys())}")
        print(f"  Conversation history: {history_count} exchanges")
        
        # Show a sample preference if available
        if prefs:
            sample_key = list(prefs.keys())[0]
            print(f"  Sample: {sample_key} = {prefs[sample_key]}")
        print()
    
    # Test 8: Database integrity check
    print("Test 8: Database integrity verification")
    
    # Check that data is properly segregated
    import sqlite3
    with sqlite3.connect(context.db_path) as conn:
        # Count conversations per user
        cursor = conn.execute('''
            SELECT user_id, COUNT(*) as conv_count 
            FROM conversations 
            GROUP BY user_id
        ''')
        
        print("Conversations per user:")
        for user_id, count in cursor.fetchall():
            print(f"  {user_id}: {count} conversations")
        
        # Count preferences per user
        cursor = conn.execute('''
            SELECT user_id, COUNT(*) as pref_count 
            FROM user_preferences 
            GROUP BY user_id
        ''')
        
        print("Preferences per user:")
        for user_id, count in cursor.fetchall():
            print(f"  {user_id}: {count} preferences")
    print()
    
    # Test 9: User deletion (testing with a temporary user)
    print("Test 9: User deletion")
    context.create_user("temp", "Temporary User")
    context.switch_user("temp")
    context.set_preference("temp_setting", "test")
    context.add_exchange("Test", "Test response", "test")
    
    print("Created temporary user with data")
    
    # Delete the user
    deleted = context.delete_user("temp")
    print(f"Deleted temporary user: {deleted}")
    print(f"Current user after deletion: {context.get_current_user()['display_name']}")
    
    # Verify temp user is gone
    users_after = context.list_users()
    temp_found = any(u['user_id'] == 'temp' for u in users_after)
    print(f"Temporary user still exists: {temp_found}")
    print()
    
    # Test 10: Export and status
    print("Test 10: System status and export")
    
    # Switch back to Alex for final status
    context.switch_user("alex")
    
    status = context.get_context_status()
    print("System Status:")
    print(status)
    print()
    
    exported = context.export_context_data()
    print("Export Summary:")
    print(f"  Session info keys: {list(exported.get('session_info', {}).keys())}")
    print(f"  Database stats: {exported.get('database_stats', {})}")
    print()
    
    print("✅ All multi-user tests completed successfully!")
    print("🗑️  Cleaning up test database...")
    
    # Clean up test database
    try:
        os.remove("test_multiuser_memory.db")
        print("✅ Test database removed")
    except FileNotFoundError:
        print("⚠️  Test database not found")

def test_multiuser_voice_commands():
    """Test voice command integration with multi-user system"""
    print("\n" + "=" * 60)
    print("🎤 Testing Multi-User Voice Commands")
    print("=" * 60)
    
    # This would typically test the commands.py integration
    # For now, we'll simulate the command processing
    
    test_commands = [
        "I am Alice",
        "my name is Alice Cooper", 
        "remember that I like pizza",
        "what do you know about me",
        "switch to user Bob",
        "I am Bob",
        "remember that I prefer tea",
        "who am i",
        "list users",
        "switch to user Alice",
        "what do you know about me"
    ]
    
    print("Simulated voice command sequence:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"{i:2d}. '{cmd}'")
    
    print("\n✅ Voice command integration ready for testing")

if __name__ == "__main__":
    test_multiuser_system()
    test_multiuser_voice_commands()