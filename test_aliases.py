#!/usr/bin/env python3
"""
Test script for Jarvis User Alias System
"""

import sys
import os
from jarvis_context import create_jarvis_context

def test_alias_system():
    """Test the user alias/multiple names functionality"""
    print("👤 Testing Jarvis User Alias System")
    print("=" * 50)
    
    # Initialize context
    context = create_jarvis_context(db_path="test_alias_memory.db", default_user="alex")
    
    print("✅ Context system initialized")
    print(f"Current user: {context.get_current_user()['display_name']}")
    print()
    
    # Test 1: Create user and add aliases
    print("Test 1: Adding aliases to user")
    
    # Set up initial user
    context.learn_about_user("name", "Alexander")
    print(f"Initial name: {context.user_name}")
    
    # Add multiple aliases
    aliases_to_add = ["Alex", "Al", "Xander", "Sandy"]
    for alias in aliases_to_add:
        success = context.add_user_alias(alias, "nickname")
        print(f"Added alias '{alias}': {success}")
    
    # Add a formal name
    context.add_user_alias("Dr. Alexander Smith", "formal")
    print("Added formal name: Dr. Alexander Smith")
    print()
    
    # Test 2: List all aliases
    print("Test 2: Listing user aliases")
    aliases = context.get_user_aliases()
    print(f"Total aliases: {len(aliases)}")
    for alias in aliases:
        primary_marker = " (PRIMARY)" if alias['is_primary'] else ""
        print(f"  - {alias['alias']} ({alias['type']}){primary_marker}")
    print()
    
    # Test 3: Find user by different aliases
    print("Test 3: Finding user by aliases")
    test_names = ["alexander", "Alex", "al", "XANDER", "Dr. Alexander Smith", "Sandy"]
    
    for name in test_names:
        found_user = context.find_user_by_alias(name)
        result = "✅ Found" if found_user == context.current_user_id else "❌ Not found"
        print(f"  '{name}' -> {result}")
    
    # Test a name that shouldn't exist
    not_found = context.find_user_by_alias("Bob")
    print(f"  'Bob' -> {'❌ Not found' if not_found is None else '⚠️ Unexpected result'}")
    print()
    
    # Test 4: Set primary alias
    print("Test 4: Setting primary alias")
    
    # Set Alex as primary
    success = context.set_primary_alias("Alex")
    print(f"Set 'Alex' as primary: {success}")
    print(f"Updated display name: {context.user_name}")
    
    # Check aliases again
    aliases = context.get_user_aliases()
    primary_aliases = [a for a in aliases if a['is_primary']]
    print(f"Primary aliases: {[a['alias'] for a in primary_aliases]}")
    print()
    
    # Test 5: User switching with aliases
    print("Test 5: User switching with aliases")
    
    # Create another user with aliases
    context.create_user("sarah", "Sarah Johnson")
    context.switch_user("sarah")
    context.add_user_alias("Sara", "nickname")  # Alternative spelling
    context.add_user_alias("Mom", "family")
    context.add_user_alias("Dr. Johnson", "professional")
    
    print("Created Sarah with aliases: Sara, Mom, Dr. Johnson")
    
    # Switch back using different aliases
    switch_tests = [
        ("Alex", "alex"),      # Should find Alex user by alias
        ("alexander", "alex"), # Should find Alex user by original name  
        ("Mom", "sarah"),      # Should find Sarah by alias
        ("Dr. Johnson", "sarah") # Should find Sarah by professional alias
    ]
    
    for alias, expected_user in switch_tests:
        # First find the user ID
        found_user = context.find_user_by_alias(alias)
        if found_user:
            context.switch_user(alias)  # Switch using alias
            current = context.get_current_user()
            result = "✅" if current['user_id'] == expected_user else "❌"
            print(f"  Switch to '{alias}' -> {result} (Now: {current['display_name']})")
    print()
    
    # Test 6: Alias removal
    print("Test 6: Removing aliases")
    
    # Switch to Alex and remove an alias
    context.switch_user("Alex")
    initial_count = len(context.get_user_aliases())
    print(f"Initial alias count: {initial_count}")
    
    # Remove "Sandy" alias
    removed = context.remove_user_alias("Sandy")
    print(f"Removed 'Sandy': {removed}")
    
    final_count = len(context.get_user_aliases())
    print(f"Final alias count: {final_count}")
    
    # Verify Sandy can't be found
    not_found = context.find_user_by_alias("Sandy")
    print(f"'Sandy' still findable: {not_found is not None}")
    print()
    
    # Test 7: Database integrity
    print("Test 7: Database integrity check")
    
    import sqlite3
    with sqlite3.connect(context.db_path) as conn:
        # Count aliases per user
        cursor = conn.execute('''
            SELECT u.display_name, COUNT(ua.alias) as alias_count
            FROM users u
            LEFT JOIN user_aliases ua ON u.user_id = ua.user_id
            GROUP BY u.user_id, u.display_name
        ''')
        
        print("Aliases per user:")
        for display_name, count in cursor.fetchall():
            print(f"  {display_name}: {count} aliases")
        
        # Check for primary aliases
        cursor = conn.execute('''
            SELECT u.display_name, ua.alias
            FROM users u
            JOIN user_aliases ua ON u.user_id = ua.user_id
            WHERE ua.is_primary = 1
        ''')
        
        print("Primary aliases:")
        for display_name, primary_alias in cursor.fetchall():
            print(f"  {display_name}: {primary_alias}")
    print()
    
    # Test 8: Edge cases
    print("Test 8: Edge case testing")
    
    # Try to add duplicate alias
    duplicate = context.add_user_alias("Alex")  # Should fail
    print(f"Adding duplicate 'Alex': {not duplicate} (should be False)")
    
    # Try to remove non-existent alias
    remove_missing = context.remove_user_alias("NonExistent")
    print(f"Removing non-existent alias: {not remove_missing} (should be False)")
    
    # Try to set primary for non-existent alias
    set_missing_primary = context.set_primary_alias("NonExistent")
    print(f"Setting non-existent as primary: {not set_missing_primary} (should be False)")
    print()
    
    # Test 9: Complex switching scenario
    print("Test 9: Complex switching scenario")
    
    # Create a family scenario
    users_and_aliases = [
        ("dad", "Robert", ["Bob", "Rob", "Bobby", "Dad", "Daddy", "Father"]),
        ("mom", "Elizabeth", ["Liz", "Beth", "Lizzy", "Mom", "Mommy", "Mother"]),
        ("son", "Michael", ["Mike", "Mikey", "Son", "Junior"])
    ]
    
    for user_id, primary_name, aliases in users_and_aliases:
        context.create_user(user_id, primary_name)
        context.switch_user(user_id)
        for alias in aliases:
            context.add_user_alias(alias)
        print(f"Created {primary_name} with {len(aliases)} aliases")
    
    # Test family member switching
    family_switches = ["Dad", "Mom", "Son", "Bobby", "Lizzy", "Junior"]
    print("Family switching test:")
    for name in family_switches:
        found_user = context.find_user_by_alias(name)
        if found_user:
            context.switch_user(name)
            current = context.get_current_user()['display_name']
            print(f"  '{name}' -> {current}")
    print()
    
    # Test 10: Export and cleanup
    print("Test 10: System summary")
    
    # List all users and their aliases
    all_users = context.list_users()
    print(f"Total users in system: {len(all_users)}")
    
    for user in all_users:
        context.switch_user(user['user_id'])
        aliases = context.get_user_aliases()
        alias_names = [a['alias'] for a in aliases]
        current_marker = " (CURRENT)" if user['is_current'] else ""
        print(f"  {user['display_name']}{current_marker}: {alias_names}")
    
    print("\n✅ All alias tests completed successfully!")
    print("🗑️  Cleaning up test database...")
    
    # Clean up test database
    try:
        os.remove("test_alias_memory.db")
        print("✅ Test database removed")
    except FileNotFoundError:
        print("⚠️  Test database not found")

def test_voice_commands_simulation():
    """Simulate voice commands for alias management"""
    print("\n" + "=" * 50)
    print("🎤 Testing Alias Voice Commands (Simulated)")
    print("=" * 50)
    
    voice_commands = [
        "I am Alexander",
        "call me also Alex",
        "call me also Al", 
        "add alias Xander",
        "my aliases",
        "primary name is Alex",
        "who am i",
        "switch to user Mom",
        "I am Elizabeth",
        "call me also Liz",
        "call me also Beth",
        "primary name is Mom",
        "switch to user Alex",
        "my aliases",
        "remove alias Al",
        "my aliases"
    ]
    
    print("Voice command sequence for alias testing:")
    for i, cmd in enumerate(voice_commands, 1):
        print(f"{i:2d}. '{cmd}'")
    
    print("\n✅ Voice command simulation complete")

if __name__ == "__main__":
    test_alias_system()
    test_voice_commands_simulation()