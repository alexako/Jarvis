#!/usr/bin/env python3
"""
Test script for Pushover notifications
"""

import asyncio
import os
from pushover_notifications import notify_new_user, notify_system_event

async def test_notifications():
    """Test the Pushover notification system"""
    
    # Set test environment variables
    os.environ['PUSHOVER_ENABLED'] = 'true'
    os.environ['PUSHOVER_USER_KEY'] = 'your_pushover_user_key_here'
    os.environ['PUSHOVER_API_TOKEN'] = 'your_pushover_api_token_here'
    
    print("Testing Pushover notifications...")
    print(f"Enabled: {os.getenv('PUSHOVER_ENABLED')}")
    print(f"User Key: {os.getenv('PUSHOVER_USER_KEY')}")
    print(f"API Token: {os.getenv('PUSHOVER_API_TOKEN')}")
    print()
    
    # Test user notification
    user_info = {
        'name': 'Test User',
        'id': 'test_user_123'
    }
    
    message = "Hello Jarvis, can you help me with something?"
    ip_address = "192.168.1.100"
    
    print("Testing new user notification...")
    result = await notify_new_user(user_info, message, ip_address)
    print(f"New user notification result: {result}")
    print()
    
    # Test system event notification
    print("Testing system event notification...")
    result = await notify_system_event("TEST", "This is a test system event notification")
    print(f"System event notification result: {result}")
    print()
    
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_notifications())