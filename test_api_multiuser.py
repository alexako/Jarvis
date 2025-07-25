#!/usr/bin/env python3
"""
Test script for Jarvis API Multi-User Support
"""

import requests
import json
import time

def test_api_multiuser():
    """Test the API with multi-user functionality"""
    base_url = "http://127.0.0.1:8000"
    
    print("🌐 Testing Jarvis API Multi-User Support")
    print("=" * 50)
    
    # Test 1: Basic API health
    print("Test 1: API Health Check")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"Overall Health: {health['healthy']}")
            print(f"Components: {health['components']}")
        print()
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Please start with: python jarvis_api.py")
        return
    
    # Test 2: Check initial users
    print("Test 2: Initial Users")
    response = requests.get(f"{base_url}/users")
    if response.status_code == 200:
        users_data = response.json()
        print(f"Current user: {users_data['current_user']['display_name']}")
        print(f"Total users: {users_data['total_users']}")
    print()
    
    # Test 3: User identification via chat
    print("Test 3: User Identification")
    
    # Create first user through chat
    chat_data = {
        "text": "I am Alice",
        "user": "alice"
    }
    
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Current user: {result['current_user']}")
    print()
    
    # Test 4: Add alias for Alice
    print("Test 4: Adding alias")
    chat_data = {
        "text": "call me also Al",
        "user": "alice"
    }
    
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
    print()
    
    # Test 5: Create second user
    print("Test 5: Second User")
    chat_data = {
        "text": "I am Bob",
        "user": "bob"
    }
    
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Current user: {result['current_user']}")
    print()
    
    # Test 6: User-specific memory
    print("Test 6: User-specific memory")
    
    # Alice remembers something
    chat_data = {
        "text": "remember that I like pizza",
        "user": "alice"
    }
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Alice: {result['response']}")
    
    # Bob remembers something different
    chat_data = {
        "text": "remember that I prefer tea",
        "user": "bob"
    }
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Bob: {result['response']}")
    print()
    
    # Test 7: Check what each user knows
    print("Test 7: User-specific knowledge")
    
    # Ask Alice what she knows
    chat_data = {
        "text": "what do you know about me",
        "user": "alice"
    }
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"About Alice: {result['response']}")
    
    # Ask Bob what he knows  
    chat_data = {
        "text": "what do you know about me",
        "user": "bob"
    }
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"About Bob: {result['response']}")
    print()
    
    # Test 8: Switch users via API
    print("Test 8: API User Switching")
    
    # Switch to Alice via alias
    response = requests.post(f"{base_url}/users/switch?user_identifier=Al")
    if response.status_code == 200:
        result = response.json()
        print(f"Switch result: {result['message']}")
        print(f"Current user: {result['current_user']['display_name']}")
    print()
    
    # Test 9: Get user details
    print("Test 9: Current User Details")
    response = requests.get(f"{base_url}/users/current")
    if response.status_code == 200:
        user_info = response.json()
        print(f"User: {user_info['user']['display_name']}")
        print(f"Aliases: {[alias['alias'] for alias in user_info['aliases']]}")
        print(f"Recent exchanges: {user_info['recent_exchanges']}")
        print(f"Preferences: {user_info['preferences']}")
    print()
    
    # Test 10: List all users
    print("Test 10: All Users")
    response = requests.get(f"{base_url}/users")
    if response.status_code == 200:
        users_data = response.json()
        print(f"Total users: {users_data['total_users']}")
        for user in users_data['users']:
            current_marker = " (CURRENT)" if user['is_current'] else ""
            print(f"  - {user['display_name']} (ID: {user['user_id']}){current_marker}")
    print()
    
    # Test 11: AI with context
    print("Test 11: AI with User Context")
    chat_data = {
        "text": "What's my favorite food?",
        "user": "alice"
    }
    response = requests.post(f"{base_url}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"AI Response for Alice: {result['response']}")
        print(f"Provider used: {result['provider_used']}")
    print()
    
    print("✅ API Multi-User testing completed!")

def test_api_concurrent_users():
    """Test concurrent users accessing the API"""
    print("\n" + "=" * 50)
    print("🔄 Testing Concurrent User Access")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    # Simulate concurrent requests from different users
    import threading
    import queue
    
    results = queue.Queue()
    
    def user_session(user_name, commands):
        """Simulate a user session"""
        session_results = []
        for cmd in commands:
            try:
                chat_data = {
                    "text": cmd,
                    "user": user_name.lower()
                }
                response = requests.post(f"{base_url}/chat", json=chat_data)
                if response.status_code == 200:
                    result = response.json()
                    session_results.append({
                        "user": user_name,
                        "command": cmd,
                        "response": result['response'][:50] + "..." if len(result['response']) > 50 else result['response'],
                        "current_user": result['current_user']
                    })
                time.sleep(0.1)  # Small delay between requests
            except Exception as e:
                session_results.append({
                    "user": user_name,
                    "command": cmd,
                    "error": str(e)
                })
        
        results.put(session_results)
    
    # Define user sessions
    user_sessions = [
        ("Charlie", ["I am Charlie", "remember that I work at Google", "what time is it"]),
        ("Diana", ["I am Diana", "call me also Dee", "remember that I like cats"]),
        ("Eve", ["I am Eve", "my name is Evelyn", "what do you know about me"])
    ]
    
    # Start concurrent sessions
    threads = []
    for user_name, commands in user_sessions:
        thread = threading.Thread(target=user_session, args=(user_name, commands))
        threads.append(thread)
        thread.start()
    
    # Wait for all sessions to complete
    for thread in threads:
        thread.join()
    
    # Collect and display results
    print("Concurrent session results:")
    while not results.empty():
        user_results = results.get()
        for result in user_results:
            if 'error' in result:
                print(f"❌ {result['user']}: {result['command']} -> ERROR: {result['error']}")
            else:
                print(f"✅ {result['user']}: {result['command']} -> {result['response']} (as {result['current_user']})")
    
    print("\n✅ Concurrent user testing completed!")

if __name__ == "__main__":
    print("Starting API multi-user tests...")
    print("Make sure the API server is running: python jarvis_api.py")
    print()
    
    test_api_multiuser()
    test_api_concurrent_users()