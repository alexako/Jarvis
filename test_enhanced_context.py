#!/usr/bin/env python3
"""
Test and demonstration script for Enhanced Jarvis Context System
Shows how to use the robust contextual database for personal information
"""

import sys
import os
from jarvis_context_enhanced import (
    create_enhanced_context, 
    MemoryCategory, 
    MemoryImportance
)
from datetime import datetime, date

def test_enhanced_context():
    """Test and demonstrate the enhanced context system"""
    print("🧠 Testing Enhanced Jarvis Context System")
    print("=" * 60)
    
    # Initialize enhanced context
    context = create_enhanced_context(
        db_path="test_enhanced_memory.db",
        encryption_key="my_secret_key_123"  # For sensitive data
    )
    
    print("✅ Enhanced context system initialized")
    print()
    
    # Test 1: Personal Information
    print("Test 1: Storing Personal Information")
    print("-" * 40)
    
    # Store various personal details
    context.remember(MemoryCategory.PERSONAL, "full_name", "Alex Reyes", MemoryImportance.HIGH)
    context.remember(MemoryCategory.PERSONAL, "birthday", "1990-01-15", MemoryImportance.HIGH)
    context.remember(MemoryCategory.PERSONAL, "age", 34, MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.PERSONAL, "hometown", "San Francisco", MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.PERSONAL, "blood_type", "O+", MemoryImportance.CRITICAL)
    context.remember(MemoryCategory.PERSONAL, "allergies", ["peanuts", "shellfish"], MemoryImportance.CRITICAL)
    
    # Recall personal info
    print(f"Name: {context.recall(MemoryCategory.PERSONAL, 'full_name')}")
    print(f"Birthday: {context.recall(MemoryCategory.PERSONAL, 'birthday')}")
    print(f"Allergies: {context.recall(MemoryCategory.PERSONAL, 'allergies')}")
    print()
    
    # Test 2: Preferences
    print("Test 2: Storing Preferences")
    print("-" * 40)
    
    context.remember(MemoryCategory.PREFERENCES, "favorite_color", "blue", MemoryImportance.LOW)
    context.remember(MemoryCategory.PREFERENCES, "coffee_order", "Large cappuccino with oat milk", MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.PREFERENCES, "music_genres", ["jazz", "electronic", "classical"], MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.PREFERENCES, "preferred_temperature", 72, MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.PREFERENCES, "wake_up_time", "7:00 AM", MemoryImportance.HIGH)
    
    print(f"Coffee order: {context.recall(MemoryCategory.PREFERENCES, 'coffee_order')}")
    print(f"Music preferences: {context.recall(MemoryCategory.PREFERENCES, 'music_genres')}")
    print()
    
    # Test 3: Work Information
    print("Test 3: Work and Professional Info")
    print("-" * 40)
    
    context.remember(MemoryCategory.WORK, "job_title", "Senior Software Engineer", MemoryImportance.HIGH)
    context.remember(MemoryCategory.WORK, "company", "Tech Innovations Inc", MemoryImportance.HIGH)
    context.remember(MemoryCategory.WORK, "skills", ["Python", "JavaScript", "Machine Learning", "Cloud Architecture"], MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.WORK, "current_project", "AI Assistant Platform", MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.WORK, "work_hours", "9 AM - 6 PM", MemoryImportance.MEDIUM)
    
    print(f"Job: {context.recall(MemoryCategory.WORK, 'job_title')} at {context.recall(MemoryCategory.WORK, 'company')}")
    print(f"Skills: {context.recall(MemoryCategory.WORK, 'skills')}")
    print()
    
    # Test 4: Relationships
    print("Test 4: Remember People and Relationships")
    print("-" * 40)
    
    context.remember_person(
        "Sarah Johnson",
        relationship="wife",
        birthday="1992-03-22",
        phone="555-1234",
        email="sarah@email.com",
        notes="Met in college, loves hiking and photography"
    )
    
    context.remember_person(
        "Bob Smith",
        relationship="best friend",
        phone="555-5678",
        notes="Known since high school, plays guitar"
    )
    
    context.remember_person(
        "Dr. Emily Chen",
        relationship="doctor",
        phone="555-9999",
        notes="Primary care physician, very thorough"
    )
    
    # Recall person info
    sarah_info = context.recall_person("Sarah Johnson")
    if sarah_info:
        print(f"Sarah: {sarah_info['relationship']}, Birthday: {sarah_info['birthday']}")
    
    bob_info = context.recall_person("Bob Smith")
    if bob_info:
        print(f"Bob: {bob_info['relationship']}, Notes: {bob_info['notes']}")
    print()
    
    # Test 5: Daily Routines
    print("Test 5: Daily Routines and Schedules")
    print("-" * 40)
    
    context.add_routine(
        "Morning Workout",
        routine_type="exercise",
        days=["Monday", "Wednesday", "Friday"],
        time="7:30 AM",
        duration=45,
        description="Gym workout - cardio and weights",
        reminder=True
    )
    
    context.add_routine(
        "Team Standup",
        routine_type="work",
        days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        time="10:00 AM",
        duration=15,
        description="Daily team sync meeting",
        reminder=True
    )
    
    context.add_routine(
        "Meditation",
        routine_type="wellness",
        days=["Every day"],
        time="9:00 PM",
        duration=20,
        description="Evening meditation session",
        reminder=False
    )
    
    routines = context.get_active_routines()
    for routine in routines:
        print(f"• {routine['name']}: {routine['time']} on {', '.join(routine['days'][:3]) if routine['days'] else 'N/A'}")
    print()
    
    # Test 6: Health Information (with encryption for sensitive data)
    print("Test 6: Health Information (Encrypted)")
    print("-" * 40)
    
    context.remember(
        MemoryCategory.HEALTH, 
        "medications", 
        ["Vitamin D", "Omega-3"],
        MemoryImportance.HIGH,
        encrypt=False  # Not sensitive
    )
    
    context.remember(
        MemoryCategory.HEALTH,
        "medical_conditions",
        "Mild asthma",
        MemoryImportance.CRITICAL,
        encrypt=True  # Sensitive - will be encrypted
    )
    
    context.remember(
        MemoryCategory.HEALTH,
        "exercise_goal",
        "Run 5K in under 25 minutes",
        MemoryImportance.MEDIUM
    )
    
    print(f"Medications: {context.recall(MemoryCategory.HEALTH, 'medications')}")
    print(f"Exercise goal: {context.recall(MemoryCategory.HEALTH, 'exercise_goal')}")
    print()
    
    # Test 7: Technical Preferences
    print("Test 7: Technical Preferences")
    print("-" * 40)
    
    context.remember(MemoryCategory.TECHNICAL, "favorite_ide", "VS Code", MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.TECHNICAL, "programming_languages", ["Python", "TypeScript", "Go"], MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.TECHNICAL, "os_preference", "macOS", MemoryImportance.MEDIUM)
    context.remember(MemoryCategory.TECHNICAL, "github_username", "alexreyes", MemoryImportance.MEDIUM)
    
    print(f"IDE: {context.recall(MemoryCategory.TECHNICAL, 'favorite_ide')}")
    print(f"Languages: {context.recall(MemoryCategory.TECHNICAL, 'programming_languages')}")
    print()
    
    # Test 8: Interests and Hobbies
    print("Test 8: Interests and Hobbies")
    print("-" * 40)
    
    context.remember(
        MemoryCategory.INTERESTS,
        "hobbies",
        ["photography", "hiking", "cooking", "reading sci-fi"],
        MemoryImportance.MEDIUM,
        tags=["leisure", "personal"]
    )
    
    context.remember(
        MemoryCategory.INTERESTS,
        "favorite_books",
        ["Dune", "The Expanse series", "Foundation"],
        MemoryImportance.LOW,
        tags=["books", "sci-fi"]
    )
    
    context.remember(
        MemoryCategory.INTERESTS,
        "travel_bucket_list",
        ["Japan", "Iceland", "New Zealand", "Peru"],
        MemoryImportance.MEDIUM,
        tags=["travel", "goals"]
    )
    
    print(f"Hobbies: {context.recall(MemoryCategory.INTERESTS, 'hobbies')}")
    print(f"Travel goals: {context.recall(MemoryCategory.INTERESTS, 'travel_bucket_list')}")
    print()
    
    # Test 9: Contextual Search
    print("Test 9: Contextual Memory Search")
    print("-" * 40)
    
    # Search for memories related to health
    health_memories = context.get_contextual_memories("health exercise workout", limit=5)
    print("Health-related memories:")
    for memory in health_memories:
        print(f"  • [{memory['category']}] {memory['key']}: {memory['value']}")
    print()
    
    # Test 10: Memory with Tags
    print("Test 10: Tagged Memories")
    print("-" * 40)
    
    # Recall memories by tags
    tagged_memories = context.recall(tags=["books"])
    print("Memories tagged with 'books':")
    for memory in tagged_memories:
        print(f"  • {memory['key']}: {memory['value']}")
    print()
    
    # Test 11: Important Memories
    print("Test 11: Critical and Important Memories")
    print("-" * 40)
    
    # Get only high-importance memories
    important_memories = context.recall(min_importance=4)
    print(f"Found {len(important_memories)} important memories:")
    for memory in important_memories[:5]:
        print(f"  • [{memory['category']}] {memory['key']} (importance: {memory['importance']})")
    print()
    
    # Test 12: Memory Summary
    print("Test 12: Memory System Summary")
    print("-" * 40)
    
    summary = context.get_memory_summary()
    print(f"Total memories stored: {summary['total_memories']}")
    print(f"Total people remembered: {summary['total_people']}")
    print(f"Active routines: {summary['active_routines']}")
    print("\nMemories by category:")
    for category, stats in summary['categories'].items():
        print(f"  • {category}: {stats['count']} items (avg importance: {stats['avg_importance']})")
    
    if summary['important_memories']:
        print("\nMost important memories:")
        for mem in summary['important_memories']:
            print(f"  • [{mem['category']}] {mem['key']} (importance: {mem['importance']})")
    print()
    
    # Test 13: Export memories
    print("Test 13: Export Memories for Backup")
    print("-" * 40)
    
    export_data = context.export_memories(include_encrypted=False)
    print(f"Exported {export_data['total_items']} items")
    print(f"  • Memories: {len(export_data['memories'])}")
    print(f"  • Relationships: {len(export_data['relationships'])}")
    print(f"  • Routines: {len(export_data['routines'])}")
    print()
    
    # Test 14: Forget a memory
    print("Test 14: Forgetting Memories")
    print("-" * 40)
    
    # Remember something temporary
    context.remember(MemoryCategory.REMINDERS, "temp_reminder", "Call dentist tomorrow", MemoryImportance.LOW)
    print(f"Temporary reminder: {context.recall(MemoryCategory.REMINDERS, 'temp_reminder')}")
    
    # Now forget it
    if context.forget(MemoryCategory.REMINDERS, "temp_reminder"):
        print("✓ Temporary reminder forgotten")
    
    # Try to recall it again
    forgotten = context.recall(MemoryCategory.REMINDERS, "temp_reminder")
    print(f"After forgetting: {forgotten}")
    print()
    
    print("=" * 60)
    print("✅ All enhanced context tests completed successfully!")
    print("\n🗑️  Cleaning up test database...")
    
    # Clean up test database
    try:
        os.remove("test_enhanced_memory.db")
        print("✅ Test database removed")
    except FileNotFoundError:
        print("⚠️  Test database not found")

if __name__ == "__main__":
    test_enhanced_context()
