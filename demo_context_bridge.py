#!/usr/bin/env python3
"""
Demonstration of Jarvis Enhanced Context System
Shows how the context bridge integrates with natural language processing
"""

import os
from jarvis_context_bridge import create_context_bridge, MemoryCategory, MemoryImportance

def demo_context_bridge():
    """Demonstrate the context bridge with natural language processing"""
    print("🤖 Jarvis Enhanced Memory System Demo")
    print("=" * 60)
    
    # Initialize the context bridge
    context = create_context_bridge(
        db_path="demo_base_memory.db",
        enhanced_db_path="demo_enhanced_memory.db",
        encryption_key="secure_key_123"
    )
    
    print("✅ Context bridge initialized")
    print()
    
    # Simulate natural language inputs
    natural_inputs = [
        "Remember that I'm allergic to peanuts",
        "My birthday is January 15th",
        "I prefer coffee with oat milk and no sugar",
        "My wife's name is Sarah Johnson",
        "I work at Tech Innovations Inc",
        "My favorite color is blue",
        "Remember that I have a meeting with Bob tomorrow at 3 PM",
        "I love hiking and photography",
        "Don't forget to call the dentist",
    ]
    
    print("📝 Processing Natural Language Memory Inputs:")
    print("-" * 40)
    
    for input_text in natural_inputs:
        print(f"Input: \"{input_text}\"")
        result = context.process_natural_memory(input_text)
        if result:
            print(f"  ✓ Stored: {result}")
        else:
            # Try to store it manually if pattern not recognized
            context.remember_enhanced(
                MemoryCategory.REMINDERS,
                f"note_{len(context.session_history)}",
                input_text,
                MemoryImportance.MEDIUM,
                context="User input"
            )
            print(f"  ✓ Stored as general note")
        print()
    
    # Now test retrieval with natural queries
    queries = [
        "What am I allergic to?",
        "When is my birthday?",
        "What's my wife's name?",
        "What do I like?",
        "Where do I work?",
        "What are my hobbies?",
    ]
    
    print("🔍 Testing Natural Language Queries:")
    print("-" * 40)
    
    for query in queries:
        print(f"Query: \"{query}\"")
        response = context.get_contextual_response(query)
        if response:
            print(f"Response: {response}")
        else:
            print("Response: No relevant information found.")
        print()
    
    # Store some structured information
    print("📊 Storing Structured Information:")
    print("-" * 40)
    
    # Personal details
    context.remember_enhanced(
        MemoryCategory.PERSONAL, "full_name", "Alex Reyes",
        MemoryImportance.HIGH
    )
    context.remember_enhanced(
        MemoryCategory.PERSONAL, "age", 34,
        MemoryImportance.MEDIUM
    )
    context.remember_enhanced(
        MemoryCategory.PERSONAL, "blood_type", "O+",
        MemoryImportance.CRITICAL
    )
    print("✓ Stored personal information")
    
    # Health information
    context.remember_enhanced(
        MemoryCategory.HEALTH, "medications", ["Vitamin D", "Omega-3"],
        MemoryImportance.HIGH
    )
    context.remember_enhanced(
        MemoryCategory.HEALTH, "exercise_routine", "Running 3x week, Gym 2x week",
        MemoryImportance.MEDIUM
    )
    print("✓ Stored health information")
    
    # Work preferences
    context.remember_enhanced(
        MemoryCategory.WORK, "job_title", "Senior Software Engineer",
        MemoryImportance.HIGH
    )
    context.remember_enhanced(
        MemoryCategory.WORK, "skills", ["Python", "Machine Learning", "Cloud Architecture"],
        MemoryImportance.MEDIUM
    )
    print("✓ Stored work information")
    
    # Add some people
    context.enhanced_context.remember_person(
        "Bob Smith",
        relationship="colleague",
        email="bob@techcorp.com",
        notes="Works on the AI team, expert in NLP"
    )
    context.enhanced_context.remember_person(
        "Dr. Emily Chen",
        relationship="doctor",
        phone="555-9999",
        notes="Primary care physician at City Medical Center"
    )
    print("✓ Stored relationship information")
    
    # Add routines
    context.enhanced_context.add_routine(
        "Morning Run",
        routine_type="exercise",
        days=["Monday", "Wednesday", "Friday"],
        time="6:30 AM",
        duration=45,
        description="5K run around the park"
    )
    context.enhanced_context.add_routine(
        "Team Meeting",
        routine_type="work",
        days=["Tuesday", "Thursday"],
        time="10:00 AM",
        duration=60,
        description="Weekly team sync and planning"
    )
    print("✓ Stored routine information")
    print()
    
    # Test contextual memory search
    print("🔎 Testing Contextual Memory Search:")
    print("-" * 40)
    
    search_terms = ["health", "work", "family"]
    
    for term in search_terms:
        print(f"Searching for: '{term}'")
        memories = context.enhanced_context.get_contextual_memories(term, limit=3)
        if memories:
            for memory in memories:
                print(f"  • [{memory['category']}] {memory['key']}: {memory['value']}")
        else:
            print("  No memories found")
        print()
    
    # Show AI context generation
    print("🤖 AI Context Generation:")
    print("-" * 40)
    
    ai_context = context.get_enhanced_context_for_ai()
    print("Generated context for AI:")
    print(ai_context)
    print()
    
    # Show memory statistics
    print("📈 Memory System Statistics:")
    print("-" * 40)
    
    stats = context.get_memory_stats()
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"Total people: {stats.get('total_people', 0)}")
    print(f"Active routines: {stats.get('active_routines', 0)}")
    print(f"Session history: {stats.get('session_history', 0)} exchanges")
    
    if 'categories' in stats:
        print("\nMemories by category:")
        for category, info in stats['categories'].items():
            print(f"  • {category}: {info['count']} items (avg importance: {info['avg_importance']})")
    
    if 'important_memories' in stats:
        print("\nMost important memories:")
        for memory in stats['important_memories'][:3]:
            print(f"  • [{memory['category']}] {memory['key']} (importance: {memory['importance']})")
    print()
    
    # Export functionality
    print("💾 Memory Export:")
    print("-" * 40)
    
    export_data = context.enhanced_context.export_memories()
    print(f"Exported {export_data.get('total_items', 0)} items:")
    print(f"  • Memories: {len(export_data.get('memories', []))}")
    print(f"  • Relationships: {len(export_data.get('relationships', []))}")
    print(f"  • Routines: {len(export_data.get('routines', []))}")
    print()
    
    # Demonstrate forgetting
    print("🗑️ Selective Forgetting:")
    print("-" * 40)
    
    # Add a temporary memory
    context.remember_enhanced(
        MemoryCategory.REMINDERS, "temp_note", "Buy milk",
        MemoryImportance.LOW
    )
    print("Added temporary reminder: 'Buy milk'")
    
    # Verify it exists
    temp_memory = context.recall_enhanced(MemoryCategory.REMINDERS, "temp_note")
    print(f"Recalled: {temp_memory}")
    
    # Forget it
    if context.enhanced_context.forget(MemoryCategory.REMINDERS, "temp_note"):
        print("✓ Successfully forgot the temporary reminder")
    
    # Try to recall again
    forgotten = context.recall_enhanced(MemoryCategory.REMINDERS, "temp_note")
    print(f"After forgetting: {forgotten}")
    print()
    
    print("=" * 60)
    print("✅ Demo completed successfully!")
    print("\n🧹 Cleaning up demo databases...")
    
    # Cleanup
    for db_file in ["demo_base_memory.db", "demo_enhanced_memory.db"]:
        try:
            os.remove(db_file)
            print(f"✓ Removed {db_file}")
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    demo_context_bridge()
