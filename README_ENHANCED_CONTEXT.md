# Jarvis Enhanced Context System

## Overview

The Jarvis Enhanced Context System provides a robust, categorized memory database that allows Jarvis to remember detailed information about you, your preferences, relationships, routines, and more. This system extends the existing Jarvis context with powerful new capabilities while maintaining backward compatibility.

## Features

### 1. **Categorized Memory Storage**
Memories are organized into intuitive categories:
- **Personal**: Name, age, birthday, allergies, blood type
- **Preferences**: Favorite colors, foods, coffee order, music
- **Work**: Job title, company, skills, projects
- **Health**: Medications, conditions, fitness goals
- **Relationships**: Family, friends, colleagues, doctors
- **Routines**: Daily schedules, recurring tasks
- **Interests**: Hobbies, favorite books/movies, travel goals
- **Technical**: Programming preferences, tools, setups
- **Reminders**: Things to remember
- **Goals**: Aspirations and objectives

### 2. **Importance Levels**
Every memory has an importance level:
- **CRITICAL (5)**: Never forget (allergies, medical conditions)
- **HIGH (4)**: Very important (birthday, anniversary)
- **MEDIUM (3)**: Useful to remember (preferences, habits)
- **LOW (2)**: Nice to know (minor preferences)
- **TRIVIAL (1)**: Can be forgotten if needed

### 3. **Natural Language Processing**
Tell Jarvis things naturally:
```
"Remember that I'm allergic to peanuts"
"My birthday is January 15th"
"I prefer coffee with oat milk"
"My wife's name is Sarah"
"I work at Tech Innovations Inc"
```

### 4. **Intelligent Querying**
Ask Jarvis questions naturally:
```
"What am I allergic to?"
"When is my birthday?"
"What's my wife's name?"
"What do I like?"
"Where do I work?"
```

### 5. **Relationship Management**
Store detailed information about people:
- Names and nicknames
- Relationship type (family, friend, colleague)
- Contact information
- Important dates (birthdays)
- Notes and context

### 6. **Routine Tracking**
Remember your daily schedules:
- Morning workouts
- Team meetings
- Medication times
- Regular appointments

### 7. **Contextual Search**
Search memories by:
- Category
- Tags
- Keywords
- Importance level
- Context

### 8. **Data Security**
- Optional encryption for sensitive data
- Separate databases for base and enhanced features
- User-specific data isolation

### 9. **Export & Backup**
- Export all memories for backup
- Selective export (exclude encrypted data)
- JSON format for portability

## Installation

The enhanced context system is already integrated into Jarvis. No additional installation required.

## Usage

### Basic Usage with Jarvis

When talking to Jarvis, simply tell him things you want him to remember:

```python
# In your conversation with Jarvis
"Jarvis, remember that I'm allergic to shellfish"
"My favorite programming language is Python"
"I have a meeting with Bob every Tuesday at 10 AM"
```

### Programmatic Usage

```python
from jarvis_context_bridge import create_context_bridge, MemoryCategory, MemoryImportance

# Initialize the context bridge
context = create_context_bridge()

# Store a memory
context.remember_enhanced(
    MemoryCategory.PERSONAL, 
    "allergies", 
    ["peanuts", "shellfish"],
    MemoryImportance.CRITICAL
)

# Recall a memory
allergies = context.recall_enhanced(MemoryCategory.PERSONAL, "allergies")

# Natural language processing
context.process_natural_memory("My birthday is March 22nd")

# Get contextual response
response = context.get_contextual_response("When is my birthday?")
```

### Advanced Features

#### Store Complex Information
```python
# Add a person
context.enhanced_context.remember_person(
    "Dr. Emily Chen",
    relationship="doctor",
    phone="555-1234",
    email="dr.chen@medical.com",
    notes="Primary care physician, very thorough"
)

# Add a routine
context.enhanced_context.add_routine(
    "Morning Run",
    routine_type="exercise",
    days=["Monday", "Wednesday", "Friday"],
    time="6:30 AM",
    duration=45,
    description="5K run in the park"
)
```

#### Search Memories
```python
# Search by context
health_memories = context.enhanced_context.get_contextual_memories("health exercise", limit=5)

# Get memories by category
work_info = context.recall_enhanced(MemoryCategory.WORK)

# Get high-importance memories only
critical_info = context.recall_enhanced(min_importance=4)
```

#### Export and Backup
```python
# Export all memories
backup = context.enhanced_context.export_memories()

# Get memory statistics
stats = context.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
```

## Database Schema

### Main Tables

1. **memories**: Core memory storage with categories, importance, and metadata
2. **relationships**: Detailed information about people
3. **routines**: Scheduled activities and recurring tasks
4. **locations**: Places and addresses
5. **memory_associations**: Links between related memories
6. **memory_patterns**: Usage patterns for intelligent retrieval

## Privacy and Security

- All data is stored locally on your machine
- Sensitive data can be encrypted with a custom key
- No data is sent to external servers
- You can selectively forget specific memories
- Full export capability for data portability

## Examples

### Example 1: Health Information
```python
context.remember_enhanced(MemoryCategory.HEALTH, "blood_type", "O+", MemoryImportance.CRITICAL)
context.remember_enhanced(MemoryCategory.HEALTH, "medications", ["Vitamin D", "Omega-3"], MemoryImportance.HIGH)
context.remember_enhanced(MemoryCategory.HEALTH, "exercise_goal", "Run 5K in under 25 minutes", MemoryImportance.MEDIUM)
```

### Example 2: Work Setup
```python
context.remember_enhanced(MemoryCategory.TECHNICAL, "favorite_ide", "VS Code", MemoryImportance.MEDIUM)
context.remember_enhanced(MemoryCategory.TECHNICAL, "programming_languages", ["Python", "TypeScript", "Go"], MemoryImportance.MEDIUM)
context.remember_enhanced(MemoryCategory.WORK, "current_project", "AI Assistant Platform", MemoryImportance.HIGH)
```

### Example 3: Personal Preferences
```python
context.remember_enhanced(MemoryCategory.PREFERENCES, "coffee_order", "Large cappuccino with oat milk", MemoryImportance.MEDIUM)
context.remember_enhanced(MemoryCategory.INTERESTS, "hobbies", ["photography", "hiking", "cooking"], MemoryImportance.MEDIUM)
context.remember_enhanced(MemoryCategory.INTERESTS, "travel_bucket_list", ["Japan", "Iceland", "New Zealand"], MemoryImportance.LOW)
```

## Testing

Run the test scripts to verify functionality:

```bash
# Test enhanced context system
python test_enhanced_context.py

# Test context bridge with natural language
python demo_context_bridge.py
```

## Architecture

```
jarvis_context.py           # Original context system
    ↓
jarvis_context_enhanced.py  # Enhanced memory system
    ↓
jarvis_context_bridge.py    # Integration bridge
    ↓
Jarvis AI System            # Main assistant
```

## Future Enhancements

- [ ] Semantic search using embeddings
- [ ] Auto-categorization of memories
- [ ] Memory consolidation and cleanup
- [ ] Time-based memory decay
- [ ] Cross-reference detection
- [ ] Pattern learning from usage
- [ ] Voice profile integration
- [ ] Multi-device sync
- [ ] Memory visualization dashboard

## Troubleshooting

### Memory Not Being Stored
- Check the category is valid
- Ensure the database file has write permissions
- Verify the key is unique within the category

### Natural Language Not Recognized
- The system recognizes specific patterns
- Try rephrasing with keywords like "remember", "my", "I prefer"
- Check the process_natural_memory() method for supported patterns

### Database Issues
- Delete the .db files to start fresh
- Check disk space
- Ensure SQLite is properly installed

## Contributing

To add new memory patterns or categories:
1. Add the category to the MemoryCategory enum
2. Update the natural language processing in process_natural_memory()
3. Add query handlers in get_contextual_response()
4. Test with the demo scripts

## License

Part of the Jarvis AI Assistant System
