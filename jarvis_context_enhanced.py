#!/usr/bin/env python3
"""
Enhanced Jarvis Context and Memory System
Provides comprehensive personal information storage, advanced preference management,
and intelligent context retrieval with categorized memory storage
"""

import sqlite3
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class MemoryCategory(Enum):
    """Categories for organizing different types of memories"""
    PERSONAL = "personal"          # Name, age, birthday, family
    PREFERENCES = "preferences"     # Likes, dislikes, favorites
    ROUTINES = "routines"          # Daily habits, schedules
    RELATIONSHIPS = "relationships" # People, connections
    WORK = "work"                  # Job, projects, skills
    HEALTH = "health"              # Medical, fitness, diet
    LOCATIONS = "locations"        # Places, addresses
    INTERESTS = "interests"        # Hobbies, topics of interest
    REMINDERS = "reminders"        # Things to remember
    GOALS = "goals"                # Aspirations, objectives
    EXPERIENCES = "experiences"    # Past events, stories
    TECHNICAL = "technical"        # Tech preferences, devices
    FINANCIAL = "financial"        # Budget, expenses (encrypted)
    SECRETS = "secrets"            # Sensitive info (encrypted)

class MemoryImportance(Enum):
    """Importance levels for memories"""
    CRITICAL = 5  # Never forget (like allergies, medical conditions)
    HIGH = 4      # Very important (birthday, anniversary)
    MEDIUM = 3    # Useful to remember (preferences, habits)
    LOW = 2       # Nice to know (minor preferences)
    TRIVIAL = 1   # Can be forgotten if needed

class EnhancedJarvisContext:
    """
    Enhanced context system with robust personal information management
    """
    
    def __init__(self, db_path: str = "jarvis_enhanced_memory.db", 
                 encryption_key: Optional[str] = None,
                 max_session_history: int = 50):
        self.db_path = Path(db_path)
        self.encryption_key = encryption_key
        self.max_session_history = max_session_history
        self.lock = threading.RLock()
        
        # Session data
        self.session_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        
        # Memory cache for faster retrieval
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamp = datetime.now()
        self.cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes
        
        # Initialize enhanced database
        self._init_enhanced_database()
        
        # Load initial memories into cache
        self._refresh_cache()
        
        logger.info(f"Enhanced Jarvis context initialized - DB: {self.db_path}")
    
    def _init_enhanced_database(self):
        """Initialize enhanced database schema with new tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enhanced memories table with categories and importance
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        data_type TEXT DEFAULT 'string',
                        importance INTEGER DEFAULT 3,
                        confidence REAL DEFAULT 1.0,
                        is_encrypted BOOLEAN DEFAULT 0,
                        source TEXT DEFAULT 'conversation',
                        context TEXT,
                        tags TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        expires_at DATETIME,
                        UNIQUE(category, key)
                    )
                ''')
                
                # Relationships table - people and connections
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS relationships (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_name TEXT NOT NULL UNIQUE,
                        relationship_type TEXT,
                        first_name TEXT,
                        last_name TEXT,
                        nickname TEXT,
                        importance INTEGER DEFAULT 3,
                        birthday DATE,
                        phone TEXT,
                        email TEXT,
                        address TEXT,
                        notes TEXT,
                        tags TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_interaction DATETIME
                    )
                ''')
                
                # Routines and schedules
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS routines (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        routine_name TEXT NOT NULL,
                        routine_type TEXT,
                        schedule TEXT,
                        days_of_week TEXT,
                        time_of_day TEXT,
                        duration_minutes INTEGER,
                        description TEXT,
                        reminder_enabled BOOLEAN DEFAULT 0,
                        importance INTEGER DEFAULT 3,
                        tags TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Locations and places
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS locations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        location_name TEXT NOT NULL UNIQUE,
                        location_type TEXT,
                        address TEXT,
                        coordinates TEXT,
                        description TEXT,
                        notes TEXT,
                        visit_frequency TEXT,
                        last_visited DATETIME,
                        tags TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Contextual associations - link memories together
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memory_associations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id_1 INTEGER NOT NULL,
                        memory_id_2 INTEGER NOT NULL,
                        association_type TEXT,
                        strength REAL DEFAULT 0.5,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (memory_id_1) REFERENCES memories (id),
                        FOREIGN KEY (memory_id_2) REFERENCES memories (id),
                        UNIQUE(memory_id_1, memory_id_2)
                    )
                ''')
                
                # Memory access patterns for intelligent retrieval
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memory_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_type TEXT,
                        pattern_data TEXT,
                        frequency INTEGER DEFAULT 1,
                        last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                        context TEXT
                    )
                ''')
                
                # Create indexes for faster queries
                conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_name ON relationships(person_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_routines_active ON routines(is_active)')
                
                conn.commit()
                logger.debug("Enhanced database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize enhanced database: {e}")
            raise
    
    def _encrypt_value(self, value: str) -> str:
        """Simple encryption for sensitive data (implement proper encryption in production)"""
        if not self.encryption_key:
            return value
        # This is a placeholder - use proper encryption library in production
        return hashlib.sha256((value + self.encryption_key).encode()).hexdigest()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive data (implement proper decryption in production)"""
        # This is a placeholder - use proper encryption library in production
        return encrypted_value  # Return as-is for now
    
    def remember(self, category: Union[str, MemoryCategory], key: str, value: Any, 
                 importance: Union[int, MemoryImportance] = MemoryImportance.MEDIUM,
                 subcategory: Optional[str] = None, tags: Optional[List[str]] = None,
                 context: Optional[str] = None, encrypt: bool = False) -> bool:
        """
        Store a memory with enhanced metadata and categorization
        
        Args:
            category: Category of the memory
            key: Unique key within the category
            value: Value to remember
            importance: How important this memory is
            subcategory: Optional subcategory for organization
            tags: Optional list of tags for searching
            context: Optional context about when/how this was learned
            encrypt: Whether to encrypt this value
        
        Returns:
            Success status
        """
        with self.lock:
            try:
                # Convert enums to values
                if isinstance(category, MemoryCategory):
                    category = category.value
                if isinstance(importance, MemoryImportance):
                    importance = importance.value
                
                # Determine data type
                data_type = 'string'
                if isinstance(value, bool):
                    data_type = 'bool'
                elif isinstance(value, int):
                    data_type = 'int'
                elif isinstance(value, float):
                    data_type = 'float'
                elif isinstance(value, (dict, list)):
                    data_type = 'json'
                
                # Convert value to string for storage
                if data_type == 'json':
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                
                # Encrypt if requested
                if encrypt:
                    value_str = self._encrypt_value(value_str)
                
                # Convert tags to string
                tags_str = json.dumps(tags) if tags else None
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO memories 
                        (category, subcategory, key, value, data_type, importance, 
                         is_encrypted, context, tags, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (category, subcategory, key, value_str, data_type, 
                          importance, encrypt, context, tags_str))
                    
                    conn.commit()
                    
                    # Invalidate cache
                    self.memory_cache.clear()
                    
                    logger.info(f"Remembered: [{category}] {key} = {value[:50] if len(str(value)) > 50 else value}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to remember {key}: {e}")
                return False
    
    def recall(self, category: Optional[Union[str, MemoryCategory]] = None, 
               key: Optional[str] = None, tags: Optional[List[str]] = None,
               min_importance: int = 1) -> Union[Any, List[Dict[str, Any]]]:
        """
        Recall memories based on various criteria
        
        Args:
            category: Optional category to filter by
            key: Optional specific key to retrieve
            tags: Optional tags to search for
            min_importance: Minimum importance level
        
        Returns:
            Single memory value if key specified, list of memories otherwise
        """
        with self.lock:
            try:
                # Check cache first
                cache_key = f"{category}:{key}" if key else f"{category}:all"
                if cache_key in self.memory_cache and self._is_cache_valid():
                    if key:
                        return self.memory_cache[cache_key]
                    
                with sqlite3.connect(self.db_path) as conn:
                    if key and category:
                        # Retrieve specific memory
                        if isinstance(category, MemoryCategory):
                            category = category.value
                        
                        cursor = conn.execute('''
                            SELECT value, data_type, is_encrypted, access_count
                            FROM memories 
                            WHERE category = ? AND key = ?
                        ''', (category, key))
                        
                        result = cursor.fetchone()
                        if result:
                            value_str, data_type, is_encrypted, access_count = result
                            
                            # Update access count and timestamp
                            conn.execute('''
                                UPDATE memories 
                                SET accessed_at = CURRENT_TIMESTAMP, 
                                    access_count = access_count + 1
                                WHERE category = ? AND key = ?
                            ''', (category, key))
                            conn.commit()
                            
                            # Decrypt if needed
                            if is_encrypted:
                                value_str = self._decrypt_value(value_str)
                            
                            # Parse value based on type
                            value = self._parse_value(value_str, data_type)
                            
                            # Cache the result
                            self.memory_cache[cache_key] = value
                            
                            return value
                        return None
                    
                    else:
                        # Retrieve multiple memories based on criteria
                        query = '''
                            SELECT category, subcategory, key, value, data_type, 
                                   importance, context, tags, created_at, updated_at
                            FROM memories 
                            WHERE importance >= ?
                        '''
                        params = [min_importance]
                        
                        if category:
                            if isinstance(category, MemoryCategory):
                                category = category.value
                            query += ' AND category = ?'
                            params.append(category)
                        
                        if tags:
                            # Search for any matching tag
                            tag_conditions = []
                            for tag in tags:
                                tag_conditions.append('tags LIKE ?')
                                params.append(f'%"{tag}"%')
                            query += f' AND ({" OR ".join(tag_conditions)})'
                        
                        query += ' ORDER BY importance DESC, updated_at DESC'
                        
                        cursor = conn.execute(query, params)
                        
                        memories = []
                        for row in cursor.fetchall():
                            memory = {
                                'category': row[0],
                                'subcategory': row[1],
                                'key': row[2],
                                'value': self._parse_value(row[3], row[4]),
                                'importance': row[5],
                                'context': row[6],
                                'tags': json.loads(row[7]) if row[7] else [],
                                'created_at': row[8],
                                'updated_at': row[9]
                            }
                            memories.append(memory)
                        
                        return memories
                        
            except Exception as e:
                logger.error(f"Failed to recall memory: {e}")
                return None if key else []
    
    def _parse_value(self, value_str: str, data_type: str) -> Any:
        """Parse stored string value back to original type"""
        try:
            if data_type == 'json':
                return json.loads(value_str)
            elif data_type == 'int':
                return int(value_str)
            elif data_type == 'float':
                return float(value_str)
            elif data_type == 'bool':
                return value_str.lower() == 'true'
            else:
                return value_str
        except Exception as e:
            logger.warning(f"Failed to parse value: {e}")
            return value_str
    
    def _is_cache_valid(self) -> bool:
        """Check if memory cache is still valid"""
        return datetime.now() - self.cache_timestamp < self.cache_ttl
    
    def _refresh_cache(self):
        """Refresh the memory cache"""
        self.memory_cache.clear()
        self.cache_timestamp = datetime.now()
    
    def remember_person(self, name: str, relationship: Optional[str] = None,
                       birthday: Optional[str] = None, phone: Optional[str] = None,
                       email: Optional[str] = None, notes: Optional[str] = None,
                       **kwargs) -> bool:
        """
        Remember information about a person
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Check if person exists
                    cursor = conn.execute(
                        "SELECT id FROM relationships WHERE person_name = ?", (name,)
                    )
                    exists = cursor.fetchone() is not None
                    
                    if exists:
                        # Update existing person
                        updates = []
                        params = []
                        
                        if relationship:
                            updates.append("relationship_type = ?")
                            params.append(relationship)
                        if birthday:
                            updates.append("birthday = ?")
                            params.append(birthday)
                        if phone:
                            updates.append("phone = ?")
                            params.append(phone)
                        if email:
                            updates.append("email = ?")
                            params.append(email)
                        if notes:
                            updates.append("notes = ?")
                            params.append(notes)
                        
                        for key, value in kwargs.items():
                            if key in ['first_name', 'last_name', 'nickname', 'address', 'tags']:
                                updates.append(f"{key} = ?")
                                params.append(value if key != 'tags' else json.dumps(value))
                        
                        if updates:
                            updates.append("updated_at = CURRENT_TIMESTAMP")
                            params.append(name)
                            
                            query = f'''
                                UPDATE relationships 
                                SET {", ".join(updates)}
                                WHERE person_name = ?
                            '''
                            conn.execute(query, params)
                    else:
                        # Insert new person
                        conn.execute('''
                            INSERT INTO relationships 
                            (person_name, relationship_type, birthday, phone, email, notes)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (name, relationship, birthday, phone, email, notes))
                    
                    conn.commit()
                    logger.info(f"Remembered person: {name} ({relationship})")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to remember person {name}: {e}")
                return False
    
    def recall_person(self, name: str) -> Optional[Dict[str, Any]]:
        """Recall information about a person"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT person_name, relationship_type, first_name, last_name,
                               nickname, birthday, phone, email, address, notes, tags,
                               created_at, updated_at, last_interaction
                        FROM relationships 
                        WHERE person_name = ? OR nickname = ?
                    ''', (name, name))
                    
                    result = cursor.fetchone()
                    if result:
                        # Update last interaction
                        conn.execute('''
                            UPDATE relationships 
                            SET last_interaction = CURRENT_TIMESTAMP
                            WHERE person_name = ?
                        ''', (result[0],))
                        conn.commit()
                        
                        return {
                            'name': result[0],
                            'relationship': result[1],
                            'first_name': result[2],
                            'last_name': result[3],
                            'nickname': result[4],
                            'birthday': result[5],
                            'phone': result[6],
                            'email': result[7],
                            'address': result[8],
                            'notes': result[9],
                            'tags': json.loads(result[10]) if result[10] else [],
                            'created_at': result[11],
                            'updated_at': result[12],
                            'last_interaction': result[13]
                        }
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to recall person {name}: {e}")
                return None
    
    def add_routine(self, name: str, routine_type: str, schedule: Optional[str] = None,
                   days: Optional[List[str]] = None, time: Optional[str] = None,
                   duration: Optional[int] = None, description: Optional[str] = None,
                   reminder: bool = False) -> bool:
        """Add or update a routine"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    days_str = json.dumps(days) if days else None
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO routines 
                        (routine_name, routine_type, schedule, days_of_week, 
                         time_of_day, duration_minutes, description, reminder_enabled)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (name, routine_type, schedule, days_str, time, 
                          duration, description, reminder))
                    
                    conn.commit()
                    logger.info(f"Added routine: {name}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to add routine {name}: {e}")
                return False
    
    def get_active_routines(self) -> List[Dict[str, Any]]:
        """Get all active routines"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT routine_name, routine_type, schedule, days_of_week,
                               time_of_day, duration_minutes, description, reminder_enabled
                        FROM routines 
                        WHERE is_active = 1
                        ORDER BY time_of_day
                    ''')
                    
                    routines = []
                    for row in cursor.fetchall():
                        routines.append({
                            'name': row[0],
                            'type': row[1],
                            'schedule': row[2],
                            'days': json.loads(row[3]) if row[3] else [],
                            'time': row[4],
                            'duration': row[5],
                            'description': row[6],
                            'reminder': bool(row[7])
                        })
                    
                    return routines
                    
            except Exception as e:
                logger.error(f"Failed to get routines: {e}")
                return []
    
    def get_contextual_memories(self, context_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories relevant to a context query using semantic search
        (This is a simple implementation - could be enhanced with embeddings)
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Simple keyword-based search across multiple fields
                    keywords = context_query.lower().split()
                    
                    conditions = []
                    params = []
                    for keyword in keywords:
                        conditions.append('''
                            (LOWER(key) LIKE ? OR 
                             LOWER(value) LIKE ? OR 
                             LOWER(context) LIKE ? OR 
                             LOWER(tags) LIKE ?)
                        ''')
                        params.extend([f'%{keyword}%'] * 4)
                    
                    query = f'''
                        SELECT category, key, value, data_type, importance, context, tags
                        FROM memories 
                        WHERE {' OR '.join(conditions)}
                        ORDER BY importance DESC, accessed_at DESC
                        LIMIT ?
                    '''
                    params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    
                    memories = []
                    for row in cursor.fetchall():
                        memories.append({
                            'category': row[0],
                            'key': row[1],
                            'value': self._parse_value(row[2], row[3]),
                            'importance': row[4],
                            'context': row[5],
                            'tags': json.loads(row[6]) if row[6] else []
                        })
                    
                    return memories
                    
            except Exception as e:
                logger.error(f"Failed to get contextual memories: {e}")
                return []
    
    def forget(self, category: Union[str, MemoryCategory], key: str) -> bool:
        """
        Forget a specific memory
        """
        with self.lock:
            try:
                if isinstance(category, MemoryCategory):
                    category = category.value
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM memories WHERE category = ? AND key = ?",
                        (category, key)
                    )
                    
                    if cursor.rowcount > 0:
                        conn.commit()
                        self.memory_cache.clear()
                        logger.info(f"Forgot: [{category}] {key}")
                        return True
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to forget {key}: {e}")
                return False
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of all stored memories"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Count memories by category
                    cursor = conn.execute('''
                        SELECT category, COUNT(*), AVG(importance)
                        FROM memories 
                        GROUP BY category
                    ''')
                    
                    categories = {}
                    for row in cursor.fetchall():
                        categories[row[0]] = {
                            'count': row[1],
                            'avg_importance': round(row[2], 2) if row[2] else 0
                        }
                    
                    # Count total memories
                    cursor = conn.execute("SELECT COUNT(*) FROM memories")
                    total_memories = cursor.fetchone()[0]
                    
                    # Count relationships
                    cursor = conn.execute("SELECT COUNT(*) FROM relationships")
                    total_people = cursor.fetchone()[0]
                    
                    # Count routines
                    cursor = conn.execute("SELECT COUNT(*) FROM routines WHERE is_active = 1")
                    active_routines = cursor.fetchone()[0]
                    
                    # Get most important memories
                    cursor = conn.execute('''
                        SELECT category, key, importance
                        FROM memories 
                        WHERE importance >= 4
                        ORDER BY importance DESC, accessed_at DESC
                        LIMIT 5
                    ''')
                    
                    important_memories = []
                    for row in cursor.fetchall():
                        important_memories.append({
                            'category': row[0],
                            'key': row[1],
                            'importance': row[2]
                        })
                    
                    return {
                        'total_memories': total_memories,
                        'categories': categories,
                        'total_people': total_people,
                        'active_routines': active_routines,
                        'important_memories': important_memories,
                        'cache_size': len(self.memory_cache),
                        'session_exchanges': len(self.session_history)
                    }
                    
            except Exception as e:
                logger.error(f"Failed to get memory summary: {e}")
                return {}
    
    def export_memories(self, include_encrypted: bool = False) -> Dict[str, Any]:
        """Export all memories for backup or transfer"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Export memories
                    query = "SELECT * FROM memories"
                    if not include_encrypted:
                        query += " WHERE is_encrypted = 0"
                    
                    cursor = conn.execute(query)
                    memories = []
                    for row in cursor.fetchall():
                        memory = {
                            'category': row[1],
                            'subcategory': row[2],
                            'key': row[3],
                            'value': self._parse_value(row[4], row[5]),
                            'data_type': row[5],
                            'importance': row[6],
                            'context': row[10],
                            'tags': json.loads(row[11]) if row[11] else [],
                            'created_at': row[12],
                            'updated_at': row[13]
                        }
                        memories.append(memory)
                    
                    # Export relationships
                    cursor = conn.execute("SELECT * FROM relationships")
                    relationships = []
                    for row in cursor.fetchall():
                        relationships.append({
                            'name': row[1],
                            'relationship': row[2],
                            'birthday': row[7],
                            'phone': row[8],
                            'email': row[9],
                            'notes': row[11]
                        })
                    
                    # Export routines
                    cursor = conn.execute("SELECT * FROM routines WHERE is_active = 1")
                    routines = []
                    for row in cursor.fetchall():
                        routines.append({
                            'name': row[1],
                            'type': row[2],
                            'schedule': row[3],
                            'days': json.loads(row[4]) if row[4] else [],
                            'time': row[5],
                            'duration': row[6],
                            'description': row[7]
                        })
                    
                    return {
                        'export_date': datetime.now().isoformat(),
                        'memories': memories,
                        'relationships': relationships,
                        'routines': routines,
                        'total_items': len(memories) + len(relationships) + len(routines)
                    }
                    
            except Exception as e:
                logger.error(f"Failed to export memories: {e}")
                return {}

# Convenience function for creating enhanced context instance
def create_enhanced_context(db_path: str = "jarvis_enhanced_memory.db", 
                           encryption_key: Optional[str] = None) -> EnhancedJarvisContext:
    """
    Create and return an EnhancedJarvisContext instance
    """
    return EnhancedJarvisContext(db_path=db_path, encryption_key=encryption_key)
