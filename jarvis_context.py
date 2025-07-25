#!/usr/bin/env python3
"""
Jarvis Context and Memory System
Provides conversational memory, user preferences, and contextual awareness
"""

import sqlite3
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class JarvisContext:
    """
    Comprehensive context and memory system for Jarvis
    Combines session memory with persistent SQLite storage
    """
    
    def __init__(self, db_path: str = "jarvis_memory.db", max_session_history: int = 20, default_user: str = "default"):
        self.db_path = Path(db_path)
        self.max_session_history = max_session_history
        self.lock = threading.RLock()
        
        # Multi-user support
        self.current_user_id: str = default_user
        self.default_user_id: str = default_user
        self.users_cache: Dict[str, Dict[str, Any]] = {}
        
        # Session memory (temporary, in-memory) - now user-specific
        self.session_history: List[Dict[str, Any]] = []
        self.current_topic: Optional[str] = None
        self.session_start = datetime.now()
        self.user_name: Optional[str] = None
        self.session_preferences: Dict[str, Any] = {}
        
        # Initialize database
        self._init_database()
        
        # Load persistent data for default user
        self._load_user_data()
        
        logger.info(f"Jarvis multi-user context initialized - DB: {self.db_path}, Default user: {default_user}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables for multi-user support"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Users table - central user registry
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                        voice_profile TEXT,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Conversations table - now user-specific
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        session_id TEXT,
                        user_input TEXT NOT NULL,
                        jarvis_response TEXT NOT NULL,
                        topic TEXT,
                        context_data TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # User preferences table - now user-specific
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        data_type TEXT DEFAULT 'string',
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, key),
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Topics and subjects discussed - now user-specific
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS topics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        topic_name TEXT NOT NULL,
                        first_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP,
                        mention_count INTEGER DEFAULT 1,
                        context_summary TEXT,
                        UNIQUE(user_id, topic_name),
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # User information and learning - now user-specific
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_info (
                        user_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        source TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, key),
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # User aliases table - multiple names per user
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_aliases (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        alias TEXT NOT NULL,
                        alias_type TEXT DEFAULT 'nickname',
                        is_primary BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, alias),
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Create default user if it doesn't exist
                conn.execute('''
                    INSERT OR IGNORE INTO users (user_id, display_name)
                    VALUES (?, ?)
                ''', (self.default_user_id, "Default User"))
                
                conn.commit()
                logger.debug("Multi-user database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _load_user_data(self, user_id: Optional[str] = None):
        """Load persistent user data and preferences for specific user"""
        if user_id is None:
            user_id = self.current_user_id
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load user display name and update last_active
                cursor = conn.execute(
                    "SELECT display_name FROM users WHERE user_id = ?", (user_id,)
                )
                result = cursor.fetchone()
                if result:
                    self.user_name = result[0]
                    # Update last active
                    conn.execute(
                        "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?", 
                        (user_id,)
                    )
                    logger.info(f"Loaded user: {self.user_name} (ID: {user_id})")
                else:
                    logger.warning(f"User {user_id} not found in database")
                    self.user_name = None
                
                # Load session preferences from user_preferences
                self.session_preferences.clear()
                cursor = conn.execute(
                    "SELECT key, value, data_type FROM user_preferences WHERE user_id = ?", 
                    (user_id,)
                )
                for key, value, data_type in cursor.fetchall():
                    try:
                        if data_type == 'json':
                            self.session_preferences[key] = json.loads(value)
                        elif data_type == 'int':
                            self.session_preferences[key] = int(value)
                        elif data_type == 'float':
                            self.session_preferences[key] = float(value)
                        elif data_type == 'bool':
                            self.session_preferences[key] = value.lower() == 'true'
                        else:
                            self.session_preferences[key] = value
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse preference {key}: {e}")
                
                conn.commit()
                logger.debug(f"Loaded {len(self.session_preferences)} preferences for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to load user data for {user_id}: {e}")
    
    def add_exchange(self, user_input: str, jarvis_response: str, topic: Optional[str] = None, 
                    context_data: Optional[Dict[str, Any]] = None):
        """
        Add a conversation exchange to both session and persistent memory
        """
        with self.lock:
            timestamp = datetime.now()
            session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
            
            # Add to session history
            exchange = {
                'timestamp': timestamp,
                'user_input': user_input,
                'jarvis_response': jarvis_response,
                'topic': topic or self.current_topic,
                'context_data': context_data
            }
            
            self.session_history.append(exchange)
            
            # Maintain session history size
            if len(self.session_history) > self.max_session_history:
                self.session_history.pop(0)
            
            # Update current topic if provided
            if topic:
                self.current_topic = topic
                self._update_topic(topic)
            
            # Save to persistent storage
            self._save_conversation(session_id, user_input, jarvis_response, topic, context_data, timestamp)
            
            logger.debug(f"Added exchange to context - Topic: {topic}, Session size: {len(self.session_history)}")
    
    def _save_conversation(self, session_id: str, user_input: str, jarvis_response: str, 
                          topic: Optional[str], context_data: Optional[Dict[str, Any]], 
                          timestamp: datetime):
        """Save conversation to persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                context_json = json.dumps(context_data) if context_data else None
                
                conn.execute('''
                    INSERT INTO conversations 
                    (user_id, timestamp, session_id, user_input, jarvis_response, topic, context_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (self.current_user_id, timestamp.isoformat(), session_id, user_input, jarvis_response, topic, context_json))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def _update_topic(self, topic: str):
        """Update topic tracking in database for current user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if topic exists for current user
                cursor = conn.execute(
                    "SELECT mention_count FROM topics WHERE user_id = ? AND topic_name = ?", 
                    (self.current_user_id, topic)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing topic
                    conn.execute('''
                        UPDATE topics 
                        SET last_mentioned = CURRENT_TIMESTAMP, mention_count = mention_count + 1
                        WHERE user_id = ? AND topic_name = ?
                    ''', (self.current_user_id, topic))
                else:
                    # Create new topic for current user
                    conn.execute('''
                        INSERT INTO topics (user_id, topic_name, context_summary)
                        VALUES (?, ?, ?)
                    ''', (self.current_user_id, topic, f"Discussion about {topic}"))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update topic: {e}")
    
    # User Management Methods
    def create_user(self, user_id: str, display_name: str, voice_profile: Optional[str] = None) -> bool:
        """Create a new user in the system"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO users (user_id, display_name, voice_profile)
                    VALUES (?, ?, ?)
                ''', (user_id, display_name, voice_profile))
                conn.commit()
                logger.info(f"Created new user: {display_name} (ID: {user_id})")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"User {user_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            return False
    
    def switch_user(self, user_identifier: str, display_name: Optional[str] = None) -> bool:
        """Switch to a different user by ID or alias, creating if necessary"""
        with self.lock:
            try:
                # First try to find user by alias/name
                found_user_id = self.find_user_by_alias(user_identifier)
                
                if found_user_id:
                    # User found by alias
                    user_id = found_user_id
                else:
                    # Try as direct user_id or create new user
                    user_id = user_identifier.lower().replace(" ", "_")
                    
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute("SELECT display_name FROM users WHERE user_id = ?", (user_id,))
                        result = cursor.fetchone()
                        
                        if not result:
                            # Create user if it doesn't exist
                            if display_name:
                                self.create_user(user_id, display_name)
                            else:
                                # Use user_identifier as display name
                                self.create_user(user_id, user_identifier.title())
                
                # Switch to the user
                old_user = self.current_user_id
                self.current_user_id = user_id
                
                # Clear session data and reload for new user
                self.session_history.clear()
                self.current_topic = None
                self._load_user_data(user_id)
                
                logger.info(f"Switched from user {old_user} to {self.current_user_id} ({self.user_name})")
                return True
                    
            except Exception as e:
                logger.error(f"Failed to switch to user {user_identifier}: {e}")
                return False
    
    def get_current_user(self) -> Dict[str, str]:
        """Get current user information"""
        return {
            'user_id': self.current_user_id,
            'display_name': self.user_name or self.current_user_id,
            'is_default': self.current_user_id == self.default_user_id
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """Get list of all users in the system"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT user_id, display_name, created_at, last_active, is_active
                    FROM users 
                    ORDER BY last_active DESC
                ''')
                
                users = []
                for row in cursor.fetchall():
                    users.append({
                        'user_id': row[0],
                        'display_name': row[1],
                        'created_at': row[2],
                        'last_active': row[3],
                        'is_active': bool(row[4]),
                        'is_current': row[0] == self.current_user_id
                    })
                
                return users
                
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all associated data"""
        if user_id == self.default_user_id:
            logger.warning("Cannot delete default user")
            return False
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete user data in order (foreign key constraints)
                conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM topics WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM user_info WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                
                conn.commit()
                
                # If we just deleted the current user, switch to default
                if user_id == self.current_user_id:
                    self.switch_user(self.default_user_id)
                
                logger.info(f"Deleted user {user_id} and all associated data")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    # User Alias Management Methods
    def add_user_alias(self, alias: str, alias_type: str = "nickname", user_id: Optional[str] = None) -> bool:
        """Add an alias/alternative name for a user"""
        if user_id is None:
            user_id = self.current_user_id
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO user_aliases (user_id, alias, alias_type)
                    VALUES (?, ?, ?)
                ''', (user_id, alias.strip(), alias_type))
                conn.commit()
                logger.info(f"Added alias '{alias}' for user {user_id}")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Alias '{alias}' already exists for user {user_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to add alias: {e}")
            return False
    
    def find_user_by_alias(self, name: str) -> Optional[str]:
        """Find user_id by any of their names/aliases"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First check display names
                cursor = conn.execute(
                    "SELECT user_id FROM users WHERE LOWER(display_name) = LOWER(?)", 
                    (name.strip(),)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                
                # Then check aliases
                cursor = conn.execute(
                    "SELECT user_id FROM user_aliases WHERE LOWER(alias) = LOWER(?)", 
                    (name.strip(),)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to find user by alias: {e}")
            return None
    
    def get_user_aliases(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all aliases for a user"""
        if user_id is None:
            user_id = self.current_user_id
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT alias, alias_type, is_primary, created_at
                    FROM user_aliases 
                    WHERE user_id = ?
                    ORDER BY is_primary DESC, created_at ASC
                ''', (user_id,))
                
                aliases = []
                for row in cursor.fetchall():
                    aliases.append({
                        'alias': row[0],
                        'type': row[1],
                        'is_primary': bool(row[2]),
                        'created_at': row[3]
                    })
                
                return aliases
                
        except Exception as e:
            logger.error(f"Failed to get user aliases: {e}")
            return []
    
    def remove_user_alias(self, alias: str, user_id: Optional[str] = None) -> bool:
        """Remove an alias from a user"""
        if user_id is None:
            user_id = self.current_user_id
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM user_aliases WHERE user_id = ? AND LOWER(alias) = LOWER(?)",
                    (user_id, alias.strip())
                )
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.info(f"Removed alias '{alias}' from user {user_id}")
                else:
                    logger.warning(f"Alias '{alias}' not found for user {user_id}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to remove alias: {e}")
            return False
    
    def set_primary_alias(self, alias: str, user_id: Optional[str] = None) -> bool:
        """Set an alias as the primary name for a user"""
        if user_id is None:
            user_id = self.current_user_id
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, unset all primary flags for this user
                conn.execute(
                    "UPDATE user_aliases SET is_primary = 0 WHERE user_id = ?",
                    (user_id,)
                )
                
                # Set the specified alias as primary
                cursor = conn.execute('''
                    UPDATE user_aliases 
                    SET is_primary = 1 
                    WHERE user_id = ? AND LOWER(alias) = LOWER(?)
                ''', (user_id, alias.strip()))
                
                if cursor.rowcount > 0:
                    # Also update the users table display_name
                    conn.execute(
                        "UPDATE users SET display_name = ? WHERE user_id = ?",
                        (alias.strip(), user_id)
                    )
                    
                    # Update session if this is current user
                    if user_id == self.current_user_id:
                        self.user_name = alias.strip()
                    
                    conn.commit()
                    logger.info(f"Set '{alias}' as primary name for user {user_id}")
                    return True
                else:
                    logger.warning(f"Alias '{alias}' not found for user {user_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to set primary alias: {e}")
            return False
    
    def get_recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history for context"""
        with self.lock:
            return self.session_history[-limit:] if self.session_history else []
    
    def get_context_for_ai(self, include_persistent: bool = True) -> str:
        """
        Generate context string for AI processing
        Includes recent conversation and relevant persistent data
        """
        context_parts = []
        
        # Add user information
        if self.user_name:
            context_parts.append(f"User's name: {self.user_name}")
        
        # Add session info
        session_duration = datetime.now() - self.session_start
        context_parts.append(f"Current session duration: {session_duration}")
        
        # Add current topic
        if self.current_topic:
            context_parts.append(f"Current topic: {self.current_topic}")
        
        # Add recent conversation history
        recent_exchanges = self.get_recent_context(3)
        if recent_exchanges:
            context_parts.append("Recent conversation:")
            for i, exchange in enumerate(recent_exchanges, 1):
                context_parts.append(f"  {i}. User: {exchange['user_input']}")
                context_parts.append(f"     Jarvis: {exchange['jarvis_response']}")
        
        # Add persistent context if requested
        if include_persistent:
            persistent_context = self._get_relevant_persistent_context()
            if persistent_context:
                context_parts.append("Relevant background:")
                context_parts.extend(persistent_context)
        
        return "\n".join(context_parts)
    
    def _get_relevant_persistent_context(self, limit: int = 3) -> List[str]:
        """Get relevant persistent context based on current topic"""
        context = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent topics
                cursor = conn.execute('''
                    SELECT topic_name, mention_count, context_summary
                    FROM topics 
                    ORDER BY last_mentioned DESC 
                    LIMIT ?
                ''', (limit,))
                
                topics = cursor.fetchall()
                if topics:
                    for topic_name, count, summary in topics:
                        context.append(f"Previously discussed: {topic_name} ({count} times) - {summary}")
                
                # Get user preferences that might be relevant
                if self.session_preferences:
                    prefs = [f"{k}: {v}" for k, v in list(self.session_preferences.items())[:3]]
                    if prefs:
                        context.append(f"User preferences: {', '.join(prefs)}")
                
        except Exception as e:
            logger.error(f"Failed to get persistent context: {e}")
        
        return context
    
    def learn_about_user(self, key: str, value: str, confidence: float = 1.0, source: str = "conversation"):
        """
        Learn and store information about the current user
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO user_info (user_id, key, value, confidence, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (self.current_user_id, key, value, confidence, source))
                    
                    conn.commit()
                    
                    # Update session data if it's the user's name
                    if key == 'name':
                        # Also update the users table display_name
                        conn.execute('''
                            UPDATE users SET display_name = ? WHERE user_id = ?
                        ''', (value, self.current_user_id))
                        conn.commit()
                        
                        self.user_name = value
                        logger.info(f"Learned user's name: {value} (ID: {self.current_user_id})")
                    
            except Exception as e:
                logger.error(f"Failed to learn about user: {e}")
    
    def set_preference(self, key: str, value: Any, data_type: Optional[str] = None):
        """
        Set a user preference with automatic type detection for current user
        """
        with self.lock:
            # Auto-detect data type if not provided
            if data_type is None:
                if isinstance(value, bool):
                    data_type = 'bool'
                elif isinstance(value, int):
                    data_type = 'int'
                elif isinstance(value, float):
                    data_type = 'float'
                elif isinstance(value, (dict, list)):
                    data_type = 'json'
                else:
                    data_type = 'string'
            
            # Convert value to string for storage
            if data_type == 'json':
                value_str = json.dumps(value)
            elif data_type == 'bool':
                value_str = str(value).lower()
            else:
                value_str = str(value)
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO user_preferences (user_id, key, value, data_type, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (self.current_user_id, key, value_str, data_type))
                    
                    conn.commit()
                    
                    # Update session preferences
                    self.session_preferences[key] = value
                    logger.debug(f"Set preference for user {self.current_user_id}: {key} = {value}")
                    
            except Exception as e:
                logger.error(f"Failed to set preference: {e}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference with fallback to default"""
        return self.session_preferences.get(key, default)
    
    def handle_pronouns_and_references(self, text: str) -> str:
        """
        Handle pronouns and contextual references in user input
        Returns enhanced text with context substitutions
        """
        text_lower = text.lower()
        recent_context = self.get_recent_context(2)
        
        if not recent_context:
            return text
        
        # Simple pronoun handling
        if any(pronoun in text_lower for pronoun in ['it', 'that', 'this', 'them']):
            # Get the last topic or subject mentioned
            last_exchange = recent_context[-1]
            if last_exchange.get('topic'):
                # Replace contextual references with the topic
                enhanced_text = f"{text} (referring to: {last_exchange['topic']})"
                logger.debug(f"Enhanced input with context: {enhanced_text}")
                return enhanced_text
        
        return text
    
    def get_conversation_summary(self, days: int = 7) -> str:
        """
        Get a summary of conversations from the past N days
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                cursor = conn.execute('''
                    SELECT COUNT(*) as exchange_count,
                           COUNT(DISTINCT topic) as topic_count,
                           COUNT(DISTINCT session_id) as session_count
                    FROM conversations 
                    WHERE timestamp > ?
                ''', (cutoff_date,))
                
                stats = cursor.fetchone()
                
                # Get top topics
                cursor = conn.execute('''
                    SELECT topic, COUNT(*) as count
                    FROM conversations 
                    WHERE timestamp > ? AND topic IS NOT NULL
                    GROUP BY topic
                    ORDER BY count DESC
                    LIMIT 5
                ''', (cutoff_date,))
                
                topics = cursor.fetchall()
                
                summary = f"Past {days} days: {stats[0]} exchanges across {stats[2]} sessions"
                if topics:
                    topic_list = [f"{topic} ({count})" for topic, count in topics]
                    summary += f"\nTop topics: {', '.join(topic_list)}"
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return "Unable to generate conversation summary"
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up old conversation data to prevent database bloat
        Keeps user preferences and recent topics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
                
                # Delete old conversations
                result = conn.execute(
                    "DELETE FROM conversations WHERE timestamp < ?", 
                    (cutoff_date,)
                )
                deleted_count = result.rowcount
                
                # Update topic last_mentioned dates to prevent orphaned topics
                conn.execute('''
                    DELETE FROM topics 
                    WHERE last_mentioned < ? AND mention_count < 3
                ''', (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old conversation records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def export_context_data(self) -> Dict[str, Any]:
        """
        Export context data for backup or analysis
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = {
                    'session_info': {
                        'session_start': self.session_start.isoformat(),
                        'user_name': self.user_name,
                        'current_topic': self.current_topic,
                        'session_history_count': len(self.session_history)
                    },
                    'preferences': self.session_preferences.copy(),
                    'recent_history': self.session_history.copy()
                }
                
                # Add database stats
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                data['database_stats'] = {
                    'total_conversations': cursor.fetchone()[0],
                }
                
                cursor = conn.execute("SELECT COUNT(*) FROM topics")
                data['database_stats']['total_topics'] = cursor.fetchone()[0]
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to export context data: {e}")
            return {}
    
    def reset_session(self):
        """
        Reset session data while keeping persistent storage
        """
        with self.lock:
            self.session_history.clear()
            self.current_topic = None
            self.session_start = datetime.now()
            logger.info("Session context reset")
    
    def get_context_status(self) -> str:
        """
        Get current context system status for debugging
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                total_conversations = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM user_preferences")
                total_preferences = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM topics")
                total_topics = cursor.fetchone()[0]
        except:
            total_conversations = total_preferences = total_topics = 0
        
        status = f"""Context System Status:
        • Session history: {len(self.session_history)} exchanges
        • Current topic: {self.current_topic or 'None'}
        • User name: {self.user_name or 'Unknown'}
        • Session preferences: {len(self.session_preferences)}
        • Total conversations: {total_conversations}
        • Total topics: {total_topics}
        • Total preferences: {total_preferences}
        • Database: {self.db_path}"""
        
        return status

# Convenience function for creating context instance
def create_jarvis_context(db_path: str = "jarvis_memory.db", max_session_history: int = 20, default_user: str = "default") -> JarvisContext:
    """
    Create and return a JarvisContext instance with multi-user support
    """
    return JarvisContext(db_path=db_path, max_session_history=max_session_history, default_user=default_user)