#!/usr/bin/env python3
"""
Jarvis Context Bridge
Integrates the enhanced context system with the existing Jarvis infrastructure
Provides seamless memory management with backward compatibility
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from jarvis_context import JarvisContext
from jarvis_context_enhanced import (
    EnhancedJarvisContext, 
    MemoryCategory, 
    MemoryImportance
)

logger = logging.getLogger(__name__)

class JarvisContextBridge(JarvisContext):
    """
    Bridge class that extends the existing JarvisContext with enhanced memory features
    Maintains backward compatibility while adding new capabilities
    """
    
    def __init__(self, db_path: str = "jarvis_memory.db", 
                 enhanced_db_path: str = "jarvis_enhanced_memory.db",
                 encryption_key: Optional[str] = None,
                 max_session_history: int = 20, 
                 default_user: str = "default"):
        
        # Initialize the base context system
        super().__init__(db_path, max_session_history, default_user)
        
        # Add enhanced context system
        self.enhanced_context = EnhancedJarvisContext(
            db_path=enhanced_db_path,
            encryption_key=encryption_key,
            max_session_history=max_session_history
        )
        
        logger.info(f"Context Bridge initialized with enhanced memory support")
    
    def remember_enhanced(self, category: Union[str, MemoryCategory], key: str, value: Any,
                          importance: Union[int, MemoryImportance] = MemoryImportance.MEDIUM,
                          **kwargs) -> bool:
        """
        Store information in the enhanced memory system
        """
        return self.enhanced_context.remember(category, key, value, importance, **kwargs)
    
    def recall_enhanced(self, category: Optional[Union[str, MemoryCategory]] = None,
                       key: Optional[str] = None, **kwargs) -> Union[Any, List[Dict[str, Any]]]:
        """
        Recall information from the enhanced memory system
        """
        return self.enhanced_context.recall(category, key, **kwargs)
    
    def process_natural_memory(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Process natural language input to extract and store memories
        
        Example inputs:
        - "Remember that I'm allergic to peanuts"
        - "My birthday is January 15th"
        - "I prefer coffee with oat milk"
        - "My wife's name is Sarah"
        """
        input_lower = user_input.lower()
        
        # Parse different types of memory statements
        memory_stored = None
        
        # Personal information patterns
        if "birthday" in input_lower:
            # Extract birthday information
            if "is" in input_lower:
                date_part = user_input.split("is", 1)[1].strip()
                self.remember_enhanced(
                    MemoryCategory.PERSONAL, "birthday", date_part,
                    MemoryImportance.HIGH,
                    context=f"User said: {user_input}"
                )
                memory_stored = {"type": "birthday", "value": date_part}
                
        elif "allergic to" in input_lower or "allergy" in input_lower:
            # Extract allergy information
            if "allergic to" in input_lower:
                allergy = user_input.split("allergic to", 1)[1].strip().rstrip(".")
                existing = self.enhanced_context.recall(MemoryCategory.PERSONAL, "allergies")
                if existing:
                    if isinstance(existing, list):
                        existing.append(allergy)
                    else:
                        existing = [existing, allergy]
                else:
                    existing = [allergy]
                
                self.remember_enhanced(
                    MemoryCategory.PERSONAL, "allergies", existing,
                    MemoryImportance.CRITICAL,
                    context=f"User said: {user_input}"
                )
                memory_stored = {"type": "allergy", "value": allergy}
                
        elif any(word in input_lower for word in ["prefer", "like", "love", "favorite"]):
            # Extract preferences
            if "coffee" in input_lower:
                # Coffee preference
                preference = user_input.split("coffee", 1)[1].strip() if "coffee" in user_input else user_input
                self.remember_enhanced(
                    MemoryCategory.PREFERENCES, "coffee_order", preference,
                    MemoryImportance.MEDIUM,
                    context=f"User said: {user_input}"
                )
                memory_stored = {"type": "preference", "category": "coffee", "value": preference}
                
            elif "color" in input_lower:
                # Color preference
                words = user_input.split()
                for i, word in enumerate(words):
                    if word in ["color", "colour"]:
                        if i + 1 < len(words):
                            color = words[i + 1].rstrip(".,!?")
                            self.remember_enhanced(
                                MemoryCategory.PREFERENCES, "favorite_color", color,
                                MemoryImportance.LOW,
                                context=f"User said: {user_input}"
                            )
                            memory_stored = {"type": "preference", "category": "color", "value": color}
                            break
                            
        elif any(word in input_lower for word in ["wife", "husband", "partner", "spouse"]):
            # Relationship information
            relationship_type = None
            for rel in ["wife", "husband", "partner", "spouse"]:
                if rel in input_lower:
                    relationship_type = rel
                    break
            
            if relationship_type and ("name is" in input_lower or "is" in input_lower):
                name_part = user_input.split("is", 1)[1].strip().rstrip(".")
                self.enhanced_context.remember_person(
                    name_part,
                    relationship=relationship_type,
                    notes=f"Mentioned: {user_input}"
                )
                memory_stored = {"type": "relationship", "person": name_part, "relation": relationship_type}
                
        elif "i work" in input_lower or "my job" in input_lower:
            # Work information
            if "at" in input_lower:
                company = user_input.split("at", 1)[1].strip().rstrip(".")
                self.remember_enhanced(
                    MemoryCategory.WORK, "company", company,
                    MemoryImportance.HIGH,
                    context=f"User said: {user_input}"
                )
                memory_stored = {"type": "work", "company": company}
                
        elif "remember that" in input_lower or "don't forget" in input_lower:
            # General reminders
            reminder = None
            if "remember that" in input_lower:
                # Find the position case-insensitively
                idx = input_lower.find("remember that")
                if idx != -1:
                    reminder = user_input[idx + len("remember that"):].strip()
            elif "don't forget" in input_lower:
                idx = input_lower.find("don't forget")
                if idx != -1:
                    reminder = user_input[idx + len("don't forget"):].strip()
            
            if not reminder:
                reminder = user_input
            
            self.remember_enhanced(
                MemoryCategory.REMINDERS, f"reminder_{len(self.session_history)}", reminder,
                MemoryImportance.MEDIUM,
                context=f"User said: {user_input}"
            )
            memory_stored = {"type": "reminder", "value": reminder}
        
        # Also store in the base context system for backward compatibility
        if memory_stored:
            self.learn_about_user(
                str(memory_stored.get("type", "general")),
                str(memory_stored.get("value", user_input)),
                confidence=0.9,
                source="natural_language"
            )
        
        return memory_stored
    
    def get_contextual_response(self, query: str) -> Optional[str]:
        """
        Generate a contextual response based on stored memories
        
        Example queries:
        - "What am I allergic to?"
        - "When is my birthday?"
        - "What's my wife's name?"
        - "What do I like?"
        """
        query_lower = query.lower()
        
        # Check for allergy queries
        if "allergic" in query_lower or "allergy" in query_lower:
            allergies = self.enhanced_context.recall(MemoryCategory.PERSONAL, "allergies")
            if allergies:
                if isinstance(allergies, list):
                    return f"You're allergic to: {', '.join(allergies)}"
                else:
                    return f"You're allergic to {allergies}"
            return "I don't have any allergy information recorded for you."
        
        # Check for birthday queries
        elif "birthday" in query_lower:
            birthday = self.enhanced_context.recall(MemoryCategory.PERSONAL, "birthday")
            if birthday:
                return f"Your birthday is {birthday}"
            return "I don't have your birthday recorded."
        
        # Check for relationship queries
        elif any(word in query_lower for word in ["wife", "husband", "partner", "spouse"]):
            # Get all relationships
            people = self.enhanced_context.recall(MemoryCategory.RELATIONSHIPS)
            for person in people:
                if person.get('value') and any(rel in str(person.get('value', '')) for rel in ["wife", "husband", "partner", "spouse"]):
                    return f"Your {person.get('value')} is {person.get('key')}"
            
            # Also check the relationships table
            with self.lock:
                try:
                    import sqlite3
                    with sqlite3.connect(self.enhanced_context.db_path) as conn:
                        cursor = conn.execute(
                            "SELECT person_name, relationship_type FROM relationships WHERE relationship_type IN (?, ?, ?, ?)",
                            ("wife", "husband", "partner", "spouse")
                        )
                        result = cursor.fetchone()
                        if result:
                            return f"Your {result[1]} is {result[0]}"
                except:
                    pass
            
            return "I don't have information about your spouse/partner."
        
        # Check for preference queries
        elif "like" in query_lower or "prefer" in query_lower or "favorite" in query_lower:
            preferences = self.enhanced_context.recall(MemoryCategory.PREFERENCES)
            if preferences:
                pref_list = []
                for pref in preferences[:5]:  # Limit to top 5
                    pref_list.append(f"• {pref['key'].replace('_', ' ').title()}: {pref['value']}")
                return "Here are some of your preferences:\n" + "\n".join(pref_list)
            return "I don't have any preferences recorded for you yet."
        
        # Check for work queries
        elif "work" in query_lower or "job" in query_lower or "company" in query_lower:
            job = self.enhanced_context.recall(MemoryCategory.WORK, "job_title")
            company = self.enhanced_context.recall(MemoryCategory.WORK, "company")
            if job or company:
                response = "Your work information:\n"
                if job:
                    response += f"• Job: {job}\n"
                if company:
                    response += f"• Company: {company}"
                return response
            return "I don't have your work information recorded."
        
        # Check for routine queries
        elif "routine" in query_lower or "schedule" in query_lower:
            routines = self.enhanced_context.get_active_routines()
            if routines:
                routine_list = []
                for routine in routines[:5]:
                    routine_list.append(f"• {routine['name']} at {routine['time']}")
                return "Your routines:\n" + "\n".join(routine_list)
            return "No routines recorded."
        
        # General memory search
        else:
            memories = self.enhanced_context.get_contextual_memories(query, limit=3)
            if memories:
                response = "Here's what I remember:\n"
                for memory in memories:
                    response += f"• {memory['key'].replace('_', ' ').title()}: {memory['value']}\n"
                return response.strip()
        
        return None
    
    def get_enhanced_context_for_ai(self, include_persistent: bool = True) -> str:
        """
        Generate enhanced context string for AI processing
        Combines base context with enhanced memories
        """
        # Get base context
        base_context = self.get_context_for_ai(include_persistent)
        
        # Add enhanced context
        enhanced_parts = []
        
        # Add critical memories
        critical_memories = self.enhanced_context.recall(min_importance=4)
        if critical_memories:
            enhanced_parts.append("Critical Information:")
            for memory in critical_memories[:3]:
                enhanced_parts.append(f"  • [{memory['category']}] {memory['key']}: {memory['value']}")
        
        # Add recent routines
        routines = self.enhanced_context.get_active_routines()
        if routines:
            enhanced_parts.append("Active Routines:")
            for routine in routines[:2]:
                enhanced_parts.append(f"  • {routine['name']} at {routine['time']}")
        
        # Add relationships
        try:
            import sqlite3
            with sqlite3.connect(self.enhanced_context.db_path) as conn:
                cursor = conn.execute(
                    "SELECT person_name, relationship_type FROM relationships ORDER BY last_interaction DESC LIMIT 3"
                )
                relationships = cursor.fetchall()
                if relationships:
                    enhanced_parts.append("Key Relationships:")
                    for name, relation in relationships:
                        enhanced_parts.append(f"  • {name} ({relation})")
        except:
            pass
        
        # Combine contexts
        if enhanced_parts:
            return base_context + "\n\nEnhanced Memory Context:\n" + "\n".join(enhanced_parts)
        return base_context
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory system
        """
        # Get base stats
        base_stats = {
            "session_history": len(self.session_history),
            "current_topic": self.current_topic,
            "user": self.current_user_id,
            "preferences": len(self.session_preferences)
        }
        
        # Get enhanced stats
        enhanced_stats = self.enhanced_context.get_memory_summary()
        
        # Combine stats
        return {
            **base_stats,
            **enhanced_stats,
            "total_storage": base_stats.get("preferences", 0) + enhanced_stats.get("total_memories", 0)
        }

# Convenience function for creating context bridge
def create_context_bridge(db_path: str = "jarvis_memory.db",
                         enhanced_db_path: str = "jarvis_enhanced_memory.db",
                         encryption_key: Optional[str] = None,
                         max_session_history: int = 20,
                         default_user: str = "default") -> JarvisContextBridge:
    """
    Create and return a JarvisContextBridge instance
    """
    return JarvisContextBridge(
        db_path=db_path,
        enhanced_db_path=enhanced_db_path,
        encryption_key=encryption_key,
        max_session_history=max_session_history,
        default_user=default_user
    )
