#!/usr/bin/env python3
"""
Speaker Identification System for Jarvis
========================================

This module provides speaker identification capabilities to automatically
identify users by their voice characteristics, eliminating the need for
manual user announcements.

Key Features:
- Local processing for privacy
- Progressive enrollment for improved accuracy
- Integration with existing STT pipeline
- Support for multiple identification backends
- Fallback mechanisms for robustness

Supported Backends:
- SpeechBrain (ECAPA-TDNN) - High accuracy
- PyTorch-based lightweight models - Fast processing
- Traditional i-Vector approach - Fallback option
"""

import os
import json
import logging
import numpy as np
import torch
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import pickle
from collections import defaultdict, deque

# Import config manager for settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add parent directory to path to import config_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class SpeakerProfile:
    """Represents a speaker's voice profile"""
    user_id: str
    name: str
    embeddings: List[np.ndarray]  # Multiple embeddings for robustness
    enrollment_samples: int
    last_updated: float
    confidence_threshold: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class IdentificationResult:
    """Result of speaker identification"""
    user_id: Optional[str]
    confidence: float
    speaker_name: Optional[str] = None
    is_enrolled: bool = False
    suggested_enrollment: bool = False

class SpeakerDatabase:
    """Manages speaker profiles and embeddings storage"""
    
    def __init__(self, db_path: str = "speaker_profiles.db"):
        self.db_path = db_path
        self.profiles: Dict[str, SpeakerProfile] = {}
        self._lock = threading.RLock()
        self._init_database()
        self._load_profiles()
    
    def _init_database(self):
        """Initialize SQLite database for speaker profiles"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embeddings BLOB NOT NULL,
                    enrollment_samples INTEGER DEFAULT 0,
                    last_updated REAL NOT NULL,
                    confidence_threshold REAL DEFAULT 0.75,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    confidence REAL,
                    audio_duration REAL,
                    model_used TEXT
                )
            """)
    
    def _load_profiles(self):
        """Load speaker profiles from database"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT * FROM speaker_profiles")
                    for row in cursor.fetchall():
                        user_id, name, embeddings_blob, samples, last_updated, threshold, meta_str = row
                        
                        # Deserialize embeddings
                        embeddings = pickle.loads(embeddings_blob)
                        metadata = json.loads(meta_str)
                        
                        profile = SpeakerProfile(
                            user_id=user_id,
                            name=name,
                            embeddings=embeddings,
                            enrollment_samples=samples,
                            last_updated=last_updated,
                            confidence_threshold=threshold,
                            metadata=metadata
                        )
                        
                        self.profiles[user_id] = profile
                        
                logger.info(f"Loaded {len(self.profiles)} speaker profiles")
                
            except Exception as e:
                logger.error(f"Failed to load speaker profiles: {e}")
    
    def save_profile(self, profile: SpeakerProfile):
        """Save or update a speaker profile"""
        with self._lock:
            try:
                embeddings_blob = pickle.dumps(profile.embeddings)
                metadata_str = json.dumps(profile.metadata)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO speaker_profiles 
                        (user_id, name, embeddings, enrollment_samples, last_updated, confidence_threshold, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        profile.user_id, profile.name, embeddings_blob,
                        profile.enrollment_samples, profile.last_updated,
                        profile.confidence_threshold, metadata_str
                    ))
                
                self.profiles[profile.user_id] = profile
                logger.info(f"Saved profile for user: {profile.name}")
                
            except Exception as e:
                logger.error(f"Failed to save speaker profile: {e}")
    
    def get_profile(self, user_id: str) -> Optional[SpeakerProfile]:
        """Get a speaker profile by user ID"""
        with self._lock:
            return self.profiles.get(user_id)
    
    def list_profiles(self) -> List[SpeakerProfile]:
        """Get all speaker profiles"""
        with self._lock:
            return list(self.profiles.values())
    
    def delete_profile(self, user_id: str) -> bool:
        """Delete a speaker profile"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM speaker_profiles WHERE user_id = ?", (user_id,))
                
                if user_id in self.profiles:
                    del self.profiles[user_id]
                    logger.info(f"Deleted profile for user: {user_id}")
                    return True
                
            except Exception as e:
                logger.error(f"Failed to delete speaker profile: {e}")
        
        return False
    
    def log_identification(self, user_id: Optional[str], confidence: float, 
                          audio_duration: float, model_used: str):
        """Log identification attempt for analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO identification_history 
                    (timestamp, user_id, confidence, audio_duration, model_used)
                    VALUES (?, ?, ?, ?, ?)
                """, (time.time(), user_id, confidence, audio_duration, model_used))
        except Exception as e:
            logger.error(f"Failed to log identification: {e}")

class SpeechBrainIdentifier:
    """SpeechBrain-based speaker identification"""
    
    def __init__(self):
        self.model = None
        self.available = False
        self._init_model()
    
    def _init_model(self):
        """Initialize SpeechBrain model"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            # Try to load pre-trained ECAPA-TDNN model
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain_ecapa"
            )
            
            self.available = True
            logger.info("SpeechBrain ECAPA-TDNN model loaded successfully")
            
        except ImportError:
            logger.warning("SpeechBrain not available. Install with: pip install speechbrain")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
    
    def extract_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio"""
        if not self.available:
            return None
        
        try:
            # Ensure audio is float32 and proper shape
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(1, -1)
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Failed to extract SpeechBrain embedding: {e}")
            return None

class LightweightIdentifier:
    """Lightweight fallback identifier using traditional methods"""
    
    def __init__(self):
        self.available = True
        logger.info("Lightweight identifier initialized")
    
    def extract_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract simple MFCC-based embedding"""
        try:
            import librosa
            
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            
            # Create simple statistical embedding
            embedding = np.concatenate([
                np.mean(mfccs, axis=1),    # Mean of each MFCC
                np.std(mfccs, axis=1),     # Std of each MFCC
                np.median(mfccs, axis=1)   # Median of each MFCC
            ])
            
            return embedding
            
        except ImportError:
            logger.error("librosa not available for lightweight identifier")
            return self._simple_spectral_embedding(audio_data, sample_rate)
        except Exception as e:
            logger.error(f"Failed to extract lightweight embedding: {e}")
            return None
    
    def _simple_spectral_embedding(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Very simple spectral features as fallback"""
        # Basic spectral features
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Simple statistical features
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.median(magnitude),
            np.max(magnitude),
            np.percentile(magnitude, 25),
            np.percentile(magnitude, 75)
        ]
        
        return np.array(features)

class SpeakerIdentificationSystem:
    """Main speaker identification system"""
    
    def __init__(self, db_path: str = "speaker_profiles.db"):
        self.db = SpeakerDatabase(db_path)
        
        # Initialize identification backends
        self.speechbrain = SpeechBrainIdentifier()
        self.lightweight = LightweightIdentifier()
        
        # Choose primary backend
        self.primary_backend = self.speechbrain if self.speechbrain.available else self.lightweight
        self.backend_name = "speechbrain" if self.speechbrain.available else "lightweight"
        
        # Configuration
        config = get_config()
        speaker_config = config.get('speaker_identification', {})
        
        self.confidence_threshold = speaker_config.get('confidence_threshold', 0.75)
        self.enrollment_threshold = speaker_config.get('enrollment_threshold', 3)  # Min samples for enrollment
        self.similarity_metric = speaker_config.get('similarity_metric', 'cosine')
        self.auto_enrollment_enabled = speaker_config.get('auto_enrollment', False)
        
        # Runtime state
        self.recent_identifications = deque(maxlen=10)  # Recent results for smoothing
        self.identification_history = defaultdict(list)
        
        # Callbacks
        self.on_user_identified: Optional[Callable[[str, float], None]] = None
        self.on_unknown_speaker: Optional[Callable[[float], None]] = None
        self.on_enrollment_suggested: Optional[Callable[[str], None]] = None
        
        logger.info(f"Speaker identification system initialized with {self.backend_name} backend")
    
    def set_callbacks(self, 
                     on_user_identified: Callable[[str, float], None] = None,
                     on_unknown_speaker: Callable[[float], None] = None,
                     on_enrollment_suggested: Callable[[str], None] = None):
        """Set callbacks for identification events"""
        self.on_user_identified = on_user_identified
        self.on_unknown_speaker = on_unknown_speaker
        self.on_enrollment_suggested = on_enrollment_suggested
    
    def enroll_user(self, user_id: str, name: str, audio_data: np.ndarray, 
                   sample_rate: int = 16000) -> bool:
        """Enroll a new user or add sample to existing user"""
        try:
            # Extract embedding
            embedding = self.primary_backend.extract_embedding(audio_data, sample_rate)
            if embedding is None:
                logger.error("Failed to extract embedding for enrollment")
                return False
            
            # Get or create profile
            profile = self.db.get_profile(user_id)
            if profile is None:
                profile = SpeakerProfile(
                    user_id=user_id,
                    name=name,
                    embeddings=[],
                    enrollment_samples=0,
                    last_updated=time.time(),
                    confidence_threshold=self.confidence_threshold
                )
            
            # Add embedding
            profile.embeddings.append(embedding)
            profile.enrollment_samples += 1
            profile.last_updated = time.time()
            
            # Keep only recent embeddings to prevent profile drift
            max_embeddings = 10
            if len(profile.embeddings) > max_embeddings:
                profile.embeddings = profile.embeddings[-max_embeddings:]
            
            # Save profile
            self.db.save_profile(profile)
            
            logger.info(f"Enrolled sample for {name} ({profile.enrollment_samples} total samples)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enroll user {name}: {e}")
            return False
    
    def identify_speaker(self, audio_data: np.ndarray, 
                        sample_rate: int = 16000) -> IdentificationResult:
        """Identify speaker from audio data"""
        start_time = time.time()
        
        try:
            # Extract embedding from audio
            embedding = self.primary_backend.extract_embedding(audio_data, sample_rate)
            if embedding is None:
                return IdentificationResult(
                    user_id=None,
                    confidence=0.0,
                    suggested_enrollment=False
                )
            
            # Compare with all enrolled profiles
            best_match = None
            best_similarity = 0.0
            
            profiles = self.db.list_profiles()
            if not profiles:
                # No enrolled users
                return IdentificationResult(
                    user_id=None,
                    confidence=0.0,
                    suggested_enrollment=True
                )
            
            for profile in profiles:
                # Calculate similarity with all embeddings in profile
                similarities = []
                for profile_embedding in profile.embeddings:
                    similarity = self._calculate_similarity(embedding, profile_embedding)
                    similarities.append(similarity)
                
                # Use best similarity for this profile
                profile_similarity = max(similarities) if similarities else 0.0
                
                if profile_similarity > best_similarity:
                    best_similarity = profile_similarity
                    best_match = profile
            
            # Determine if identification is confident enough
            if best_match and best_similarity >= best_match.confidence_threshold:
                result = IdentificationResult(
                    user_id=best_match.user_id,
                    confidence=best_similarity,
                    speaker_name=best_match.name,
                    is_enrolled=True,
                    suggested_enrollment=False
                )
                
                # Log successful identification
                audio_duration = len(audio_data) / sample_rate
                self.db.log_identification(
                    best_match.user_id, best_similarity, 
                    audio_duration, self.backend_name
                )
                
                # Update identification history
                self.identification_history[best_match.user_id].append({
                    'timestamp': time.time(),
                    'confidence': best_similarity,
                    'audio_duration': audio_duration
                })
                
                # Call callback
                if self.on_user_identified:
                    self.on_user_identified(best_match.user_id, best_similarity)
                
            else:
                # Unknown speaker or low confidence
                result = IdentificationResult(
                    user_id=None,
                    confidence=best_similarity,
                    suggested_enrollment=self.auto_enrollment_enabled
                )
                
                # Log unknown speaker
                audio_duration = len(audio_data) / sample_rate
                self.db.log_identification(
                    None, best_similarity, audio_duration, self.backend_name
                )
                
                # Call callbacks
                if self.on_unknown_speaker:
                    self.on_unknown_speaker(best_similarity)
                
                if result.suggested_enrollment and self.on_enrollment_suggested:
                    self.on_enrollment_suggested("unknown_speaker")
            
            # Store recent identification for smoothing
            self.recent_identifications.append(result)
            
            processing_time = time.time() - start_time
            logger.debug(f"Speaker identification completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return IdentificationResult(
                user_id=None,
                confidence=0.0,
                suggested_enrollment=False
            )
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two embeddings"""
        try:
            if self.similarity_metric == 'cosine':
                # Cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                # Convert from [-1, 1] to [0, 1]
                return (similarity + 1) / 2
            
            elif self.similarity_metric == 'euclidean':
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                # Convert distance to similarity (0 = identical, higher = more different)
                similarity = 1 / (1 + distance)
                return similarity
            
            else:
                logger.warning(f"Unknown similarity metric: {self.similarity_metric}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user"""
        profile = self.db.get_profile(user_id)
        if not profile:
            return {}
        
        history = self.identification_history.get(user_id, [])
        recent_history = [h for h in history if time.time() - h['timestamp'] < 86400]  # Last 24h
        
        return {
            'name': profile.name,
            'enrollment_samples': profile.enrollment_samples,
            'last_updated': profile.last_updated,
            'confidence_threshold': profile.confidence_threshold,
            'total_identifications': len(history),
            'recent_identifications': len(recent_history),
            'avg_confidence': np.mean([h['confidence'] for h in recent_history]) if recent_history else 0.0,
            'backend': self.backend_name
        }
    
    def list_enrolled_users(self) -> List[Dict]:
        """List all enrolled users with stats"""
        users = []
        for profile in self.db.list_profiles():
            stats = self.get_user_stats(profile.user_id)
            users.append(stats)
        return users
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user profile"""
        if user_id in self.identification_history:
            del self.identification_history[user_id]
        return self.db.delete_profile(user_id)
    
    def update_confidence_threshold(self, user_id: str, threshold: float) -> bool:
        """Update confidence threshold for a user"""
        profile = self.db.get_profile(user_id)
        if profile:
            profile.confidence_threshold = threshold
            self.db.save_profile(profile)
            return True
        return False

# Example usage and testing
if __name__ == "__main__":
    # Create identification system
    speaker_id = SpeakerIdentificationSystem()
    
    def on_user_identified(user_id: str, confidence: float):
        print(f"✅ User identified: {user_id} (confidence: {confidence:.2f})")
    
    def on_unknown_speaker(confidence: float):
        print(f"❓ Unknown speaker detected (best match confidence: {confidence:.2f})")
    
    def on_enrollment_suggested(speaker_id: str):
        print(f"💡 Consider enrolling this speaker: {speaker_id}")
    
    # Set callbacks
    speaker_id.set_callbacks(
        on_user_identified=on_user_identified,
        on_unknown_speaker=on_unknown_speaker,
        on_enrollment_suggested=on_enrollment_suggested
    )
    
    print("Speaker identification system initialized")
    print("Available backends:", "SpeechBrain" if speaker_id.speechbrain.available else "Lightweight only")
    print("To test, use the enrollment and identification methods with audio data")