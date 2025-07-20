#!/usr/bin/env python3
"""
Config Manager - Centralized configuration management for Jarvis
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Singleton configuration manager for Jarvis project"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None):
        """Load configuration from JSON file"""
        if config_path is None:
            # Look for config.json in the project root
            project_root = Path(__file__).parent
            config_path = project_root / "config.json"
        
        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'audio.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if self._config is None:
            self.load_config()
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_audio_config(self, performance_mode: str = None) -> Dict[str, Any]:
        """Get audio configuration section with optional performance mode"""
        if performance_mode and performance_mode in ['fast', 'balanced', 'accurate']:
            mode_config = self.get(f'performance_modes.{performance_mode}.audio', {})
            if mode_config:
                return mode_config
        return self.get('audio', {})
    
    def get_stt_config(self, performance_mode: str = None) -> Dict[str, Any]:
        """Get STT configuration section with optional performance mode"""
        if performance_mode and performance_mode in ['fast', 'balanced', 'accurate']:
            mode_config = self.get(f'performance_modes.{performance_mode}.stt', {})
            if mode_config:
                return mode_config
        return self.get('stt', {})
    
    def get_optimizations_config(self) -> Dict[str, Any]:
        """Get optimizations configuration section"""
        return self.get('optimizations', {})
    
    def get_wake_words(self) -> list:
        """Get wake words list"""
        return self.get('wake_words', ['jarvis', 'hey jarvis'])
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration section"""
        return self.get('debug', {})
    
    def get_tts_config(self, performance_mode: str = None) -> Dict[str, Any]:
        """Get TTS configuration section with optional performance mode"""
        if performance_mode and performance_mode in ['fast', 'balanced', 'accurate']:
            mode_config = self.get(f'performance_modes.{performance_mode}.tts', {})
            if mode_config:
                # Merge with base TTS config
                base_config = self.get('tts', {})
                merged_config = base_config.copy()
                merged_config.update(mode_config)
                return merged_config
        return self.get('tts', {})
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        if self._config is None:
            self.load_config()
        
        keys = key_path.split('.')
        current = self._config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration back to file"""
        if config_path is None:
            project_root = Path(__file__).parent
            config_path = project_root / "config.json"
        
        with open(config_path, 'w') as f:
            json.dump(self._config, f, indent=2)


# Convenience function for getting config instance
def get_config() -> ConfigManager:
    """Get the global config manager instance"""
    return ConfigManager()