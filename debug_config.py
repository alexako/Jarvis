#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from commands.commands import create_ai_config

# Test the configuration when prefer_anthropic=False
config = create_ai_config(prefer_anthropic=False)
print("Configuration when prefer_anthropic=False:")
print(f"Anthropic priority: {config['providers']['anthropic']['priority']}")
print(f"DeepSeek priority: {config['providers']['deepseek']['priority']}")

# Test the configuration when prefer_anthropic=True
config = create_ai_config(prefer_anthropic=True)
print("\nConfiguration when prefer_anthropic=True:")
print(f"Anthropic priority: {config['providers']['anthropic']['priority']}")
print(f"DeepSeek priority: {config['providers']['deepseek']['priority']}")