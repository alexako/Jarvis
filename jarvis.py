#!/usr/bin/env python3
"""
Main entry point for Jarvis Voice Assistant
"""

import sys
import os

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.jarvis_assistant import main

if __name__ == "__main__":
    main()