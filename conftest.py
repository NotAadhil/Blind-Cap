"""
conftest.py - Pytest Configuration
====================================
Ensures the project root is on sys.path so test imports work.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
