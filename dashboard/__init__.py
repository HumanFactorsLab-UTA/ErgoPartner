"""
Dashboard package initialization - centralized path setup for all dashboard modules.

This eliminates the need to repeat sys.path manipulation in every file.
All dashboard pages and utilities should import from here instead of computing PROJECT_ROOT inline.
"""

import sys

# Add project root (parent of dashboard/) to Python path using paths module
from paths import PROJECT_ROOT_STR

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)
