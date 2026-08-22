"""
tools/hardware_check.py - Standalone Hardware Diagnostics
==========================================================
Run::

    python tools/hardware_check.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hardware import run_diagnostics

if __name__ == "__main__":
    run_diagnostics(verbose=True)
