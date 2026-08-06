"""
scripts/train_convnext.py

Legacy entry point for Multi-Head ConvNeXt model training.
Delegates to the unified master training application (`train.py`).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import main

if __name__ == "__main__":
    main()
