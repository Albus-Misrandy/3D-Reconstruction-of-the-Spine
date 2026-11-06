"""
Spine Vertebrae Segmentation and Surgical Planning STL Mesh Pipeline Entry Point
"""

import sys
from pathlib import Path

# Insert workspace root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from main_pipeline import main

if __name__ == "__main__":
    main()
