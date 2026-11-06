"""
Spine Segmentation and Surgical Planning Configuration
Configuration file defining anatomical label mappings, model paths, and processing parameters.
"""

import os
from pathlib import Path

# Workspace Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Default Model Paths
MODEL_DIR = BASE_DIR / "CTSpine1K"
DEFAULT_CHECKPOINT_MODEL = MODEL_DIR / "model_final_checkpoint.model"
DEFAULT_CHECKPOINT_PKL = MODEL_DIR / "model_final_checkpoint.model.pkl"
DEFAULT_PLANS_PKL = MODEL_DIR / "plans.pkl"

# ITK-SNAP Executable Candidate Locations
ITK_SNAP_PATHS = [
    r"D:\ITK-SNAP 4.4\bin\ITK-SNAP.exe",
    r"C:\Program Files\ITK-SNAP 4.4\bin\ITK-SNAP.exe",
    r"C:\Program Files (x86)\ITK-SNAP 4.4\bin\ITK-SNAP.exe",
    r"D:\Program Files\ITK-SNAP 4.4\bin\ITK-SNAP.exe",
]

# Vertebrae anatomical label mapping based on VerSe / CTSpine1K standards:
# 1-7: Cervical (C1-C7)
# 8-19: Thoracic (T1-T12)
# 20-24: Lumbar (L1-L5)
# 25: Sacrum / S1
VERTEBRAE_LABELS = {
    1: {"name": "C1", "chinese": "寰椎(C1)", "group": "Cervical", "color": [255, 0, 0]},
    2: {"name": "C2", "chinese": "枢椎(C2)", "group": "Cervical", "color": [0, 255, 0]},
    3: {"name": "C3", "chinese": "颈椎(C3)", "group": "Cervical", "color": [0, 0, 255]},
    4: {"name": "C4", "chinese": "颈椎(C4)", "group": "Cervical", "color": [255, 255, 0]},
    5: {"name": "C5", "chinese": "颈椎(C5)", "group": "Cervical", "color": [0, 255, 255]},
    6: {"name": "C6", "chinese": "颈椎(C6)", "group": "Cervical", "color": [255, 0, 255]},
    7: {"name": "C7", "chinese": "隆椎(C7)", "group": "Cervical", "color": [192, 192, 192]},
    8: {"name": "T1", "chinese": "胸椎(T1)", "group": "Thoracic", "color": [128, 0, 0]},
    9: {"name": "T2", "chinese": "胸椎(T2)", "group": "Thoracic", "color": [128, 128, 0]},
    10: {"name": "T3", "chinese": "胸椎(T3)", "group": "Thoracic", "color": [0, 128, 0]},
    11: {"name": "T4", "chinese": "胸椎(T4)", "group": "Thoracic", "color": [128, 0, 128]},
    12: {"name": "T5", "chinese": "胸椎(T5)", "group": "Thoracic", "color": [0, 128, 128]},
    13: {"name": "T6", "chinese": "胸椎(T6)", "group": "Thoracic", "color": [0, 0, 128]},
    14: {"name": "T7", "chinese": "胸椎(T7)", "group": "Thoracic", "color": [255, 165, 0]},
    15: {"name": "T8", "chinese": "胸椎(T8)", "group": "Thoracic", "color": [255, 215, 0]},
    16: {"name": "T9", "chinese": "胸椎(T9)", "group": "Thoracic", "color": [173, 255, 47]},
    17: {"name": "T10", "chinese": "胸椎(T10)", "group": "Thoracic", "color": [34, 139, 34]},
    18: {"name": "T11", "chinese": "胸椎(T11)", "group": "Thoracic", "color": [0, 250, 154]},
    19: {"name": "T12", "chinese": "胸椎(T12)", "group": "Thoracic", "color": [72, 209, 204]},
    20: {"name": "L1", "chinese": "腰椎(L1)", "group": "Lumbar", "color": [30, 144, 255]},
    21: {"name": "L2", "chinese": "腰椎(L2)", "group": "Lumbar", "color": [138, 43, 226]},
    22: {"name": "L3", "chinese": "腰椎(L3)", "group": "Lumbar", "color": [255, 20, 147]},
    23: {"name": "L4", "chinese": "腰椎(L4)", "group": "Lumbar", "color": [255, 105, 180]},
    24: {"name": "L5", "chinese": "腰椎(L5)", "group": "Lumbar", "color": [240, 128, 128]},
    25: {"name": "Sacrum", "chinese": "骶骨(Sacrum/S1)", "group": "Sacrum", "color": [210, 105, 30]},
}

# Reverse mapping for convenience
NAME_TO_LABEL = {info["name"].upper(): label for label, info in VERTEBRAE_LABELS.items()}

# Marching Cubes Default Parameters
MARCHING_CUBES_DEFAULTS = {
    "sigma": 0.8,              # Gaussian pre-filter sigma to eliminate staircase artifacts
    "level": 0.5,              # Isosurface threshold on smoothed binary mask
    "keep_largest_component": True,  # Filter out floating debris and spurious voxels
    "min_component_voxels": 200,     # Discard components smaller than this voxel count
    "taubin_iterations": 10,   # Taubin surface mesh smoothing iterations (non-shrinking)
    "taubin_nu": -0.53,
    "taubin_lambda": 0.5,
}

def find_itksnap_path() -> str | None:
    """Finds the ITK-SNAP executable path on the local machine."""
    for p in ITK_SNAP_PATHS:
        if os.path.exists(p):
            return p
    return None
