"""
Spine Segmentation and Surgical Planning Mesh Generation Package
"""

from spine_segmentation.config import (
    VERTEBRAE_LABELS,
    NAME_TO_LABEL,
    MARCHING_CUBES_DEFAULTS,
    find_itksnap_path
)
from spine_segmentation.predictor import SpinePredictor
from spine_segmentation.marching_cubes_stl import MarchingCubesSTLExporter, extract_vertebra_mesh
from spine_segmentation.itksnap_helper import launch_itksnap_session, generate_itksnap_label_description
from spine_segmentation.visualizer_qa import generate_qa_report
from spine_segmentation.finetuner import finetune_on_clinical_cohort

__all__ = [
    "VERTEBRAE_LABELS",
    "NAME_TO_LABEL",
    "MARCHING_CUBES_DEFAULTS",
    "find_itksnap_path",
    "SpinePredictor",
    "MarchingCubesSTLExporter",
    "extract_vertebra_mesh",
    "launch_itksnap_session",
    "generate_itksnap_label_description",
    "generate_qa_report",
    "finetune_on_clinical_cohort"
]
