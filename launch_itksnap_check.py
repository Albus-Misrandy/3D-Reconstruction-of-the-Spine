"""
Standalone Tool: ITK-SNAP Clinical Verification and Manual Correction Launcher
Loads raw CT alongside initial/refined segmentation mask for interactive editing.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from spine_segmentation import launch_itksnap_session


def main():
    parser = argparse.ArgumentParser(
        description="Launch ITK-SNAP with raw CT and segmentation overlay for clinical verification"
    )
    parser.add_argument(
        "--ct", "-c",
        type=str,
        required=True,
        help="原始脊柱 CT 图像路径 (.nii 或 .nii.gz)"
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        required=True,
        help="待核查修正的分割掩模路径 (.nii 或 .nii.gz)"
    )
    parser.add_argument(
        "--label_desc", "-l",
        type=str,
        default=None,
        help="可选: 自定义 ITK-SNAP 标签描述文件 (.txt)"
    )

    args = parser.parse_args()

    launch_itksnap_session(
        ct_image_path=args.ct,
        segmentation_path=args.mask,
        label_desc_path=args.label_desc
    )


if __name__ == "__main__":
    main()
