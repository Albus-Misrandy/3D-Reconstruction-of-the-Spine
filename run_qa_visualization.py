"""
Standalone Tool: Clinical Multi-Planar Reconstruction (MPR) QA Slice Visualizer
Generates high-resolution orthogonal cross-sections for verifying endplates, pedicles, and canal.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from spine_segmentation import generate_qa_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate clinical multi-planar QA review slice report"
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
        help="分割掩模路径 (.nii 或 .nii.gz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="qa_reports/clinical_qa_slice.png",
        help="输出切片报告 PNG 路径 (默认: qa_reports/clinical_qa_slice.png)"
    )
    parser.add_argument(
        "--focus_label", "-f",
        type=int,
        default=None,
        help="指定视口中心对准的椎骨编号 (例如: 20 代表 L1)"
    )

    args = parser.parse_args()

    generate_qa_report(
        ct_image_path=args.ct,
        mask_path=args.mask,
        output_png_path=args.output,
        target_vertebra_label=args.focus_label
    )


if __name__ == "__main__":
    main()
