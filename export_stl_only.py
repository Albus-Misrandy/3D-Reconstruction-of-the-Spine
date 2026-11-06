"""
Standalone Tool: Marching Cubes STL Mesh Extractor
Extracts 3D watertight, patient-coordinate-aligned STL meshes directly from any NIfTI segmentation mask.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from spine_segmentation import MarchingCubesSTLExporter


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-precision STL meshes from vertebrae segmentation NIfTI mask"
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        required=True,
        help="输入分割掩模路径 (.nii 或 .nii.gz)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="exported_stl",
        help="STL 文件输出目录 (默认: exported_stl)"
    )
    parser.add_argument(
        "--scope", "-s",
        type=str,
        choices=["all", "individual", "both"],
        default="both",
        help="输出范围: individual(逐椎骨独立输出), all(合并输出), both(两者均输出)"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help="指定要导出的椎骨编号或名称，例如: 'L1,L2,L3' 或 '20,21,22'"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.8,
        help="Marching Cubes 前置平滑系数 (消除台阶效应，默认 0.8)"
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.5,
        help="等值面提取阈值 (默认 0.5)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="spine",
        help="输出文件名前缀 (默认: spine)"
    )

    args = parser.parse_args()

    target_labels = None
    if args.labels:
        target_labels = [item.strip() for item in args.labels.split(",") if item.strip()]

    exporter = MarchingCubesSTLExporter(
        sigma=args.sigma,
        level=args.level
    )

    exporter.export_from_nifti(
        mask_nifti_path=args.mask,
        output_dir=args.output_dir,
        scope=args.scope,
        target_labels=target_labels,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
