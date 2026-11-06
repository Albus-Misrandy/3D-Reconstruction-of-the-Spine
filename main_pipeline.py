"""
End-to-End Spinal Vertebrae Segmentation and STL Mesh Generation Pipeline
Corresponds to the clinical pipeline described in the literature:
  1. nnU-Net pre-trained inference on spinal CT
  2. Multi-planar QA slice review & ITK-SNAP manual inspection/refinement
  3. Marching Cubes isosurface extraction
  4. Surgical planning STL triangle mesh generation
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from spine_segmentation import (
    SpinePredictor,
    MarchingCubesSTLExporter,
    generate_qa_report,
    launch_itksnap_session,
    VERTEBRAE_LABELS
)


def run_pipeline(
    input_ct_path: str,
    output_dir: str = "output_results",
    mask_input: str = None,
    scope: str = "both",
    target_labels: list = None,
    fast_mode: bool = True,
    interactive_check: bool = False,
    sigma: float = 0.8,
    level: float = 0.5,
    prefix: str = "spine"
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*75)
    print("      🏥 脊柱 CT 深度学习分割与手术规划 STL 网格生成流水线")
    print("#"*75)

    # ---------------------------------------------------------
    # 步骤 1: 深度学习辅助分割 (nnU-Net Inference)
    # ---------------------------------------------------------
    if mask_input is not None and os.path.exists(mask_input):
        print(f"\n[阶段 1] 检测到已提供分割掩模，跳过深度学习推理:")
        print(f"         {mask_input}")
        current_mask = Path(mask_input).resolve()
    else:
        if not input_ct_path or not os.path.exists(input_ct_path):
            raise FileNotFoundError(f"输入原始 CT 文件不存在: {input_ct_path}")

        ct_path = Path(input_ct_path).resolve()
        case_name = ct_path.name.replace(".nii.gz", "").replace(".nii", "")
        pred_mask_path = output_dir / f"{case_name}_pred_init.nii.gz"

        print(f"\n[阶段 1] 执行深度学习辅助分割 (nnU-Net 预训练模型)...")
        predictor = SpinePredictor()
        current_mask = Path(predictor.predict(
            input_ct_path=ct_path,
            output_mask_path=pred_mask_path,
            fast_mode=fast_mode
        ))

    # ---------------------------------------------------------
    # 步骤 2: 终板、椎弓根及椎管关键解剖质检切片生成
    # ---------------------------------------------------------
    if input_ct_path and os.path.exists(input_ct_path):
        print(f"\n[阶段 2] 自动生成正交多平面 (MPR) 临床质检切片图...")
        qa_png_path = output_dir / f"{prefix}_clinical_qa_review.png"
        generate_qa_report(
            ct_image_path=input_ct_path,
            mask_path=current_mask,
            output_png_path=qa_png_path
        )

    # ---------------------------------------------------------
    # 步骤 3: 人工核查与修正 (ITK-SNAP 交互式界面)
    # ---------------------------------------------------------
    if interactive_check and input_ct_path and os.path.exists(input_ct_path):
        print(f"\n[阶段 3] 唤起 ITK-SNAP 进行临床逐例人工核查与修正...")
        launch_itksnap_session(
            ct_image_path=input_ct_path,
            segmentation_path=current_mask
        )
        print("\n提示: 如果在 ITK-SNAP 中对掩模进行了修正并另存，")
        print("可在修正完成后重新使用 --mask_input 指定修正后的掩模提取最终 STL。")

    # ---------------------------------------------------------
    # 步骤 4: 移动立方体 (Marching Cubes) 提取等值面，输出 STL
    # ---------------------------------------------------------
    print(f"\n[阶段 4] 确定分割范围，采用移动立方体 (Marching Cubes) 提取等值面...")
    stl_output_dir = output_dir / "stl_models"
    exporter = MarchingCubesSTLExporter(sigma=sigma, level=level)
    
    results = exporter.export_from_nifti(
        mask_nifti_path=current_mask,
        output_dir=stl_output_dir,
        scope=scope,
        target_labels=target_labels,
        prefix=prefix
    )

    print("\n" + "="*75)
    print("[SUCCESS] 任务全部圆满完成！输出清单汇总:")
    print("="*75)
    if "COMBINED" in results["mesh_stats"]:
        c_stat = results["mesh_stats"]["COMBINED"]
        print(f" [-] 整体融合 STL: {c_stat['stl_path']}")
        print(f"     - 三角面片总数: {c_stat['faces']} | 是否水密闭合: {c_stat['is_watertight']}")
    
    ind_count = len([k for k in results["mesh_stats"].keys() if k != "COMBINED"])
    if ind_count > 0:
        print(f" [-] 独立椎体 STL: 共生成 {ind_count} 个独立椎骨 STL (位于 {stl_output_dir})")
        print("     可用在手术规划软件中单独测量、设计椎弓根螺钉置钉通道或进行 3D 打印！")
    print("="*75 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Spinal Vertebrae Segmentation and Marching Cubes STL Mesh Generator")
    parser.add_argument("--input", "-i", type=str, default=None, help="输入原始 CT 图像路径 (.nii 或 .nii.gz)")
    parser.add_argument("--output_dir", "-o", type=str, default="output_results", help="结果输出根目录 (默认: output_results)")
    parser.add_argument("--mask_input", "-m", type=str, default=None, help="可选: 直接传入已有的/已修正的分割掩模路径，跳过深度学习推理")
    parser.add_argument("--scope", "-s", type=str, choices=["all", "individual", "both"], default="both", help="STL 输出范围: individual(各个独立椎骨), all(整体合并), both(两者均输出)")
    parser.add_argument("--labels", "-l", type=str, default=None, help="指定导出的椎骨编号或名称，逗号隔开 (例如: L1,L2,L3 或 20,21,22)")
    parser.add_argument("--clinical_mode", action="store_true", help="启用全精度临床模式 (开启 8x TTA 镜像推理，耗时稍长但分割精度最高)")
    parser.add_argument("--interactive_check", action="store_true", help="自动唤起 ITK-SNAP 打开 CT 与预测掩模进行交互式核查修正")
    parser.add_argument("--sigma", type=float, default=0.8, help="Marching Cubes 前置高斯抗锯齿平滑因子 (默认 0.8，消除体素台阶伪影)")
    parser.add_argument("--level", type=float, default=0.5, help="等值面提取阈值 (默认 0.5)")
    parser.add_argument("--prefix", type=str, default="patient", help="输出文件名前缀 (默认: patient)")

    args = parser.parse_args()

    # Parse labels list if provided
    target_labels = None
    if args.labels:
        target_labels = [item.strip() for item in args.labels.split(",") if item.strip()]

    # If no input provided, default to sample data in CTSpine1K
    input_ct = args.input
    mask_in = args.mask_input

    if not input_ct and not mask_in:
        default_sample = Path("CTSpine1K/data/colon/1.3.6.1.4.1.9328.50.4.0001.nii.gz")
        default_label = Path("CTSpine1K/label/conlon/1.3.6.1.4.1.9328.50.4.0001_seg.nii.gz")
        if default_sample.exists():
            print(f"[*] 未指定输入，自动选用 CTSpine1K 测试样例: {default_sample}")
            input_ct = str(default_sample)
            if default_label.exists():
                print(f"[*] 发现对应金标准标注，作为验证对比: {default_label}")

    run_pipeline(
        input_ct_path=input_ct,
        output_dir=args.output_dir,
        mask_input=mask_in,
        scope=args.scope,
        target_labels=target_labels,
        fast_mode=not args.clinical_mode,
        interactive_check=args.interactive_check,
        sigma=args.sigma,
        level=args.level,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
