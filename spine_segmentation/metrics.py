"""
Clinical Segmentation, CT Image Quality, and 3D Surface Reconstruction Metrics
Provides:
  1. Reference-free metrics (CT Noise/CNR, Metal Artifact Check, Mesh Volume Fidelity)
  2. Reference-based metrics against Ground Truth (DSC, IoU, Precision, Recall,
     Geometric MAE [ASSD], Geometric RMSE [RMSD], HD95, RVD)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import cKDTree

from spine_segmentation.config import VERTEBRAE_LABELS


def _extract_surface_world_coords(binary_mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Extracts 3D surface voxel coordinates and maps them to physical space (mm)."""
    if not np.any(binary_mask):
        return np.empty((0, 3), dtype=np.float64)
    # 6-connectivity surface extraction
    surface = binary_mask & (~binary_erosion(binary_mask, border_value=0))
    ijk = np.argwhere(surface).astype(np.float64)
    if len(ijk) == 0:
        return np.empty((0, 3), dtype=np.float64)
    homo = np.hstack([ijk, np.ones((len(ijk), 1), dtype=np.float64)])
    return (affine @ homo.T).T[:, :3]


def compute_binary_seg_and_surface_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    affine: np.ndarray
) -> Dict[str, float]:
    """
    Computes volumetric overlap and 3D physical surface reconstruction metrics (in mm)
    for a pair of binary 3D masks.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))

    pred_sum = tp + fp
    gt_sum = tp + fn

    if pred_sum == 0 and gt_sum == 0:
        return {
            "dsc": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0,
            "rvd_pct": 0.0, "assd_mae_mm": 0.0, "rmsd_rmse_mm": 0.0, "hd95_mm": 0.0
        }

    dsc = (2.0 * tp) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / pred_sum if pred_sum > 0 else 0.0
    recall = tp / gt_sum if gt_sum > 0 else 0.0
    rvd_pct = ((pred_sum - gt_sum) / gt_sum * 100.0) if gt_sum > 0 else float("nan")

    # 3D Surface Distance Metrics (Geometric MAE / RMSE / HD95)
    if pred_sum == 0 or gt_sum == 0:
        assd = float("nan")
        rmsd = float("nan")
        hd95 = float("nan")
    else:
        pts_pred = _extract_surface_world_coords(pred, affine)
        pts_gt = _extract_surface_world_coords(gt, affine)

        tree_gt = cKDTree(pts_gt)
        tree_pred = cKDTree(pts_pred)

        d_p2g, _ = tree_gt.query(pts_pred, k=1)
        d_g2p, _ = tree_pred.query(pts_gt, k=1)

        all_d = np.concatenate([d_p2g, d_g2p])
        assd = float(np.mean(all_d))                              # Geometric MAE (mm)
        rmsd = float(np.sqrt(np.mean(all_d ** 2)))                # Geometric RMSE (mm)
        hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

    return {
        "dsc": dsc,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "rvd_pct": rvd_pct,
        "assd_mae_mm": assd,
        "rmsd_rmse_mm": rmsd,
        "hd95_mm": hd95,
    }


def evaluate_ct_image_quality(
    ct_path: Union[str, Path],
    mask_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Evaluates raw CT image quality around the segmented spine (No GT required):
      - Bone mean HU & std
      - Paraspinal soft tissue noise sigma (HU)
      - Bone-to-Tissue Contrast-to-Noise Ratio (CNR)
      - Metal artifact warning (> 2500 HU near spine)
    """
    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))

    ct_vol = np.asanyarray(ct_img.dataobj).astype(np.float32)
    mask_vol = np.asanyarray(mask_img.dataobj) > 0

    if not np.any(mask_vol):
        return {"valid": False}

    # Restrict analysis to tight bounding box around spine (+15 voxels margin)
    coords = np.argwhere(mask_vol)
    mins = np.maximum(coords.min(axis=0) - 15, 0)
    maxs = np.minimum(coords.max(axis=0) + 15, ct_vol.shape)

    ct_roi = ct_vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    mask_roi = mask_vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    bone_vals = ct_roi[mask_roi]
    bone_mean = float(np.mean(bone_vals))
    bone_std = float(np.std(bone_vals))

    # Paraspinal soft-tissue shell (avoid air & CT FOV constant padding)
    shell = binary_dilation(mask_roi, iterations=8) & (~binary_dilation(mask_roi, iterations=2))
    tissue_vals = ct_roi[shell & (ct_roi > -100.0) & (ct_roi < 150.0)]

    if len(tissue_vals) > 100:
        tissue_mean = float(np.mean(tissue_vals))
        tissue_noise_std = float(np.std(tissue_vals))
        cnr = abs(bone_mean - tissue_mean) / (tissue_noise_std + 1e-6)
    else:
        tissue_mean = float("nan")
        tissue_noise_std = float("nan")
        cnr = float("nan")

    # Metal implant / severe artifact check (HU > 2500)
    metal_voxels = int(np.sum(ct_roi > 2500.0))
    metal_ratio_pct = metal_voxels / float(ct_roi.size) * 100.0

    return {
        "valid": True,
        "spacing_mm": tuple(round(float(z), 3) for z in ct_img.header.get_zooms()[:3]),
        "bone_mean_hu": round(bone_mean, 1),
        "bone_std_hu": round(bone_std, 1),
        "tissue_noise_std_hu": round(tissue_noise_std, 1),
        "bone_tissue_cnr": round(cnr, 2),
        "metal_voxels": metal_voxels,
        "metal_ratio_pct": round(metal_ratio_pct, 4),
        "metal_warning": metal_voxels > 50,
    }


def evaluate_against_ground_truth(
    pred_mask_path: Union[str, Path],
    gt_mask_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Full multi-class (per-vertebra) and binary (whole spine) evaluation against Ground Truth,
    aligned with CTSpine1K benchmark standards + geometric MAE/RMSE.
    """
    pred_img = nib.load(str(pred_mask_path))
    gt_img = nib.load(str(gt_mask_path))

    pred_vol = np.asanyarray(pred_img.dataobj).astype(np.int32)
    gt_vol = np.asanyarray(gt_img.dataobj).astype(np.int32)

    if pred_vol.shape != gt_vol.shape:
        raise ValueError(f"Shape mismatch: pred {pred_vol.shape} vs gt {gt_vol.shape}")

    affine = gt_img.affine
    all_labels = sorted([int(l) for l in np.union1d(np.unique(pred_vol), np.unique(gt_vol)) if l > 0])

    per_vertebra = {}
    for lbl in all_labels:
        v_name = VERTEBRAE_LABELS.get(lbl, {}).get("name", f"L_{lbl}")
        m = compute_binary_seg_and_surface_metrics(pred_vol == lbl, gt_vol == lbl, affine)
        per_vertebra[v_name] = {"label_id": lbl, **m}

    # Aggregate Mean of Vertebrae (multi-class)
    keys = ["dsc", "iou", "precision", "recall", "rvd_pct", "assd_mae_mm", "rmsd_rmse_mm", "hd95_mm"]
    mean_vertebrae = {
        k: float(np.nanmean([v[k] for v in per_vertebra.values()])) if per_vertebra else float("nan")
        for k in keys
    }

    # Whole Spine (binary)
    whole_spine = compute_binary_seg_and_surface_metrics(pred_vol > 0, gt_vol > 0, affine)

    return {
        "per_vertebra": per_vertebra,
        "mean_vertebrae": mean_vertebrae,
        "whole_spine": whole_spine,
    }


def print_metrics_report(
    ct_quality: Optional[Dict[str, Any]] = None,
    inference_stats: Optional[Dict[str, float]] = None,
    gt_metrics: Optional[Dict[str, Any]] = None,
    mesh_stats: Optional[Dict[str, Any]] = None
) -> None:
    """Prints a clean, comprehensive clinical & geometric evaluation table to console."""
    print("\n" + "=" * 88)
    print("  📊 脊柱 CT 分割与 3D 表面重建定量评估报告")
    print("=" * 88)

    # 1. Input CT Quality & Noise
    if ct_quality and ct_quality.get("valid"):
        print("\n[1] 输入 CT 图像质量与噪声分析 (无监督前置质控)")
        print(f"    - 体素分辨率 (Spacing):       {ct_quality['spacing_mm']} mm")
        print(f"    - 椎骨区域平均密度 (HU):      {ct_quality['bone_mean_hu']} ± {ct_quality['bone_std_hu']} HU")
        print(f"    - 椎旁软组织噪声 (Noise σ):   {ct_quality['tissue_noise_std_hu']} HU")
        print(f"    - 骨-软组织对比噪声比 (CNR):  {ct_quality['bone_tissue_cnr']}")
        flag = "⚠️ 检测到高密度金属/内固定伪影，请务必在 ITK-SNAP 核查边界！" if ct_quality["metal_warning"] else "正常 (未见显著金属伪影)"
        print(f"    - 高密度伪影监测 (>2500 HU):  {ct_quality['metal_voxels']} voxels ({ct_quality['metal_ratio_pct']}%) -> {flag}")

    # 2. Inference Uncertainty
    if inference_stats:
        print("\n[2] 深度学习推理置信度与不确定度 (基于 nnU-Net Softmax)")
        print(f"    - 前景平均置信度 (Mean Conf): {inference_stats['mean_fg_confidence']:.4f} (越接近 1.0 越确信)")
        print(f"    - 前景平均预测熵 (Entropy):   {inference_stats['mean_fg_entropy']:.4f} nats (越低越稳定)")
        print(f"    - 低置信体素占比 (<0.80):     {inference_stats['low_conf_voxel_pct']:.2f}%")

    # 3. Ground Truth Metrics (Segmentation + 3D Surface MAE/RMSE/HD95)
    if gt_metrics:
        print("\n[3] 分割与 3D 表面重建定量精度 (对比 Ground Truth)")
        header = (
            f"    {'椎骨':<8} | {'DSC':>6} | {'IoU':>6} | {'Prec':>6} | {'Rec':>6} | "
            f"{'MAE(ASSD)':>9} | {'RMSE(RMSD)':>10} | {'HD95(mm)':>8} | {'RVD(%)':>7}"
        )
        print(header)
        print("    " + "-" * (len(header) - 4))
        for v_name, m in gt_metrics["per_vertebra"].items():
            print(
                f"    {v_name:<8} | {m['dsc']:>6.4f} | {m['iou']:>6.4f} | "
                f"{m['precision']:>6.4f} | {m['recall']:>6.4f} | "
                f"{m['assd_mae_mm']:>9.3f} | {m['rmsd_rmse_mm']:>10.3f} | "
                f"{m['hd95_mm']:>8.3f} | {m['rvd_pct']:>+7.2f}"
            )
        print("    " + "-" * (len(header) - 4))
        mv = gt_metrics["mean_vertebrae"]
        ws = gt_metrics["whole_spine"]
        print(
            f"    {'Mean(多类)':<6} | {mv['dsc']:>6.4f} | {mv['iou']:>6.4f} | "
            f"{mv['precision']:>6.4f} | {mv['recall']:>6.4f} | "
            f"{mv['assd_mae_mm']:>9.3f} | {mv['rmsd_rmse_mm']:>10.3f} | "
            f"{mv['hd95_mm']:>8.3f} | {mv['rvd_pct']:>+7.2f}"
        )
        print(
            f"    {'Spine(二值)':<5} | {ws['dsc']:>6.4f} | {ws['iou']:>6.4f} | "
            f"{ws['precision']:>6.4f} | {ws['recall']:>6.4f} | "
            f"{ws['assd_mae_mm']:>9.3f} | {ws['rmsd_rmse_mm']:>10.3f} | "
            f"{ws['hd95_mm']:>8.3f} | {ws['rvd_pct']:>+7.2f}"
        )

    # 4. Marching Cubes STL Fidelity
    if mesh_stats:
        print("\n[4] Marching Cubes STL 网格重建保真度 (体素体积 vs. 平滑网格体积)")
        print(f"    {'模型名称':<10} | {'面片数':>8} | {'水密闭合':<6} | {'体素体积(cm3)':>13} | {'网格体积(cm3)':>13} | {'体积畸变ΔV%':>11}")
        print("    " + "-" * 76)
        for name, st in mesh_stats.items():
            vox_v = st.get("voxel_volume_cm3", -1)
            msh_v = st.get("volume_cm3", -1)
            dv = st.get("volume_diff_pct", float("nan"))
            print(
                f"    {name:<10} | {st['faces']:>8d} | {str(st['is_watertight']):<8} | "
                f"{vox_v:>13.2f} | {msh_v:>13.2f} | {dv:>+10.2f}%"
            )

    print("=" * 88 + "\n")
