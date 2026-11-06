"""
Quality Assessment (QA) Multi-Planar Slice Visualizer
Generates multi-planar orthogonal inspection slices (Axial, Sagittal, Coronal)
with CT bone-window settings and segmentation contour overlays.
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import center_of_mass

from spine_segmentation.config import VERTEBRAE_LABELS

# ----------------- 添加以下两行配置以支持中文显示 -----------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 优先调用黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False                                      # 保证坐标轴负号正常显示
# ------------------------------------------------------------------


def apply_bone_window(ct_data: np.ndarray, wl: float = 350.0, ww: float = 1800.0) -> np.ndarray:
    """
    Applies clinical spine bone windowing to CT Hounsfield Units (HU).
    Window Level = 350, Window Width = 1800 (standard for cortical/trabecular bone).
    """
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    clipped = np.clip(ct_data, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return normalized


def generate_qa_report(
    ct_image_path: Union[str, Path],
    mask_path: Union[str, Path],
    output_png_path: Union[str, Path],
    target_vertebra_label: Optional[int] = None
) -> str:
    """
    Generates a multi-planar QA visual report for clinical review of segmentation boundaries.
    Focuses on: Endplates (Sagittal), Pedicles & Spinal Canal (Axial), and Frontal profile (Coronal).
    """
    ct_path = Path(ct_image_path).resolve()
    mask_path = Path(mask_path).resolve()
    output_png_path = Path(output_png_path).resolve()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))

    ct_vol = np.asanyarray(ct_img.dataobj).astype(np.float32)
    mask_vol = np.asanyarray(mask_img.dataobj).astype(np.int32)

    unique_labels = sorted([int(l) for l in np.unique(mask_vol) if l > 0])
    if not unique_labels:
        print("[!] No vertebrae segmented to visualize in QA report.")
        return ""

    # Choose vertebra of interest for centering
    if target_vertebra_label and target_vertebra_label in unique_labels:
        focus_label = target_vertebra_label
    else:
        # Default to middle vertebra (e.g. L1/T12)
        focus_label = unique_labels[len(unique_labels) // 2]

    v_info = VERTEBRAE_LABELS.get(focus_label, {"name": f"Label_{focus_label}", "chinese": f"节段_{focus_label}"})
    v_name = v_info["name"]

    # Calculate center of mass of the target vertebra
    focus_mask = (mask_vol == focus_label)
    cy, cx, cz = center_of_mass(focus_mask)
    cy, cx, cz = int(round(cy)), int(round(cx)), int(round(cz))

    # Apply bone window
    ct_norm = apply_bone_window(ct_vol, wl=350, ww=1800)

    # Prepare figure with 3 orthogonal views
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=150)
    fig.patch.set_facecolor('#1a1a1a')

    # Color map for segmentation overlay
    cmap = plt.cm.get_cmap("tab20", 25)

    # 1. Axial slice (Pedicles, Canal, Body)
    ax = axes[0]
    ax.set_facecolor('black')
    ct_slice_ax = np.rot90(ct_norm[:, :, cz])
    mask_slice_ax = np.rot90(mask_vol[:, :, cz])
    
    ax.imshow(ct_slice_ax, cmap='gray', vmin=0, vmax=1)
    masked_ax = np.ma.masked_where(mask_slice_ax == 0, mask_slice_ax)
    ax.imshow(masked_ax, cmap=cmap, alpha=0.55, vmin=1, vmax=25)
    ax.set_title(f"Axial View (Z={cz})\n[核查: 椎弓根、椎管及骨折后缘]", color='white', fontsize=12, pad=10)
    ax.axis('off')

    # 2. Sagittal slice (Endplates, Height, Alignment)
    ax = axes[1]
    ax.set_facecolor('black')
    ct_slice_sag = np.rot90(ct_norm[cy, :, :])
    mask_slice_sag = np.rot90(mask_vol[cy, :, :])
    
    ax.imshow(ct_slice_sag, cmap='gray', vmin=0, vmax=1)
    masked_sag = np.ma.masked_where(mask_slice_sag == 0, mask_slice_sag)
    ax.imshow(masked_sag, cmap=cmap, alpha=0.55, vmin=1, vmax=25)
    ax.set_title(f"Sagittal View (Y={cy})\n[核查: 上下终板、椎体高度及滑脱]", color='white', fontsize=12, pad=10)
    ax.axis('off')

    # 3. Coronal slice (Scoliosis, Transverse Process)
    ax = axes[2]
    ax.set_facecolor('black')
    ct_slice_cor = np.rot90(ct_norm[:, cx, :])
    mask_slice_cor = np.rot90(mask_vol[:, cx, :])
    
    ax.imshow(ct_slice_cor, cmap='gray', vmin=0, vmax=1)
    masked_cor = np.ma.masked_where(mask_slice_cor == 0, mask_slice_cor)
    ax.imshow(masked_cor, cmap=cmap, alpha=0.55, vmin=1, vmax=25)
    ax.set_title(f"Coronal View (X={cx})\n[核查: 左右对称性、横突与小关节]", color='white', fontsize=12, pad=10)
    ax.axis('off')

    plt.suptitle(
        f"Spine Vertebrae Clinical Segmentation QA Review (Focus: {v_name})\n"
        f"CT Bone Window (WL=350, WW=1800) | Labels Present: {', '.join([VERTEBRAE_LABELS.get(l, {}).get('name', str(l)) for l in unique_labels])}",
        color='#00ffcc',
        fontsize=14,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.savefig(str(output_png_path), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"[OK] Clinical QA review slice report generated: {output_png_path}")
    return str(output_png_path)
