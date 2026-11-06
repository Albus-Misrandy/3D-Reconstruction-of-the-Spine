"""
ITK-SNAP Integration Helper for Clinical Inspection and Manual Refinement
Generates custom anatomical label description files and launches ITK-SNAP
for interactive slice-by-slice review and manual contour correction.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union
from spine_segmentation.config import (
    VERTEBRAE_LABELS,
    find_itksnap_path
)


def generate_itksnap_label_description(output_txt_path: Union[str, Path]) -> str:
    """
    Generates an ITK-SNAP compatible label description file with vertebrae anatomical names and colors.
    Format:
    IDX   R   G   B   A   VIS  MSH  LABEL
    """
    output_txt_path = Path(output_txt_path).resolve()
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "################################################",
        "# ITK-SNAP Label Description File",
        "# Spinal Vertebrae Anatomical Annotations",
        "################################################",
        "# IDX   R     G     B     A   VIS MSH  LABEL",
        "    0     0     0     0     0   0   0    \"Clear Label\""
    ]

    for label_id, info in VERTEBRAE_LABELS.items():
        name = info["name"]
        cn = info["chinese"]
        r, g, b = info["color"]
        # Format: IDX R G B A VIS MSH "LABEL"
        line = f"{label_id:5d} {r:5d} {g:5d} {b:5d}     1   1   1    \"{name} - {cn}\""
        lines.append(line)

    content = "\n".join(lines) + "\n"
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_txt_path)


def launch_itksnap_session(
    ct_image_path: Union[str, Path],
    segmentation_path: Union[str, Path],
    label_desc_path: Optional[Union[str, Path]] = None,
    itksnap_exe: Optional[Union[str, Path]] = None
) -> bool:
    """
    Launches ITK-SNAP with raw CT as greyscale volume and segmentation mask as overlay.

    Args:
        ct_image_path: Path to raw CT scan (.nii / .nii.gz).
        segmentation_path: Path to initial segmentation mask.
        label_desc_path: Optional path to label description file (.txt).
        itksnap_exe: Optional explicit path to ITK-SNAP.exe.

    Returns:
        True if successfully launched, False otherwise.
    """
    exe_path = itksnap_exe or find_itksnap_path()
    if not exe_path or not os.path.exists(exe_path):
        print(f"[!] ITK-SNAP executable not found. Looked in default locations.")
        print(f"    Please manually open ITK-SNAP and load:\n      - Grey: {ct_image_path}\n      - Seg:  {segmentation_path}")
        return False

    ct_image_path = Path(ct_image_path).resolve()
    segmentation_path = Path(segmentation_path).resolve()

    if label_desc_path is None:
        label_desc_path = segmentation_path.parent / "itksnap_labels.txt"
        generate_itksnap_label_description(label_desc_path)

    cmd = [
        str(exe_path),
        "-g", str(ct_image_path),
        "-s", str(segmentation_path),
        "-l", str(label_desc_path)
    ]

    print("\n" + "="*70)
    print("📋 [临床人工核查与修正工作流指引 - ITK-SNAP]")
    print("="*70)
    print("正在启动 ITK-SNAP 交互式界面，加载原始 CT 与初始椎骨轮廓...")
    print("\n🔍 重点核查与人工修正要点（依据临床规范）：")
    print(" 1. 终板区域 (Endplates): 仔细检查矢状位上下终板形态，清除软骨下增生伪影；")
    print(" 2. 椎弓根区域 (Pedicles): 检查内外侧骨皮质完整度，确保椎弓根螺钉通道边界精确；")
    print(" 3. 椎管临近区域 (Spinal Canal): 确认椎体后缘与硬膜囊界限，勿将韧带钙化误判；")
    print(" 4. 骨折后缘 (Fracture Edges): 爆裂性骨折后突进入椎管的骨块边界必须精准包含；")
    print(" 5. 金属内固定伪影 (Metal Artifacts): 修正金属放射状伪影导致的过分割或缺失。")
    print("\n💾 修正完成后：")
    print(" 在 ITK-SNAP 中点击菜单: Segmentation -> Save Segmentation Image (Ctrl+S)")
    print(f" 建议保存为: {segmentation_path.parent / (segmentation_path.stem.replace('.nii', '') + '_refined.nii.gz')}")
    print("="*70 + "\n")

    try:
        subprocess.Popen(cmd)
        print(f"[OK] ITK-SNAP launched successfully with command: {' '.join(cmd)}")
        return True
    except Exception as e:
        print(f"[!] Failed to launch ITK-SNAP: {e}")
        return False
