"""
Marching Cubes Isosurface Extraction and STL Triangle Mesh Exporter
Converts spinal vertebrae segmentation masks into high-precision, watertight,
and coordinate-aligned STL models for 3D printing and surgical planning CAD.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, label as nd_label
from skimage.measure import marching_cubes
import trimesh

from spine_segmentation.config import (
    VERTEBRAE_LABELS,
    NAME_TO_LABEL,
    MARCHING_CUBES_DEFAULTS
)


def transform_vertices_to_world(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Transforms marching cubes voxel-space vertices (i, j, k) to patient world RAS physical space (x, y, z) in mm.
    """
    homo_coords = np.hstack([vertices, np.ones((len(vertices), 1), dtype=vertices.dtype)])
    world_coords = (affine @ homo_coords.T).T[:, :3]
    return world_coords


def filter_largest_component(binary_mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """
    Filters out spurious floating voxels, keeping the largest connected component.
    """
    labeled_array, num_features = nd_label(binary_mask)
    if num_features <= 1:
        return binary_mask

    counts = np.bincount(labeled_array.ravel())
    # 0 is background
    counts[0] = 0
    largest_label = counts.argmax()

    cleaned_mask = (labeled_array == largest_label)
    return cleaned_mask


def extract_vertebra_mesh(
    binary_mask: np.ndarray,
    affine: np.ndarray,
    sigma: float = 0.8,
    level: float = 0.5,
    keep_largest: bool = True,
    min_size: int = 200,
    taubin_iter: int = 0
) -> Optional[trimesh.Trimesh]:
    """
    Extracts an isosurface triangular mesh for a binary vertebrae mask using Marching Cubes.
    
    Args:
        binary_mask: 3D boolean/uint8 numpy array representing a single vertebra or spine segment.
        affine: 4x4 NIfTI affine matrix mapping voxel indices to world coordinates (mm).
        sigma: Standard deviation for Gaussian smoothing prior to marching cubes.
        level: Isosurface contour value (0.5 for smoothed binary).
        keep_largest: Whether to discard isolated small disconnected fragments.
        min_size: Minimum voxel count for a valid component.
        taubin_iter: Number of Taubin mesh smoothing iterations.

    Returns:
        A cleaned, watertight trimesh.Trimesh object, or None if mask is empty.
    """
    voxel_count = np.sum(binary_mask)
    if voxel_count < min_size:
        return None

    if keep_largest:
        binary_mask = filter_largest_component(binary_mask, min_size=min_size)

    # Anti-aliasing Gaussian smoothing to avoid staircase artifacts while preserving volume
    if sigma > 0:
        vol = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
    else:
        vol = binary_mask.astype(np.float32)

    # Marching Cubes Isosurface Extraction
    try:
        verts, faces, normals, _ = marching_cubes(
            vol,
            level=level,
            gradient_direction='descent',
            step_size=1
        )
    except (ValueError, RuntimeError) as e:
        print(f"[!] Marching Cubes extraction failed: {e}")
        return None

    if len(verts) == 0 or len(faces) == 0:
        return None

    # Transform vertices from voxel grid (i, j, k) to patient physical coordinates (mm)
    verts_world = transform_vertices_to_world(verts, affine)

    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        process=True
    )

    # Topology and normal orientation cleanup
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    # Optional Taubin smoothing (non-shrinking mesh surface filter)
    if taubin_iter > 0:
        try:
            trimesh.smoothing.filter_taubin(
                mesh,
                lamb=0.5,
                nu=-0.53,
                iterations=taubin_iter
            )
            trimesh.repair.fix_normals(mesh)
        except Exception as e:
            pass

    return mesh


class MarchingCubesSTLExporter:
    """
    High-level manager for spinal vertebrae mesh generation and STL export.
    """
    def __init__(
        self,
        sigma: float = MARCHING_CUBES_DEFAULTS["sigma"],
        level: float = MARCHING_CUBES_DEFAULTS["level"],
        keep_largest: bool = MARCHING_CUBES_DEFAULTS["keep_largest_component"],
        min_size: int = MARCHING_CUBES_DEFAULTS["min_component_voxels"],
        taubin_iter: int = MARCHING_CUBES_DEFAULTS["taubin_iterations"]
    ):
        self.sigma = sigma
        self.level = level
        self.keep_largest = keep_largest
        self.min_size = min_size
        self.taubin_iter = taubin_iter

    def export_from_nifti(
        self,
        mask_nifti_path: Union[str, Path],
        output_dir: Union[str, Path],
        scope: str = "both",
        target_labels: Optional[List[Union[int, str]]] = None,
        prefix: str = "spine"
    ) -> Dict[str, Any]:
        """
        Extracts STL meshes from a multi-class or binary spinal vertebrae segmentation mask.

        Args:
            mask_nifti_path: Path to the segmentation mask (.nii / .nii.gz).
            output_dir: Directory to save generated STL files.
            scope: Export mode:
                   - 'individual': Export separate STL files for each vertebra (e.g. L1.stl, L2.stl).
                   - 'all': Export a single combined STL file for the entire spinal segment.
                   - 'both': Export both individual STL files and the combined whole spine STL.
            target_labels: Optional list of label IDs (e.g. [20, 21, 22]) or names (e.g. ['L1', 'L2']).
            prefix: Prefix for output filenames.

        Returns:
            Dictionary containing paths to generated STL files and mesh statistics.
        """
        mask_path = Path(mask_nifti_path).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        print(f"\n{'='*70}")
        print(f"[*] Starting Marching Cubes Surface Mesh Extraction")
        print(f"    - Input Mask:  {mask_path.name}")
        print(f"    - Output Dir:  {out_dir}")
        print(f"    - Export Scope: {scope}")
        print(f"    - Anti-aliasing Sigma: {self.sigma}")
        print(f"{'='*70}")

        t0 = time.time()
        img = nib.load(str(mask_path))
        data = np.asanyarray(img.dataobj)
        affine = img.affine
        voxel_spacing = img.header.get_zooms()

        print(f"[*] Mask Dimensions: {data.shape}, Spacing: {voxel_spacing[:3]} mm")

        # Discover unique labels present in the mask (excluding background 0)
        unique_labels = sorted([int(l) for l in np.unique(data) if l > 0])
        print(f"[*] Discovered {len(unique_labels)} vertebrae labels in mask: {unique_labels}")

        # Filter target labels if specified
        if target_labels is not None:
            resolved_labels = []
            for t in target_labels:
                if isinstance(t, str):
                    t_upper = t.strip().upper()
                    if t_upper in NAME_TO_LABEL:
                        resolved_labels.append(NAME_TO_LABEL[t_upper])
                    else:
                        print(f"[!] Warning: Unknown vertebra name '{t}'")
                elif isinstance(t, (int, np.integer)):
                    resolved_labels.append(int(t))
            active_labels = [l for l in unique_labels if l in resolved_labels]
        else:
            active_labels = unique_labels

        if not active_labels:
            print("[!] No matching vertebrae labels found to export.")
            return {"exported_files": [], "meshes": {}}

        exported_files = []
        mesh_stats = {}
        individual_meshes = {}

        # 1. Process individual vertebrae
        if scope in ["individual", "both"]:
            print("\n[+] Processing Individual Vertebrae...")
            for label_id in active_labels:
                v_info = VERTEBRAE_LABELS.get(label_id, {"name": f"Vertebra_{label_id}", "chinese": f"椎骨_{label_id}"})
                v_name = v_info["name"]
                v_cn = v_info["chinese"]

                bin_mask = (data == label_id)
                v_voxels = np.sum(bin_mask)
                if v_voxels < self.min_size:
                    continue

                mesh = extract_vertebra_mesh(
                    bin_mask,
                    affine,
                    sigma=self.sigma,
                    level=self.level,
                    keep_largest=self.keep_largest,
                    min_size=self.min_size,
                    taubin_iter=self.taubin_iter
                )

                if mesh is not None and len(mesh.faces) > 0:
                    individual_meshes[label_id] = mesh
                    stl_filename = f"{prefix}_{v_name}.stl"
                    stl_path = out_dir / stl_filename
                    mesh.export(str(stl_path))
                    exported_files.append(str(stl_path))

                    vol_cm3 = abs(mesh.volume) / 1000.0 if mesh.is_watertight else -1
                    mesh_stats[v_name] = {
                        "label_id": label_id,
                        "chinese_name": v_cn,
                        "stl_path": str(stl_path),
                        "vertices": len(mesh.vertices),
                        "faces": len(mesh.faces),
                        "is_watertight": mesh.is_watertight,
                        "volume_cm3": round(vol_cm3, 2),
                        "bounds_mm": [np.round(mesh.bounds[0], 2).tolist(), np.round(mesh.bounds[1], 2).tolist()]
                    }
                    print(f"    [OK] [{v_name:^6}] ({v_cn:^10}) -> Faces: {len(mesh.faces):>6}, Watertight: {str(mesh.is_watertight):<5}, Vol: {vol_cm3:>6.2f} cm3 -> {stl_filename}")

        # 2. Process combined whole spine segment
        if scope in ["all", "both"]:
            print("\n[+] Processing Combined Spine Segment Mesh...")
            combined_mask = np.isin(data, active_labels)
            
            combined_mesh = extract_vertebra_mesh(
                combined_mask,
                affine,
                sigma=self.sigma,
                level=self.level,
                keep_largest=False,  # Keep all vertebrae in the segment
                min_size=self.min_size,
                taubin_iter=self.taubin_iter
            )

            if combined_mesh is not None and len(combined_mesh.faces) > 0:
                combined_stl_filename = f"{prefix}_combined_spine.stl"
                combined_stl_path = out_dir / combined_stl_filename
                combined_mesh.export(str(combined_stl_path))
                exported_files.append(str(combined_stl_path))

                vol_cm3 = abs(combined_mesh.volume) / 1000.0 if combined_mesh.is_watertight else -1
                mesh_stats["COMBINED"] = {
                    "stl_path": str(combined_stl_path),
                    "vertices": len(combined_mesh.vertices),
                    "faces": len(combined_mesh.faces),
                    "is_watertight": combined_mesh.is_watertight,
                    "volume_cm3": round(vol_cm3, 2),
                    "bounds_mm": [np.round(combined_mesh.bounds[0], 2).tolist(), np.round(combined_mesh.bounds[1], 2).tolist()]
                }
                print(f"    [OK] [COMBINED SPINE] -> Vertices: {len(combined_mesh.vertices)}, Faces: {len(combined_mesh.faces)}, Saved to: {combined_stl_filename}")

        total_elapsed = time.time() - t0
        print(f"\n[OK] STL Mesh Extraction Completed in {total_elapsed:.2f}s")
        print(f"    Total STL Files Exported: {len(exported_files)}")
        print(f"    Target Directory: {out_dir}")
        print(f"{'='*70}\n")

        return {
            "exported_files": exported_files,
            "mesh_stats": mesh_stats,
            "output_dir": str(out_dir)
        }
