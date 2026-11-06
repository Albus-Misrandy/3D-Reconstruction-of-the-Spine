"""
nnU-Net Inference Predictor for Spinal Vertebrae Segmentation
Wraps the CTSpine1K pre-trained nnU-Net model, handles PyTorch 2.6+ compatibility,
and executes 3D sliding-window segmentation on input CT NIfTI scans.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

# Safe monkey-patch for PyTorch 2.6+ weights_only default behavior
import torch
_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Suppress harmless warnings and initialize nnUNet
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from nnunet.training.model_restore import restore_model
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from spine_segmentation.config import (
    DEFAULT_CHECKPOINT_MODEL,
    DEFAULT_CHECKPOINT_PKL,
    DEFAULT_PLANS_PKL
)


class SpinePredictor:
    """
    Spine vertebrae segmentation predictor using pre-trained nnU-Net.
    """
    def __init__(
        self,
        model_pkl: Optional[Union[str, Path]] = None,
        checkpoint_model: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_pkl = Path(model_pkl or DEFAULT_CHECKPOINT_PKL)
        self.checkpoint_model = Path(checkpoint_model or DEFAULT_CHECKPOINT_MODEL)
        self.device = torch.device(device)

        if not self.model_pkl.exists():
            raise FileNotFoundError(f"Model pkl not found at: {self.model_pkl}")
        if not self.checkpoint_model.exists():
            raise FileNotFoundError(f"Checkpoint model not found at: {self.checkpoint_model}")

        print(f"[*] Initializing SpinePredictor on device: {self.device}")
        print(f"    - Model PKL: {self.model_pkl.name}")
        print(f"    - Checkpoint: {self.checkpoint_model.name}")

        t0 = time.time()
        self.trainer = restore_model(
            str(self.model_pkl),
            checkpoint=str(self.checkpoint_model),
            train=False
        )

        if self.device.type == "cuda":
            self.trainer.network.to(self.device)
            self.trainer.network.eval()
            print(f"[*] Model loaded to GPU ({torch.cuda.get_device_name(0)}) in {time.time()-t0:.2f}s")
        else:
            self.trainer.network.eval()
            print(f"[*] Model loaded to CPU in {time.time()-t0:.2f}s")

    def predict(
        self,
        input_ct_path: Union[str, Path],
        output_mask_path: Union[str, Path],
        fast_mode: bool = False,
        step_size: float = 0.5,
        mixed_precision: bool = True
    ) -> str:
        """
        Executes spinal vertebrae segmentation on a 3D CT NIfTI file.

        Args:
            input_ct_path: Path to the raw CT image (.nii or .nii.gz).
            output_mask_path: Path where the initial segmentation mask will be saved.
            fast_mode: If True, disables 8x test-time augmentation (TTA) for 8x faster inference.
            step_size: Sliding window stride (default 0.5; use 0.7 for even faster prediction).
            mixed_precision: Use FP16 mixed precision on GPU.

        Returns:
            The absolute path to the saved segmentation mask file.
        """
        input_ct_path = Path(input_ct_path).resolve()
        output_mask_path = Path(output_mask_path).resolve()
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_ct_path.exists():
            raise FileNotFoundError(f"Input CT file does not exist: {input_ct_path}")

        print(f"\n{'='*70}")
        print(f"[*] Starting Spinal Vertebrae Segmentation")
        print(f"    - Input CT:  {input_ct_path.name}")
        print(f"    - Output Mask: {output_mask_path.name}")
        print(f"    - Fast Mode (TTA Off): {fast_mode}")
        print(f"    - Mixed Precision: {mixed_precision}")
        print(f"{'='*70}")

        start_time = time.time()

        # Step 1: Preprocessing (Resampling to target spacing and intensity windowing)
        print("[1/3] Preprocessing input CT volume...")
        d, _, properties = self.trainer.preprocess_patient([str(input_ct_path)])
        print(f"      Preprocessed volume shape: {d.shape}")

        # Step 2: Sliding Window Neural Network Inference
        do_mirror = not fast_mode and self.trainer.data_aug_params.get("do_mirror", True)
        mirror_axes = self.trainer.data_aug_params.get("mirror_axes", (0, 1, 2)) if do_mirror else ()

        print(f"[2/3] Performing 3D sliding-window neural network inference (TTA={do_mirror})...")
        inference_t0 = time.time()
        
        with torch.no_grad():
            res = self.trainer.predict_preprocessed_data_return_seg_and_softmax(
                d,
                do_mirroring=do_mirror,
                mirror_axes=mirror_axes,
                use_sliding_window=True,
                step_size=step_size,
                use_gaussian=True,
                pad_border_mode="constant",
                pad_kwargs={"constant_values": 0},
                verbose=False,
                all_in_gpu=False,
                mixed_precision=mixed_precision and (self.device.type == "cuda")
            )
            # res is tuple: (seg, softmax)
            softmax_pred = res[1]

        print(f"      Inference completed in {time.time()-inference_t0:.2f}s")

        # Step 3: Resample prediction back to original resolution and save NIfTI
        print("[3/3] Resampling back to original patient spacing and exporting NIfTI...")
        # Transpose backward
        softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.trainer.transpose_backward])

        if "segmentation_export_params" in self.trainer.plans.keys():
            force_separate_z = self.trainer.plans["segmentation_export_params"].get("force_separate_z", None)
            interpolation_order = self.trainer.plans["segmentation_export_params"].get("interpolation_order", 1)
            interpolation_order_z = self.trainer.plans["segmentation_export_params"].get("interpolation_order_z", 0)
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0

        save_segmentation_nifti_from_softmax(
            softmax_pred,
            str(output_mask_path),
            properties,
            interpolation_order,
            self.trainer.regions_class_order,
            None,
            None,
            None,
            None,
            force_separate_z=force_separate_z,
            interpolation_order_z=interpolation_order_z
        )

        total_time = time.time() - start_time
        print(f"[OK] Initial segmentation mask successfully saved: {output_mask_path}")
        print(f"    Total elapsed time: {total_time:.2f}s")
        return str(output_mask_path)
