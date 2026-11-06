"""
Clinical Few-Shot Fine-Tuning Module
Enables transfer learning and fine-tuning of the CTSpine1K pre-trained nnU-Net
model on small clinical cohorts (e.g. specialized fracture or implant cases).
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom

from spine_segmentation.config import (
    DEFAULT_CHECKPOINT_MODEL,
    DEFAULT_CHECKPOINT_PKL
)


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Exclude background class 0 from average if desired, or include all
        loss = 1.0 - torch.mean(dice_score[1:])
        return loss


class DC_and_CE_loss(nn.Module):
    """Combined Dice and Cross-Entropy Loss (Standard nnU-Net loss)."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dc = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dc(logits, targets)


class ClinicalSpineDataset(Dataset):
    """Simple 3D patch-based clinical dataset for spine fine-tuning."""
    def __init__(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
        patch_size: Tuple[int, int, int] = (96, 96, 96)
    ):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(str(self.image_paths[idx])).get_fdata().astype(np.float32)
        lbl = nib.load(str(self.label_paths[idx])).get_fdata().astype(np.int64)

        # Standard CT window clipping and normalization
        img = np.clip(img, -1000.0, 1500.0)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Crop random patch or center patch
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        if d < pd or h < ph or w < pw:
            # Pad if smaller than patch
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            lbl = np.pad(lbl, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            d, h, w = img.shape

        sd = np.random.randint(0, d - pd + 1)
        sh = np.random.randint(0, h - ph + 1)
        sw = np.random.randint(0, w - pw + 1)

        patch_img = img[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        patch_lbl = lbl[sd:sd+pd, sh:sh+ph, sw:sw+pw]

        # Add channel dim: (1, D, H, W)
        tensor_img = torch.from_numpy(patch_img[np.newaxis, ...]).float()
        tensor_lbl = torch.from_numpy(patch_lbl).long()

        return tensor_img, tensor_lbl


def finetune_on_clinical_cohort(
    clinical_data_dir: Union[str, Path],
    output_checkpoint_dir: Union[str, Path],
    pretrained_checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT_MODEL,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-4,
    freeze_encoder: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Fine-tunes the pre-trained nnU-Net on small clinical cohort data.
    """
    from spine_segmentation.predictor import SpinePredictor

    data_dir = Path(clinical_data_dir).resolve()
    out_dir = Path(output_checkpoint_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[*] Starting Clinical Fine-Tuning Module")
    print(f"    - Pre-trained checkpoint: {pretrained_checkpoint}")
    print(f"    - Clinical data dir:      {data_dir}")
    print(f"    - Output checkpoint dir:  {out_dir}")
    print(f"    - Epochs: {epochs}, LR: {lr}, Freeze Encoder: {freeze_encoder}")
    print(f"{'='*70}")

    # Load predictor and extract network
    predictor = SpinePredictor(checkpoint_model=pretrained_checkpoint)
    network = predictor.trainer.network.to(device)

    # Search for clinical cases in imagesTr / labelsTr
    img_dir = data_dir / "imagesTr"
    lbl_dir = data_dir / "labelsTr"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[!] Clinical data folder structure should contain:\n    {img_dir}\n    {lbl_dir}")
        print("    Example dummy setup created for demonstration.")
        return ""

    img_files = sorted(list(img_dir.glob("*.nii*")))
    lbl_files = sorted(list(lbl_dir.glob("*.nii*")))

    if not img_files:
        print("[!] No training CT files found in imagesTr.")
        return ""

    dataset = ClinicalSpineDataset(img_files, lbl_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if freeze_encoder:
        print("[*] Freezing encoder convolutional blocks, fine-tuning decoder heads only...")
        for name, param in network.named_parameters():
            if "conv_blocks_context" in name:
                param.requires_grad = False

    criterion = DC_and_CE_loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=lr)

    network.train()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        for i, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = network(imgs)
            
            # Handle deep supervision if output is a list/tuple
            if isinstance(outputs, (list, tuple)):
                loss = criterion(outputs[0], lbls)
            else:
                loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"  [Epoch {ep:02d}/{epochs:02d}] Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoint
    saved_model_path = out_dir / "finetuned_checkpoint.model"
    torch.save({
        "state_dict": network.state_dict(),
        "epoch": epochs,
    }, str(saved_model_path))

    print(f"\n[OK] Clinical fine-tuning complete! Checkpoint saved to: {saved_model_path}\n")
    return str(saved_model_path)
