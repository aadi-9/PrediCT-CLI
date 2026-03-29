"""Project 1 – Heart-mask segmentation: training pipeline.

This module provides:
- ``build_project1_model``: a factory that builds the MONAI 3D UNet used in
  both training and evaluation.  Centralising the architecture here ensures
  that checkpoint keys always match.
- ``run_training_pipeline``: a production-ready training loop with
  DiceCE loss, cosine-annealing LR, gradient clipping, validation Dice
  tracking, and best-model checkpointing.

Dependencies: torch, monai, SimpleITK, scikit-learn, scipy, numpy, tqdm.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import SimpleITK as sitk
except ImportError:
    pass

try:
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        RandZoomd,
    )
except ImportError:
    pass

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

# Default architecture hyper-parameters — keep in sync with evaluation code.
_DEFAULT_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)
_DEFAULT_STRIDES: Tuple[int, ...] = (2, 2, 2)


def build_project1_model(
    *,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: Tuple[int, ...] = _DEFAULT_CHANNELS,
    strides: Tuple[int, ...] = _DEFAULT_STRIDES,
    norm: str = "batch",
    dropout: float = 0.1,
) -> nn.Module:
    """Build the MONAI 3D U-Net used for Project 1 heart segmentation.

    This is the **single source of truth** for the architecture.  Both the
    training loop (``run_training_pipeline``) and the evaluation helper
    (``project1_eval._build_model``) should call this function so that
    checkpoint key names always match.
    """
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        norm=norm,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with older notebooks)
# ---------------------------------------------------------------------------

def load_dicom_series(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def resample(image, new_spacing=(0.7, 0.7, 3.0), is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def hu_window(image_array, min_hu=-200, max_hu=800):
    image_array = np.clip(image_array, min_hu, max_hu)
    image_array = (image_array - min_hu) / (max_hu - min_hu)
    return image_array.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset (legacy – kept for notebooks that still define inline datasets)
# ---------------------------------------------------------------------------

class HeartDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sitk_image = load_dicom_series(self.image_paths[idx])
        sitk_image = resample(sitk_image, is_mask=False)
        image = sitk.GetArrayFromImage(sitk_image)

        sitk_mask = sitk.ReadImage(str(self.mask_paths[idx]))
        sitk_mask = resample(sitk_mask, is_mask=True)
        mask = sitk.GetArrayFromImage(sitk_mask)

        image = hu_window(image)

        sample = {
            "image": image[np.newaxis, ...],
            "mask": mask[np.newaxis, ...],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def get_train_transforms():
    return Compose([
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "mask"], prob=0.5),
        RandZoomd(keys=["image", "mask"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
        RandGaussianNoised(keys=["image"], prob=0.2),
    ])


# ---------------------------------------------------------------------------
# Stratified split helpers
# ---------------------------------------------------------------------------

def compute_calcium_score(image_array):
    return (image_array > 130).sum()


def stratified_split_by_calcium(image_paths, mask_paths):
    scores = []
    for path in image_paths:
        sitk_image = load_dicom_series(path)
        arr = sitk.GetArrayFromImage(sitk_image)
        scores.append(compute_calcium_score(arr))

    bins = np.percentile(scores, [33, 66])
    labels = np.digitize(scores, bins)

    return train_test_split(
        image_paths, mask_paths, test_size=0.2, stratify=labels, random_state=42
    )


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def dataset_stats(images):
    hu_values = np.concatenate([img.flatten() for img in images])
    print("Mean HU:", np.mean(hu_values))
    print("Std HU:", np.std(hu_values))
    print("Min/Max:", np.min(hu_values), np.max(hu_values))


# ---------------------------------------------------------------------------
# Center-crop utility (for skip-connection alignment in custom models)
# ---------------------------------------------------------------------------

def _center_crop(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Center-crop a 5-D tensor (N, C, D, H, W) to *target_shape*."""
    _, _, d_t, h_t, w_t = target_shape
    _, _, d, h, w = tensor.shape
    d0 = (d - d_t) // 2
    h0 = (h - h_t) // 2
    w0 = (w - w_t) // 2
    return tensor[:, :, d0: d0 + d_t, h0: h0 + h_t, w0: w0 + w_t]


# ---------------------------------------------------------------------------
# Production training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Outputs from ``run_training_pipeline``."""
    best_dice: float
    final_dice: float
    best_epoch: int
    total_epochs: int
    avg_epoch_seconds: float
    log_rows: List[Dict[str, Any]]


def run_training_pipeline(
    image_paths=None,
    mask_paths=None,
    *,
    # If using the VolumeDataset-based loaders (preferred):
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    # Hyper-parameters:
    num_epochs: int = 50,
    batch_size: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip_norm: float = 1.0,
    use_amp: bool = True,
    # Output paths:
    ckpt_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> TrainResult:
    """Train a MONAI 3D U-Net for coarse heart-mask segmentation.

    Key improvements over the v1 pipeline:
    - **DiceCELoss** (differentiable soft-Dice + cross-entropy) instead of
      the broken hard-threshold Dice + BCE.
    - **MONAI UNet** with BatchNorm, 4 encoder levels (32→64→128→256),
      and dropout — ~2 M parameters.
    - **Cosine-annealing LR** with warm restarts for stable convergence.
    - **Gradient clipping** (max-norm 1.0).
    - **Best-model checkpointing** (saves the epoch with highest val Dice).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- mixed-precision setup -------
    # bfloat16 on A100 (wide dynamic range, no scaler needed).
    # float16 + GradScaler on T4/V100.  Disabled on CPU.
    _amp_on = use_amp and device.type == "cuda"
    _amp_dtype = torch.bfloat16 if (_amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    _scaler = torch.amp.GradScaler(device.type, enabled=(_amp_on and _amp_dtype == torch.float16))
    if _amp_on:
        print(f"[train] AMP enabled — dtype={_amp_dtype}")

    # ------- build loaders if legacy paths are provided -------
    if train_loader is None:
        if image_paths is None or mask_paths is None:
            raise ValueError(
                "Provide either (train_loader, val_loader) or (image_paths, mask_paths)."
            )
        X_train, X_val, y_train, y_val = stratified_split_by_calcium(
            image_paths, mask_paths
        )
        train_ds = HeartDataset(X_train, y_train, transform=get_train_transforms())
        val_ds = HeartDataset(X_val, y_val, transform=None)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # ------- model -------
    model = build_project1_model().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {num_params:,}")

    # ------- loss (DIFFERENTIABLE Dice + CE) -------
    loss_function = DiceCELoss(
        sigmoid=True,        # model outputs raw logits
        squared_pred=True,   # denominator uses p² for smoother gradients
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    # ------- optimizer + scheduler -------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ------- metrics -------
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # ------- training state -------
    best_dice = -1.0
    best_epoch = 0
    log_rows: List[Dict[str, Any]] = []

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        epoch_losses: list[float] = []
        epoch_dices: list[float] = []
        t0 = time.perf_counter()

        loader_iter = train_loader
        if tqdm is not None:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")

        for batch in loader_iter:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=_amp_dtype, enabled=_amp_on):
                logits = model(inputs)

                # Crop labels to match logits spatial size (for architectures
                # where the output is slightly smaller than the input).
                if logits.shape != labels.shape:
                    labels = _center_crop(labels, logits.shape)

                loss = loss_function(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            _scaler.scale(loss).backward()
            _scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            _scaler.step(optimizer)
            _scaler.update()

            # Track per-batch metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                p_bin = (probs > 0.5).float()
                l_bin = (labels > 0.5).float()

                # Quick inline Dice for logging
                inter = (p_bin * l_bin).sum()
                union = p_bin.sum() + l_bin.sum()
                batch_dice = float((2.0 * inter + 1e-6) / (union + 1e-6))

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_dices.append(batch_dice)

            if tqdm is not None and hasattr(loader_iter, "set_postfix"):
                loader_iter.set_postfix(
                    loss=f"{epoch_losses[-1]:.4f}",
                    dice=f"{batch_dice:.4f}",
                    avg_p=f"{float(probs.mean().cpu()):.4f}",
                )

        scheduler.step(epoch)

        # ---- validate ----
        model.eval()
        val_dices: list[float] = []
        dice_metric.reset()

        val_iter = val_loader
        if tqdm is not None:
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")

        with torch.no_grad():
            for batch in val_iter:
                inputs = batch["image"].to(device, non_blocking=True)
                labels = batch["mask"].to(device, non_blocking=True)
                logits = model(inputs)

                if logits.shape != labels.shape:
                    labels = _center_crop(labels, logits.shape)

                probs = torch.sigmoid(logits)
                p_bin = (probs > 0.5).float()
                l_bin = (labels > 0.5).float()
                dice_metric(y_pred=p_bin, y=l_bin)

                # Per-sample Dice for logging
                inter = (p_bin * l_bin).sum()
                union = p_bin.sum() + l_bin.sum()
                d = float((2.0 * inter + 1e-6) / (union + 1e-6))
                val_dices.append(d)

                if tqdm is not None and hasattr(val_iter, "set_postfix"):
                    val_iter.set_postfix(dice=f"{d:.4f}")

        epoch_val_dice = float(dice_metric.aggregate().detach().cpu())
        dt = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(epoch_losses)),
            "train_dice_mean": float(np.mean(epoch_dices)),
            "val_dice_mean": epoch_val_dice,
            "lr": current_lr,
            "seconds": float(dt),
        }
        log_rows.append(row)
        print(
            f"\nEpoch {epoch} — train_loss: {row['train_loss_mean']:.4f}  "
            f"train_dice: {row['train_dice_mean']:.4f}  "
            f"val_dice: {epoch_val_dice:.4f}  "
            f"lr: {current_lr:.2e}  "
            f"({dt:.1f}s)"
        )

        # ---- best-model checkpoint ----
        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            best_epoch = epoch
            if ckpt_dir is not None:
                ckpt_dir = Path(ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "model.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": epoch_val_dice,
                        "resample_spacing": (3.0, 0.7, 0.7),
                        "hu": (-200.0, 800.0),
                        "architecture": "monai_unet",
                        "channels": list(_DEFAULT_CHANNELS),
                        "strides": list(_DEFAULT_STRIDES),
                    },
                    ckpt_path,
                )
                print(f"  ✓ Best model saved (dice={epoch_val_dice:.4f}) → {ckpt_path}")

    # ---- save training log ----
    if log_path is not None and log_rows:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
        print(f"Training log saved → {log_path}")

    avg_epoch_time = float(np.mean([r["seconds"] for r in log_rows])) if log_rows else 0.0
    final_dice = log_rows[-1]["val_dice_mean"] if log_rows else 0.0

    print(f"\n{'='*60}")
    print(f"Training complete.  Best val Dice = {best_dice:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    return TrainResult(
        best_dice=best_dice,
        final_dice=final_dice,
        best_epoch=best_epoch,
        total_epochs=num_epochs,
        avg_epoch_seconds=avg_epoch_time,
        log_rows=log_rows,
    )
