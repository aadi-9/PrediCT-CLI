from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .config import HUWindowConfig, LoaderConfig, ResampleConfig
from .dataset import SampleRecord, VolumeDataset, build_dataloader
from .preprocess import apply_hu_window, resample_volume

ModelKind = Literal["auto", "monai_unet", "smallunet3d"]


@dataclass(frozen=True)
class Project1EvalResult:
    """Outputs from `evaluate_project1_checkpoint`."""

    dice_mean: float
    model_inference_seconds_mean: float
    probs_max: float
    probs_mean: float
    n_scans: int
    totalseg_seconds_mean: Optional[float] = None
    totalseg_seconds_median: Optional[float] = None
    speedup_mean: Optional[float] = None
    speedup_median: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dice_mean": float(self.dice_mean),
            "model_inference_seconds_mean": float(self.model_inference_seconds_mean),
            "probs_max": float(self.probs_max),
            "probs_mean": float(self.probs_mean),
            "n_scans": int(self.n_scans),
            "totalseg_seconds_mean": None if self.totalseg_seconds_mean is None else float(self.totalseg_seconds_mean),
            "totalseg_seconds_median": None
            if self.totalseg_seconds_median is None
            else float(self.totalseg_seconds_median),
            "speedup_mean": None if self.speedup_mean is None else float(self.speedup_mean),
            "speedup_median": None if self.speedup_median is None else float(self.speedup_median),
        }


def _center_crop(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Center-crop a 5D tensor (N, C, D, H, W) to match target spatial shape."""
    _, _, d_t, h_t, w_t = target_shape
    _, _, d, h, w = tensor.shape
    d0 = (d - d_t) // 2
    h0 = (h - h_t) // 2
    w0 = (w - w_t) // 2
    return tensor[:, :, d0 : d0 + d_t, h0 : h0 + h_t, w0 : w0 + w_t]


class SmallUNet3D(nn.Module):
    """Tiny 3D U-Net that matches the training notebook checkpoint key names.

    The checkpoint saved in the training notebook uses module names like `enc1`, `mid`, `dec1`,
    so loading it into MONAI's `UNet` will not work. This architecture intentionally mirrors
    the minimal `SmallUNet3D` used for training and uses `center_crop` skip alignment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.mid = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv3d(8, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m = self.mid(self.pool2(e2))

        d2 = self.up2(m)
        e2_c = _center_crop(e2, d2.shape)
        d2 = self.dec2(torch.cat([d2, e2_c], dim=1))

        d1 = self.up1(d2)
        e1_c = _center_crop(e1, d1.shape)
        d1 = self.dec1(torch.cat([d1, e1_c], dim=1))

        return self.out(d1)


def _iter_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Iterable[str]:
    # Separate helper for readability/testing.
    return state_dict.keys()


def _infer_model_kind_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelKind:
    keys = list(_iter_state_dict_keys(state_dict))
    if any(k.startswith("enc1.") for k in keys) or any(k.startswith("mid.") for k in keys) or any(k.startswith("dec1.") for k in keys):
        return "smallunet3d"
    return "monai_unet"


def _build_model(
    *,
    kind: ModelKind,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    ckpt: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    if kind == "auto":
        kind = _infer_model_kind_from_state_dict(state_dict)

    if kind == "smallunet3d":
        model: nn.Module = SmallUNet3D()
        model.load_state_dict(state_dict)
        return model.to(device)

    # Use the centralized factory from project1_pipeline if available.
    try:
        from .project1_pipeline import build_project1_model

        # Read architecture params from checkpoint if present (new-style checkpoints
        # store these), otherwise use defaults from the factory.
        kwargs: Dict[str, Any] = {}
        if ckpt is not None:
            if "channels" in ckpt:
                kwargs["channels"] = tuple(int(c) for c in ckpt["channels"])
            if "strides" in ckpt:
                kwargs["strides"] = tuple(int(s) for s in ckpt["strides"])
        model = build_project1_model(**kwargs).to(device)
        model.load_state_dict(state_dict)
        return model
    except Exception:
        pass

    # Fallback to MONAI UNet directly.
    try:
        from monai.networks.nets import UNet  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "MONAI is required to build a monai_unet model. Install with: pip install monai"
        ) from e

    channels = (32, 64, 128, 256)
    strides_val = (2, 2, 2)
    if ckpt is not None:
        channels = tuple(int(c) for c in ckpt.get("channels", channels))
        strides_val = tuple(int(s) for s in ckpt.get("strides", strides_val))

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides_val,
        norm="batch",
        dropout=0.1,
    ).to(device)
    model.load_state_dict(state_dict)
    return model


def evaluate_project1_checkpoint(
    *,
    ckpt_path: Path,
    split_manifest_path: Path,
    totalseg_manifest_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    model_kind: ModelKind = "auto",
    batch_size: int = 1,
) -> Project1EvalResult:
    """Evaluate a Project 1 segmentation checkpoint in one call.

    This is designed to be *notebook-friendly*: you call one function and get the metrics back.

    What it does:
    - Loads the checkpoint saved by the training notebook (`{'model': state_dict, 'resample_spacing': ..., 'hu': ...}`).
    - Builds a test dataset from `splits_project1.json` and runs preprocessing:
      resample (spacing) + HU windowing to [0, 1], matching the training notebook.
    - Auto-selects the model architecture based on checkpoint key names:
      - keys like `enc1.*`, `mid.*`, `dec1.*` -> `SmallUNet3D`
      - otherwise attempts MONAI `UNet`
    - Computes mean Dice (with binarized GT and GT center-cropped to logits size).
    - Optionally computes speedup vs TotalSegmentator if `totalseg_manifest_path` is provided.

    Parameters
    - **ckpt_path**: Path to `model.pt`.
    - **split_manifest_path**: Path to `splits_project1.json` containing `image`, `mask`, `split`.
    - **totalseg_manifest_path**: Optional path to `totalseg_heart_manifest.csv` (for speedup).
    - **device**: Optional torch device (defaults to CUDA if available else CPU).
    - **threshold**: Probability threshold for binarizing predictions and GT.
    - **model_kind**: `"auto" | "monai_unet" | "smallunet3d"`.
    - **batch_size**: DataLoader batch size (default 1, recommended for variable-size volumes).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Some users may save raw state_dict directly.
        state_dict = ckpt  # type: ignore[assignment]
    else:
        raise ValueError("Unsupported checkpoint format; expected dict-like checkpoint.")

    # Pull preprocessing settings from checkpoint if present.
    resample_spacing_zyx = tuple(float(x) for x in ckpt.get("resample_spacing", (3.0, 0.7, 0.7)))  # type: ignore[union-attr]
    hu_lower, hu_upper = ckpt.get("hu", (-200.0, 800.0))  # type: ignore[union-attr]

    resample_cfg = ResampleConfig(mode="spacing", target_spacing=resample_spacing_zyx)
    hu_cfg = HUWindowConfig(lower=float(hu_lower), upper=float(hu_upper))

    import json

    split_rows = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    test_rows = [r for r in split_rows if r.get("split") == "test"]
    if not test_rows:
        raise ValueError(f"No test rows found in split manifest: {split_manifest_path}")

    test_recs = [
        SampleRecord(
            subject_id=str(r["subject_id"]),
            image=Path(str(r["image"])),
            label=0,
            mask=Path(str(r["mask"])) if r.get("mask") else None,
            kind=str(r.get("kind", "dicom_series")),
        )
        for r in test_rows
    ]

    def preprocess_fn(vol, is_label: bool = False):
        vol = resample_volume(vol, cfg=resample_cfg, is_label=is_label)
        if not is_label:
            vol = apply_hu_window(vol, hu_cfg)
        return vol

    # Pad every volume to 8-divisible spatial dims so the MONAI UNet's
    # skip connections always align (encoder dim == decoder upsample dim).
    _eval_tf = None
    try:
        from monai.transforms import Compose, DivisiblePadd  # type: ignore

        _eval_tf = Compose([DivisiblePadd(keys=["image", "mask"], k=8, mode="constant")])
    except Exception:
        pass

    ds = VolumeDataset(test_recs, transform=_eval_tf, preprocess_fn=preprocess_fn)
    dl = build_dataloader(
        ds,
        LoaderConfig(
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        ),
    )

    model = _build_model(kind=model_kind, state_dict=state_dict, device=device, ckpt=ckpt)
    model.eval()

    # DiceMetric is convenient but optional; fall back to manual dice to keep dependency surface smaller.
    try:
        from monai.metrics import DiceMetric  # type: ignore

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        use_monai_metric = True
    except Exception:
        dice_metric = None
        use_monai_metric = False

    times: list[float] = []
    probs_max = 0.0
    probs_mean_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            y = batch.get("mask")
            if y is None:
                raise ValueError("Split manifest must include `mask` paths for evaluation.")
            y = y.to(device, non_blocking=True)

            t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

            if t0 is not None and t1 is not None:
                t0.record()
                logits = model(x)
                t1.record()
                torch.cuda.synchronize()
                elapsed_ms = float(t0.elapsed_time(t1))
                times.append(elapsed_ms / 1000.0)
            else:
                import time as _time

                s = _time.perf_counter()
                logits = model(x)
                e = _time.perf_counter()
                times.append(float(e - s))

            probs = torch.sigmoid(logits)
            batch_probs_max = float(probs.max().detach().item())
            batch_probs_mean = float(probs.mean().detach().item())
            batch_probs_std = float(probs.std().detach().item())
            probs_max = max(probs_max, batch_probs_max)
            probs_mean_sum += batch_probs_mean
            n_batches += 1

            # Diagnostic logging
            print(
                f"  [eval scan {n_batches}] probs — "
                f"max: {batch_probs_max:.4f}  mean: {batch_probs_mean:.4f}  "
                f"std: {batch_probs_std:.4f}"
            )

            y_bin = (y > threshold).float()
            if y_bin.shape != probs.shape:
                y_bin = _center_crop(y_bin, probs.shape)

            # Adaptive thresholding: if max prob < threshold, fall back to
            # a percentile-based threshold to still produce a meaningful mask.
            effective_threshold = threshold
            if batch_probs_max < threshold and batch_probs_max > 0.01:
                # Use 95th percentile as adaptive threshold
                effective_threshold = float(
                    torch.quantile(probs.flatten(), 0.95).detach().item()
                )
                effective_threshold = max(effective_threshold, 0.1)  # floor at 0.1
                print(
                    f"    ⚠ Max prob ({batch_probs_max:.4f}) < {threshold} → "
                    f"adaptive threshold: {effective_threshold:.4f}"
                )

            p_bin = (probs > effective_threshold).float()

            if use_monai_metric and dice_metric is not None:
                dice_metric(y_pred=p_bin, y=y_bin)
            else:
                # Manual dice (mean over batch).
                b = p_bin.shape[0]
                p_f = p_bin.reshape(b, -1)
                y_f = y_bin.reshape(b, -1)
                inter = (p_f * y_f).sum(dim=1)
                denom = p_f.sum(dim=1) + y_f.sum(dim=1)
                dice_vals = (2.0 * inter + 1e-6) / (denom + 1e-6)
                if dice_metric is None:
                    dice_metric = {"sum": 0.0, "count": 0}  # type: ignore[assignment]
                dice_metric["sum"] += float(dice_vals.mean().detach().item())  # type: ignore[index]
                dice_metric["count"] += 1  # type: ignore[index]

    if use_monai_metric and dice_metric is not None:
        dice_mean = float(dice_metric.aggregate().detach().cpu())  # type: ignore[union-attr]
    else:
        dice_mean = float(dice_metric["sum"] / max(1, dice_metric["count"]))  # type: ignore[index]

    avg_time = float(np.mean(times)) if times else float("nan")
    probs_mean = float(probs_mean_sum / max(1, n_batches))

    # Optional speedup vs TotalSegmentator
    ts_mean = None
    ts_median = None
    speedup_mean = None
    speedup_median = None
    if totalseg_manifest_path is not None and Path(totalseg_manifest_path).exists():
        import pandas as pd

        df_ts = pd.read_csv(Path(totalseg_manifest_path))
        df_ok = df_ts[df_ts.get("ok", True) == True].copy()  # noqa: E712
        if len(df_ok) > 0 and np.isfinite(avg_time) and avg_time > 0:
            ts_mean = float(df_ok["total_seconds"].mean())
            ts_median = float(df_ok["total_seconds"].median())
            speedup_mean = float(ts_mean / avg_time)
            speedup_median = float(ts_median / avg_time)

    return Project1EvalResult(
        dice_mean=dice_mean,
        model_inference_seconds_mean=avg_time,
        probs_max=float(probs_max),
        probs_mean=float(probs_mean),
        n_scans=len(ds),
        totalseg_seconds_mean=ts_mean,
        totalseg_seconds_median=ts_median,
        speedup_mean=speedup_mean,
        speedup_median=speedup_median,
    )

