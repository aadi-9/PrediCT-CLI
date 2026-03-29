"""Data augmentation for 3D medical-image segmentation.

Provides paired image+mask transforms using MONAI's dictionary-based
transforms so that spatial augmentations are applied identically to
images and their corresponding segmentation masks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_eval_transforms() -> Callable[[Any], Any] | None:
    """Minimal transforms for validation/test: pad to 8-divisible spatial dims.

    Ensures the MONAI UNet (3 stride-2 levels) receives inputs whose
    spatial dimensions are divisible by 8, which prevents skip-connection
    shape mismatches.  No augmentation is applied.

    Returns ``None`` if MONAI is not installed.
    """
    try:
        from monai.transforms import Compose, DivisiblePadd  # type: ignore[import-not-found]
    except Exception:
        return None

    return Compose(
        [DivisiblePadd(keys=["image", "mask"], k=8, mode="constant")]
    )


def build_monai_transforms(
    enable: bool = True,
    prob: float = 0.5,
    rotate_degrees: float = 15.0,
) -> Callable[[Any], Any] | None:
    """Build an augmentation pipeline for 3D segmentation training.

    Returns a MONAI ``Compose`` that operates on dicts with keys
    ``"image"`` and ``"mask"``.  Spatial transforms are applied
    identically to both; intensity transforms apply only to ``"image"``.

    Returns ``None`` if ``enable=False`` or if MONAI is not installed.
    """
    if not enable:
        return None

    try:
        from monai.transforms import (  # type: ignore[import-not-found]
            Compose,
            DivisiblePadd,
            RandAffined,
            RandFlipd,
            RandGaussianNoised,
            RandRotate90d,
            RandScaleIntensityd,
            RandZoomd,
        )
    except Exception:
        return None

    return Compose(
        [
            # Pad to 8-divisible spatial dims FIRST so that RandZoomd's
            # internal SpatialPadd never receives a dimension that is
            # smaller than the padding it needs to apply (which triggers
            # an error in reflect/edge padding modes for small dims like H=46).
            DivisiblePadd(keys=["image", "mask"], k=8, mode="constant"),
            # --- spatial (applied to image + mask) ---
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "mask"], prob=0.5),
            RandZoomd(
                keys=["image", "mask"],
                prob=0.3,
                min_zoom=0.85,
                max_zoom=1.15,
                mode=("trilinear", "nearest"),
                padding_mode="constant",
            ),
            RandAffined(
                keys=["image", "mask"],
                prob=0.3,
                rotate_range=(0.15, 0.15, 0.15),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            # --- intensity (image only) ---
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        ]
    )
