from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from .config import HUWindowConfig, ResampleConfig
from .io import Volume


def hu_windowing(
    image_zyx: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """
    HU windowing for CT.

    Example (from instructions):
        img = hu_windowing(img, lower_bound=-200, upper_bound=400)

    Next steps:
    - Confirm the pixel values are in HU. If not, apply DICOM rescale slope/intercept first.
    """
    img = np.asarray(image_zyx, dtype=np.float32)
    img = np.clip(img, lower_bound, upper_bound)
    img = (img - lower_bound) / (upper_bound - lower_bound + 1e-8)
    img = img * (out_max - out_min) + out_min
    return img.astype(np.float32)


def apply_hu_window(volume: Volume, cfg: HUWindowConfig) -> Volume:
    return replace(volume, array=hu_windowing(volume.array, cfg.lower, cfg.upper))


def _scipy_zoom(
    arr: np.ndarray, zoom_zyx: tuple[float, float, float], order: int
) -> np.ndarray:
    try:
        from scipy.ndimage import zoom
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("scipy is required for resampling via zoom. Install: pip install scipy") from e
    return zoom(arr, zoom=zoom_zyx, order=order)


def resample_volume(
    volume: Volume,
    cfg: ResampleConfig,
    is_label: bool = False,
) -> Volume:
    """
    Resample a 3D volume to a target shape or spacing.

    - mode="shape": always outputs cfg.target_shape (Z, Y, X)
    - mode="spacing": uses volume.spacing_zyx and cfg.target_spacing

    Next steps:
    - Decide if you resample images and labels the same way.
    - For segmentation: image -> linear interpolation, label -> nearest interpolation.
    """
    interp: Literal["linear", "nearest"] = cfg.interpolator
    if is_label:
        interp = "nearest"

    order = 1 if interp == "linear" else 0
    arr = np.asarray(volume.array)

    original_size_zyx = tuple(int(v) for v in arr.shape)
    existing_meta = dict(volume.meta or {})

    if cfg.mode == "shape":
        tz, ty, tx = cfg.target_shape
        z, y, x = arr.shape
        zoom_zyx = (tz / z, ty / y, tx / x)
        out = _scipy_zoom(arr, zoom_zyx=zoom_zyx, order=order)
        out_arr = out.astype(np.float32)
        out_meta = {
            **existing_meta,
            "original_size_zyx": original_size_zyx,
            "processed_size_zyx": tuple(int(v) for v in out_arr.shape),
            "processed_spacing_zyx": None,
        }
        return Volume(array=out_arr, spacing_zyx=None, meta=out_meta)

    if cfg.mode == "spacing":
        if volume.spacing_zyx is None:
            raise ValueError("volume.spacing_zyx is required when cfg.mode='spacing'")
        sz, sy, sx = volume.spacing_zyx
        tz, ty, tx = cfg.target_spacing
        zoom_zyx = (sz / tz, sy / ty, sx / tx)
        out = _scipy_zoom(arr, zoom_zyx=zoom_zyx, order=order)
        out_arr = out.astype(np.float32)
        out_meta = {
            **existing_meta,
            "original_size_zyx": original_size_zyx,
            "processed_size_zyx": tuple(int(v) for v in out_arr.shape),
            "processed_spacing_zyx": cfg.target_spacing,
        }
        return Volume(array=out_arr, spacing_zyx=cfg.target_spacing, meta=out_meta)

    raise ValueError(f"Unknown resample mode: {cfg.mode}")
