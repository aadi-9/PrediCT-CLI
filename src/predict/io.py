from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Volume:
    array: np.ndarray
    spacing_zyx: tuple[float, float, float] | None = None
    meta: dict[str, Any] | None = None


def discover_subject_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted([p for p in raw_dir.iterdir() if p.is_dir()])


def read_dicom_series(series_dir: Path) -> Volume:
    """
    Reads a DICOM series folder into a numpy array shaped (Z, Y, X).

    Next steps:
    - Confirm how COCA is stored on disk (single folder per series vs nested).
    - If you have multiple phases/series per subject, add filtering logic here.
    """
    try:
        import SimpleITK as sitk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "SimpleITK is required to read DICOM series. Install with: pip install SimpleITK"
        ) from e

    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")
    reader.SetFileNames(file_names)
    image = reader.Execute()

    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    size_zyx = tuple(int(v) for v in arr_zyx.shape)
    meta: dict[str, Any] = {
        "origin_xyz": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
        "spacing_xyz": tuple(float(v) for v in spacing_xyz),
        "size_xyz": tuple(int(v) for v in image.GetSize()),
        "spacing_zyx": spacing_zyx,
        "size_zyx": size_zyx,
        "source_dir": str(series_dir),
    }
    return Volume(array=arr_zyx, spacing_zyx=spacing_zyx, meta=meta)


def save_numpy_volume(volume: Volume, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, volume.array)


def load_numpy_volume(path: Path) -> Volume:
    arr = np.load(path)
    size_zyx = tuple(int(v) for v in arr.shape)
    spacing_zyx = (1.0, 1.0, 1.0)
    return Volume(
        array=arr,
        spacing_zyx=spacing_zyx,
        meta={
            "source_file": str(path),
            "size_zyx": size_zyx,
            "spacing_zyx": spacing_zyx,
        },
    )


def load_nifti_volume(path: Path) -> Volume:
    try:
        import SimpleITK as sitk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("SimpleITK is required to read NIfTI volumes. Install with: pip install SimpleITK") from e

    image = sitk.ReadImage(str(path))
    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    size_zyx = tuple(int(v) for v in arr_zyx.shape)
    return Volume(
        array=arr_zyx,
        spacing_zyx=spacing_zyx,
        meta={
            "source_file": str(path),
            "size_zyx": size_zyx,
            "spacing_xyz": tuple(float(v) for v in spacing_xyz),
            "spacing_zyx": spacing_zyx,
            "origin_xyz": tuple(float(v) for v in image.GetOrigin()),
            "direction": tuple(float(v) for v in image.GetDirection()),
        },
    )
