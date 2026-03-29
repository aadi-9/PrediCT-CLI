from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RadiomicsConfig:
    bin_width: float = 25.0
    normalize: bool = True
    normalize_scale: float = 100.0


def _read_dicom_sitk(series_dir: Path):
    try:
        import SimpleITK as sitk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("SimpleITK is required to read DICOM series. Install: pip install SimpleITK") from e

    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def _read_mask_sitk(mask_path: Path):
    try:
        import SimpleITK as sitk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("SimpleITK is required to read mask images. Install: pip install SimpleITK") from e

    return sitk.ReadImage(str(mask_path))


def _spacing_zyx_from_sitk(img) -> tuple[float, float, float]:
    sp = img.GetSpacing()
    return (float(sp[2]), float(sp[1]), float(sp[0]))


def extract_selected_radiomics_features(
    image_dicom_dir: Path,
    mask_path: Path,
    cfg: RadiomicsConfig | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    cfg = cfg or RadiomicsConfig()

    try:
        from radiomics import featureextractor  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("PyRadiomics is required. Install with: pip install pyradiomics") from e

    img = _read_dicom_sitk(image_dicom_dir)
    msk = _read_mask_sitk(mask_path)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=float(cfg.bin_width),
        normalize=bool(cfg.normalize),
        normalizeScale=float(cfg.normalize_scale),
    )

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(
        "shape", ["Sphericity", "SurfaceVolumeRatio", "Maximum3DDiameter"]
    )
    extractor.enableFeatureClassByName(
        "glcm", ["Contrast", "Correlation", "InverseDifferenceMoment"]
    )
    extractor.enableFeatureClassByName(
        "glszm", ["SmallAreaEmphasis", "LargeAreaEmphasis", "ZonePercentage"]
    )
    extractor.enableFeatureClassByName(
        "glrlm", ["ShortRunEmphasis", "LongRunEmphasis", "RunPercentage"]
    )

    result = extractor.execute(img, msk)

    features: dict[str, float] = {}
    for k, v in result.items():
        if not str(k).startswith("original_"):
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            features[str(k)] = float(v)

    meta: dict[str, Any] = {
        "image_dir": str(image_dicom_dir),
        "mask_path": str(mask_path),
        "spacing_zyx": _spacing_zyx_from_sitk(img),
        "bin_width": float(cfg.bin_width),
        "normalize": bool(cfg.normalize),
        "normalize_scale": float(cfg.normalize_scale),
    }
    return features, meta

