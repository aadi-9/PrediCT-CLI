from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import os

PREDICT_RAW_DIR_ENV = "PREDICT_RAW_DIR"
DEFAULT_RAW_DICOM_DIR = Path(r"I:\My Drive\GSoC_PrediCT\data_raw\dicom\Gated_release_final\patient")


def resolve_raw_dir(project_root: Path, raw_dir: Path | None = None) -> Path:
    if raw_dir is None:
        env_raw = os.getenv(PREDICT_RAW_DIR_ENV, "").strip()
        raw_dir = Path(env_raw) if env_raw else DEFAULT_RAW_DICOM_DIR
    if raw_dir.is_absolute():
        return raw_dir
    return (project_root / raw_dir).resolve()


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path

    @staticmethod
    def from_project_root(project_root: Path, raw_dir: Path | None = None) -> "PathsConfig":
        data_dir = project_root / "data"
        return PathsConfig(
            project_root=project_root,
            data_dir=data_dir,
            raw_dir=resolve_raw_dir(project_root, raw_dir=raw_dir),
            processed_dir=data_dir / "processed",
            outputs_dir=project_root / "outputs",
        )


@dataclass(frozen=True)
class ResampleConfig:
    mode: Literal["shape", "spacing"] = "spacing"
    target_shape: tuple[int, int, int] = (128, 128, 128)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolator: Literal["linear", "nearest"] = "linear"

    # Next steps:
    # - Decide whether you standardize by voxel spacing (mm) or by output shape.
    # - For CT + segmentation masks, typically: image uses linear; mask uses nearest.


@dataclass(frozen=True)
class HUWindowConfig:
    lower: float = -200.0
    upper: float = 400.0

    # Next steps:
    # - Pick HU bounds appropriate for cardiac CT windowing and justify them.
    # - Keep the bounds fixed across train/val/test.


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # Next steps:
    # - Use stratification labels that reflect your downstream task (e.g., disease class).
    # - Ensure splits are patient-level (not slice-level) to avoid leakage.


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True

    # Next steps:
    # - On Colab GPU, set pin_memory=True and num_workers>0 if it improves throughput.
