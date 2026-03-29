from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import LoaderConfig
from .io import Volume, load_nifti_volume, load_numpy_volume, read_dicom_series


@dataclass(frozen=True)
class SampleRecord:
    subject_id: str
    image: Path
    label: int
    mask: Path | None = None
    kind: str = "dicom_series"

    # Extended this record with paths to segmentation masks (for Project 1).


def default_load_volume(rec: SampleRecord) -> Volume:
    if rec.kind == "dicom_series":
        return read_dicom_series(rec.image)
    if rec.kind == "numpy":
        return load_numpy_volume(rec.image)
    if rec.kind in {"nifti", "nii", "nifti_gz"}:
        return load_nifti_volume(rec.image)
    raise ValueError(f"Unknown record kind: {rec.kind}")


class VolumeDataset:
    """
    Minimal PyTorch Dataset wrapper.

    Next steps:
    - Add caching (e.g., preprocessed .npy files) once resampling/windowing is finalized.
    - For segmentation, return (image, mask) instead of (image, label).
    """

    def __init__(
        self,
        records: list[SampleRecord],
        load_fn: Callable[[SampleRecord], Volume] = default_load_volume,
        transform: Callable[[Any], Any] | None = None,
        preprocess_fn: Callable[[Volume, bool], Volume] | None = None,
    ) -> None:
        self.records = records
        self.load_fn = load_fn
        self.transform = transform
        self.preprocess_fn = preprocess_fn

        try:
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "PyTorch is required for VolumeDataset. Install with: pip install torch"
            ) from e

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        vol = self.load_fn(rec)
        if self.preprocess_fn is not None:
            vol = self.preprocess_fn(vol)

        arr = np.asarray(vol.array, dtype=np.float32)

        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for __getitem__") from e

        # Ensure image has channel dimension
        image = torch.from_numpy(arr).unsqueeze(0)  # (C=1, Z, Y, X)
        
        sample = {"image": image}
        
        # Load mask if present
        if rec.mask is not None:
            mask_vol = load_nifti_volume(rec.mask)
            if self.preprocess_fn is not None:
                # Preprocess mask using the same resampler but with nearest neighbor if handled properly
                # We assume preprocess_fn handles is_label flag, or we pass it
                # For simplicity here, we assume mask_vol is aligned or handled by transforms
                mask_vol = self.preprocess_fn(mask_vol, is_label=True)
            mask_arr = np.asarray(mask_vol.array, dtype=np.float32)
            mask = torch.from_numpy(mask_arr).unsqueeze(0)
            sample["mask"] = mask
        else:
            sample["label"] = int(rec.label)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def pad_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-size volumes by zero-padding to the max shape in the batch.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            # Determine the maximum size along each spatial dimension
            max_shape = [max(v.shape[d] for v in values) for d in range(len(values[0].shape))]
            padded = []
            for v in values:
                pad: list[int] = []
                for d in reversed(range(len(v.shape))):
                    diff = max_shape[d] - v.shape[d]
                    pad.extend([0, diff])
                padded.append(F.pad(v, pad, mode="constant", value=0.0))
            collated[key] = torch.stack(padded, dim=0)
        else:
            collated[key] = torch.tensor(values)
            
    return collated

def build_dataloader(dataset: VolumeDataset, cfg: LoaderConfig):
    """
    Create torch.utils.data.DataLoader.

    Example (from instructions):
        dataset = CustomDataset(data_resampled, labels_resampled)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    try:
        from torch.utils.data import DataLoader  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("PyTorch is required to build a DataLoader") from e

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=pad_collate_fn,
    )
