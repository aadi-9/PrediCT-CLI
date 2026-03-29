# `save_numpy_volume()`

**Source:** `src/predict/io.py`

Saves only the `array` component of a [`Volume`](Volume.md) to a NumPy `.npy` file, creating parent directories as needed.

---

## Signature

```python
def save_numpy_volume(
    volume:   Volume,
    out_path: Path,
) -> None
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `volume` | [`Volume`](Volume.md) | Source volume; only `volume.array` is saved |
| `out_path` | `Path` | Destination file path; should end with `.npy` |

---

## Return Value

`None`. The function writes to disk as a side effect.

---

## Description

`save_numpy_volume()`:

1. Creates the parent directory of `out_path` with `mkdir(parents=True, exist_ok=True)`.
2. Calls `np.save(out_path, volume.array)`.

> **Note:** Only `volume.array` is persisted. `spacing_zyx` and `meta` are **not** saved. When the file is later loaded with [`load_numpy_volume()`](load_numpy_volume.md), spacing defaults to `(1.0, 1.0, 1.0)` and `meta` is `None`.

---

## In the Data Pipeline

`save_numpy_volume()` is called during the `export_processed=True` phase of [`run_pipeline()`](../pipeline/run_pipeline.md), after HU windowing and resampling.

```
apply_hu_window() → resample_volume()
  └─► Volume(preprocessed array)
        └─► save_numpy_volume(volume, out_path)   ← here
              └─► data/processed/<subject_id>.npy
                    └─► load_numpy_volume() → VolumeDataset
```

---

## Usage Example

```python
from pathlib import Path
from predict.io import read_dicom_series, save_numpy_volume
from predict.config import HUWindowConfig, ResampleConfig
from predict.preprocess import apply_hu_window, resample_volume

volume = read_dicom_series(Path("data/raw/patient_001"))
volume = apply_hu_window(volume, HUWindowConfig())
volume = resample_volume(volume, ResampleConfig())

save_numpy_volume(volume, Path("data/processed/patient_001.npy"))
# Creates data/processed/ if it doesn't exist
```

---

## Notes

- The `.npy` file stores only the raw array; no spacing or DICOM metadata is included.
- If `out_path` already exists, it is **silently overwritten**.
- Use this function only for preprocessed (float32, windowed, resampled) arrays; the downstream loader [`load_numpy_volume()`](load_numpy_volume.md) assigns dummy spacing `(1.0, 1.0, 1.0)`.

---

## Related

- [`load_numpy_volume()`](load_numpy_volume.md) — the corresponding load function
- [`Volume`](Volume.md) — the dataclass being serialised
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls this function when `export_processed=True`
- [`resample_volume()`](../preprocess/resample_volume.md) — typically called just before saving
