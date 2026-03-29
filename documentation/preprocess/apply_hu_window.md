# `apply_hu_window()`

**Source:** `src/predict/preprocess.py`

Applies HU windowing to a [`Volume`](../io/Volume.md) using a [`HUWindowConfig`](../config/HUWindowConfig.md), returning a new `Volume` with a normalised `float32` array.

---

## Signature

```python
def apply_hu_window(
    volume: Volume,
    cfg:    HUWindowConfig,
) -> Volume
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `volume` | [`Volume`](../io/Volume.md) | Input volume with a 3D `(Z, Y, X)` array (typically raw HU values) |
| `cfg` | [`HUWindowConfig`](../config/HUWindowConfig.md) | Window configuration specifying `lower` and `upper` HU bounds |

---

## Return Value

| Type | Description |
|---|---|
| [`Volume`](../io/Volume.md) | New `Volume` with windowed `float32` array; `spacing_zyx` and `meta` are preserved from the input |

---

## Description

`apply_hu_window()` is the high-level wrapper around [`hu_windowing()`](hu_windowing.md). It:

1. Extracts `volume.array`, `cfg.lower`, and `cfg.upper`.
2. Calls `hu_windowing(volume.array, cfg.lower, cfg.upper)`.
3. Returns a new `Volume(array=windowed_array, spacing_zyx=volume.spacing_zyx, meta=volume.meta)`.

The original `volume` is **not** modified (`Volume` is a frozen dataclass).

---

## In the Data Pipeline

`apply_hu_window()` is called inside [`run_pipeline()`](../pipeline/run_pipeline.md) for each subject volume, either eagerly (when `export_processed=True`) or lazily inside `VolumeDataset.__getitem__()`.

```
read_dicom_series(series_dir)
  └─► Volume(array[HU], spacing_zyx, meta)
        └─► apply_hu_window(volume, hu_cfg)   ← here
              └─► Volume(array[0..1], spacing_zyx, meta)
                    └─► resample_volume(volume, resample_cfg)
```

---

## Usage Example

```python
from predict.config import HUWindowConfig
from predict.preprocess import apply_hu_window
from predict.io import read_dicom_series
from pathlib import Path

volume = read_dicom_series(Path("data/raw/patient_001"))
cfg = HUWindowConfig(lower=-200.0, upper=400.0)

windowed = apply_hu_window(volume, cfg)

print(windowed.array.dtype)   # float32
print(windowed.array.min())   # 0.0
print(windowed.array.max())   # 1.0
print(windowed.spacing_zyx)   # preserved from original
```

---

## Notes

- The returned `Volume` shares the same `spacing_zyx` and `meta` as the input; only the `array` is replaced.
- `Volume` is frozen, so the original is never mutated.
- The output array dtype is always `float32`.

---

## Related

- [`hu_windowing()`](hu_windowing.md) — the underlying NumPy operation
- [`HUWindowConfig`](../config/HUWindowConfig.md) — configuration dataclass
- [`Volume`](../io/Volume.md) — the dataclass wrapping the array
- [`resample_volume()`](resample_volume.md) — next preprocessing step after HU windowing
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls this function per subject
