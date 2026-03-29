# `Volume`

**Source:** `src/predict/io.py`

Frozen dataclass that represents a 3D medical imaging volume, combining the voxel array, physical spacing, and optional metadata into a single, immutable object.

---

## Signature

```python
@dataclass(frozen=True)
class Volume:
    array:       np.ndarray
    spacing_zyx: tuple[float, float, float] | None
    meta:        dict[str, Any] | None
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `array` | `np.ndarray` | 3D voxel array shaped `(Z, Y, X)`; dtype varies by source (typically `float32` after preprocessing, `int16` or `float32` from DICOM) |
| `spacing_zyx` | `tuple[float, float, float] \| None` | Physical voxel size in millimetres along `(Z, Y, X)` axes; `None` if spacing is unknown |
| `meta` | `dict[str, Any] \| None` | Optional dictionary of source metadata (e.g., SimpleITK image metadata, DICOM header fields, origin, direction); `None` if not available |

---

## Axis Convention

All array dimensions follow the **(Z, Y, X)** convention throughout PrediCT:

```
Z → slice / axial direction (inferior → superior)
Y → coronal direction (anterior → posterior)
X → sagittal direction (left → right)
```

This matches the output of `SimpleITK`'s `GetArrayFromImage()` after axis transposition.

---

## Typical Values by Source

| Source function | `array` dtype | `spacing_zyx` | `meta` |
|---|---|---|---|
| [`read_dicom_series()`](read_dicom_series.md) | `float32` | From DICOM header (mm) | SimpleITK metadata |
| [`load_numpy_volume()`](load_numpy_volume.md) | As stored | `(1.0, 1.0, 1.0)` | `None` |
| [`load_nifti_volume()`](load_nifti_volume.md) | `float32` | From NIfTI header (mm) | SimpleITK metadata |
| After [`apply_hu_window()`](../preprocess/apply_hu_window.md) | `float32` | Preserved | Preserved |
| After [`resample_volume()`](../preprocess/resample_volume.md) | `float32` | Updated | Preserved |

---

## In the Data Pipeline

`Volume` is the primary data carrier between I/O, preprocessing, and dataset layers.

```
read_dicom_series() / load_numpy_volume() / load_nifti_volume()
  └─► Volume(array, spacing_zyx, meta)   ← created here
        └─► apply_hu_window()  →  Volume (new array)
              └─► resample_volume()  →  Volume (new array + spacing)
                    └─► VolumeDataset.__getitem__()
                          └─► torch.Tensor(1, Z, Y, X)
```

---

## Usage Example

```python
import numpy as np
from predict.io import Volume

# Construct directly
vol = Volume(
    array=np.zeros((64, 128, 128), dtype=np.float32),
    spacing_zyx=(2.5, 0.977, 0.977),
    meta={"origin": (0.0, 0.0, 0.0)},
)

print(vol.array.shape)    # (64, 128, 128)
print(vol.spacing_zyx)    # (2.5, 0.977, 0.977)

# Access from DICOM
from predict.io import read_dicom_series
from pathlib import Path

vol = read_dicom_series(Path("data/raw/patient_001"))
print(vol.array.shape)
print(vol.spacing_zyx)
```

---

## Notes

- `Volume` is `frozen=True`, meaning fields cannot be reassigned after construction. All functions that "modify" a volume return a **new** `Volume` instance.
- When `spacing_zyx` is `None`, functions that require physical spacing (e.g., `resample_volume()` in `"spacing"` mode) will raise a `ValueError`. Always load from a source that provides spacing, or use `mode="shape"` resampling.
- The `meta` dict is not standardised — its contents depend on the loading function. Do not rely on specific keys being present without checking first.

---

## Related

- [`read_dicom_series()`](read_dicom_series.md) — creates `Volume` from a DICOM folder
- [`load_numpy_volume()`](load_numpy_volume.md) — creates `Volume` from a `.npy` file
- [`load_nifti_volume()`](load_nifti_volume.md) — creates `Volume` from a NIfTI file
- [`save_numpy_volume()`](save_numpy_volume.md) — saves `volume.array` to a `.npy` file
- [`apply_hu_window()`](../preprocess/apply_hu_window.md) — returns a new `Volume` with windowed array
- [`resample_volume()`](../preprocess/resample_volume.md) — returns a new `Volume` at a different resolution
- [`VolumeDataset`](../dataset/VolumeDataset.md) — converts `Volume` to a PyTorch tensor
