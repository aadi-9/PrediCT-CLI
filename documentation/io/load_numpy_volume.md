# `load_numpy_volume()`

**Source:** `src/predict/io.py`

Loads a NumPy `.npy` file from disk and wraps the array in a [`Volume`](Volume.md) with a default unit spacing.

---

## Signature

```python
def load_numpy_volume(path: Path) -> Volume
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | Path to the `.npy` file to load |

---

## Return Value

| Type | Description |
|---|---|
| [`Volume`](Volume.md) | Volume with the loaded array, `spacing_zyx=(1.0, 1.0, 1.0)`, and `meta=None` |

---

## Raises

| Exception | Condition |
|---|---|
| `FileNotFoundError` | `path` does not exist |
| `ValueError` | The file is not a valid NumPy array |

---

## Description

`load_numpy_volume()` calls `np.load(path)` and wraps the result in a `Volume`:

```python
array = np.load(path)
return Volume(array=array, spacing_zyx=(1.0, 1.0, 1.0), meta=None)
```

Because `.npy` files do not store physical spacing or DICOM metadata, the spacing is fixed at `(1.0, 1.0, 1.0)` mm — a nominal unit spacing. The `meta` field is always `None`.

---

## In the Data Pipeline

`load_numpy_volume()` is the I/O function used when a subject's `kind` is `"numpy"`, typically after a prior `export_processed=True` run of [`run_pipeline()`](../pipeline/run_pipeline.md) has already saved preprocessed volumes.

```
save_numpy_volume()
  └─► data/processed/patient_001.npy
        └─► SampleRecord(kind="numpy", image=Path("data/processed/patient_001.npy"))
              └─► default_load_volume(rec)
                    └─► load_numpy_volume(rec.image)   ← here
                          └─► Volume(array, spacing=(1,1,1), meta=None)
                                └─► VolumeDataset.__getitem__()
```

---

## Usage Example

```python
from pathlib import Path
from predict.io import load_numpy_volume

volume = load_numpy_volume(Path("data/processed/patient_001.npy"))

print(volume.array.shape)    # e.g. (128, 128, 128) if resampled to fixed shape
print(volume.array.dtype)    # as stored, typically float32
print(volume.spacing_zyx)    # (1.0, 1.0, 1.0)
print(volume.meta)           # None
```

---

## Notes

> **Warning:** The spacing `(1.0, 1.0, 1.0)` is a **nominal placeholder**, not the true physical spacing of the original scan. Do not use spacing-dependent calculations (e.g., `resample_volume()` in `"spacing"` mode) on volumes loaded this way without first setting the correct spacing.

- If you need to preserve spacing across export/load cycles, consider using NIfTI format ([`load_nifti_volume()`](load_nifti_volume.md)) instead.
- The function does not validate that the array is 3D. If the `.npy` file contains a 2D or 4D array, subsequent pipeline steps may fail with shape errors.

---

## Related

- [`save_numpy_volume()`](save_numpy_volume.md) — the corresponding save function
- [`Volume`](Volume.md) — the returned data structure
- [`default_load_volume()`](../dataset/default_load_volume.md) — dispatches to this function for `kind="numpy"`
- [`load_nifti_volume()`](load_nifti_volume.md) — alternative for formats that preserve spacing
