# `load_nifti_volume()`

**Source:** `src/predict/io.py`

Reads a NIfTI (`.nii` or `.nii.gz`) file using SimpleITK and returns a [`Volume`](Volume.md) with a `(Z, Y, X)` array, physical spacing, and metadata.

---

## Signature

```python
def load_nifti_volume(path: Path) -> Volume
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | Path to the `.nii` or `.nii.gz` file |

---

## Return Value

| Type | Description |
|---|---|
| [`Volume`](Volume.md) | Volume with `array` shaped `(Z, Y, X)`, physical `spacing_zyx` in mm, and `meta` dict |

---

## Raises

| Exception | Condition |
|---|---|
| `FileNotFoundError` | `path` does not exist |
| `RuntimeError` | SimpleITK is not installed |
| `RuntimeError` | SimpleITK fails to read the file |

---

## Description

The function uses `SimpleITK.ReadImage(str(path))` to load the NIfTI file, then:

1. Converts the image to a NumPy array via `GetArrayFromImage()`, yielding `(Z, Y, X)` axis order.
2. Extracts `GetSpacing()` (X, Y, Z order in SimpleITK) and reverses to `(Z, Y, X)`.
3. Collects `origin`, `direction`, `size_xyz`, and `spacing_xyz` into the `meta` dict.
4. Returns `Volume(array=array.astype(float32), spacing_zyx=spacing_zyx, meta=meta)`.

---

## Supported Formats

| Extension | Description |
|---|---|
| `.nii` | Uncompressed NIfTI-1 |
| `.nii.gz` | Gzip-compressed NIfTI-1 |

Other formats supported by SimpleITK (e.g., MetaImage `.mha`) may also work but are not officially tested in PrediCT.

---

## In the Data Pipeline

`load_nifti_volume()` is called by [`default_load_volume()`](../dataset/default_load_volume.md) when a `SampleRecord` has `kind="nifti"` or `kind="nii"`.

```
SampleRecord(kind="nifti", image=Path("data/raw/patient_001/ct.nii.gz"))
  └─► default_load_volume(rec)
        └─► load_nifti_volume(rec.image)   ← here
              └─► Volume(array, spacing_zyx, meta)
                    └─► apply_hu_window() → resample_volume()
```

---

## Usage Example

```python
from pathlib import Path
from predict.io import load_nifti_volume

volume = load_nifti_volume(Path("data/raw/patient_001/ct.nii.gz"))

print(volume.array.shape)    # e.g. (400, 512, 512)
print(volume.array.dtype)    # float32
print(volume.spacing_zyx)    # e.g. (0.625, 0.488, 0.488)
print(volume.meta["origin"]) # e.g. (-130.0, -200.0, -400.0)
```

---

## Notes

> **Warning:** Requires `SimpleITK`. Install with `pip install SimpleITK`.

- Unlike [`load_numpy_volume()`](load_numpy_volume.md), NIfTI loading preserves physical spacing, making it compatible with `resample_volume()` in `"spacing"` mode.
- The output array is cast to `float32`; the original NIfTI data type (e.g., `int16`, `uint8`) is not preserved.
- For large compressed `.nii.gz` files, the decompression step may take a few seconds.

---

## Related

- [`Volume`](Volume.md) — the returned data structure
- [`default_load_volume()`](../dataset/default_load_volume.md) — dispatches to this function for `kind="nifti"` / `kind="nii"`
- [`read_dicom_series()`](read_dicom_series.md) — analogous function for DICOM input
- [`load_numpy_volume()`](load_numpy_volume.md) — analogous function for NumPy input (no spacing metadata)
