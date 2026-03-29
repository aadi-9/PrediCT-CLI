# `read_dicom_series()`

**Source:** `src/predict/io.py`

Reads a DICOM series folder using SimpleITK and returns a [`Volume`](Volume.md) with a `(Z, Y, X)` NumPy array, physical voxel spacing, and metadata.

---

## Signature

```python
def read_dicom_series(series_dir: Path) -> Volume
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `series_dir` | `Path` | Path to the directory containing `.dcm` files for a single subject/series |

---

## Return Value

| Type | Description |
|---|---|
| [`Volume`](Volume.md) | Volume with `array` shaped `(Z, Y, X)`, `spacing_zyx` in mm, and `meta` dict |

---

## Raises

| Exception | Condition |
|---|---|
| `FileNotFoundError` | No `.dcm` files are found in `series_dir`, or `series_dir` does not exist |
| `RuntimeError` | SimpleITK is not installed or is not importable |
| `RuntimeError` | SimpleITK fails to read the series (e.g., corrupted files, incomplete series) |

---

## Description

The function uses SimpleITK's GDCM-based DICOM reader:

1. Uses `SimpleITK.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir))` to discover the ordered list of DICOM slice files.
2. If no files are returned, raises `FileNotFoundError`.
3. Constructs a `SimpleITK.ImageSeriesReader`, loads the series, and calls `Execute()`.
4. Converts the 3D SimpleITK image to a NumPy array via `GetArrayFromImage()`, which yields `(Z, Y, X)` axis order.
5. Extracts `GetSpacing()` (X, Y, Z order in SimpleITK) and reverses it to `(Z, Y, X)` for the `Volume`.
6. Collects origin, direction, and other header fields into the `meta` dict.
7. Returns `Volume(array=array.astype(float32), spacing_zyx=spacing_zyx, meta=meta)`.

---

## Output `meta` Dictionary Keys

| Key | Type | Description |
|---|---|---|
| `origin` | `tuple` | World-space origin `(x, y, z)` from the DICOM header |
| `direction` | `tuple` | Direction cosines (9 values, row-major) |
| `size_xyz` | `tuple` | Original image size `(X, Y, Z)` in voxels |
| `spacing_xyz` | `tuple` | Original spacing `(X, Y, Z)` in mm (pre-reversal) |

---

## In the Data Pipeline

`read_dicom_series()` is the primary I/O function for DICOM subjects. It is called by [`default_load_volume()`](../dataset/default_load_volume.md) and directly inside [`run_pipeline()`](../pipeline/run_pipeline.md) during the `export_processed=True` phase.

```
SampleRecord(kind="dicom_series")
  └─► default_load_volume(rec)
        └─► read_dicom_series(rec.image)   ← here
              └─► Volume(array, spacing_zyx, meta)
                    └─► apply_hu_window() → resample_volume()
```

---

## Usage Example

```python
from pathlib import Path
from predict.io import read_dicom_series

volume = read_dicom_series(Path("data/raw/patient_001"))

print(volume.array.shape)    # e.g. (394, 512, 512)
print(volume.array.dtype)    # float32
print(volume.spacing_zyx)    # e.g. (0.625, 0.488, 0.488)
print(volume.meta["origin"]) # e.g. (-130.0, -200.0, -400.0)
```

---

## Notes

> **Warning:** SimpleITK is a required dependency for this function. If SimpleITK is not installed, a `RuntimeError` is raised. Install with `pip install SimpleITK`.

> **Warning:** `read_dicom_series()` reads **all** `.dcm` files in the directory as a single series. Directories containing multiple DICOM series (multiple series UIDs) will be read using only the first series found by `GetGDCMSeriesFileNames()`. Pre-sort or split multi-series directories before using PrediCT.

- The output array dtype is `float32`. Raw DICOM pixel values (typically `int16`) are cast during the SimpleITK read.
- For validation without loading pixel data, see [`validate_dicom_series_dir()`](../validate/validate_dicom_series_dir.md).

---

## Related

- [`Volume`](Volume.md) — the returned data structure
- [`validate_dicom_series_dir()`](../validate/validate_dicom_series_dir.md) — validates readability without loading pixel data
- [`default_load_volume()`](../dataset/default_load_volume.md) — dispatch function that calls this for `kind="dicom_series"`
- [`save_numpy_volume()`](save_numpy_volume.md) — saves a loaded volume to disk as `.npy`
- [`load_numpy_volume()`](load_numpy_volume.md) — loads a previously exported `.npy` file
