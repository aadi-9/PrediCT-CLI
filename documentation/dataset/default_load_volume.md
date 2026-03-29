# `default_load_volume()`

**Source:** `src/predict/dataset.py`

Dispatches to the appropriate volume loader based on the `kind` field of a [`SampleRecord`](SampleRecord.md), returning a [`Volume`](../io/Volume.md).

---

## Signature

```python
def default_load_volume(rec: SampleRecord) -> Volume
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `rec` | [`SampleRecord`](SampleRecord.md) | Record describing the subject's data location and format |

---

## Return Value

| Type | Description |
|---|---|
| [`Volume`](../io/Volume.md) | Loaded volume with `array`, `spacing_zyx`, and `meta` populated according to the format |

---

## Raises

| Exception | Condition |
|---|---|
| `ValueError` | `rec.kind` is not a recognised value |

---

## Dispatch Table

| `rec.kind` | Calls | Notes |
|---|---|---|
| `"dicom_series"` | [`read_dicom_series(rec.image)`](../io/read_dicom_series.md) | `rec.image` must be a directory containing `.dcm` files |
| `"numpy"` | [`load_numpy_volume(rec.image)`](../io/load_numpy_volume.md) | `rec.image` must be a `.npy` file; spacing defaults to `(1.0, 1.0, 1.0)` |
| `"nifti"` / `"nii"` / `"nifti_gz"` | [`load_nifti_volume(rec.image)`](../io/load_nifti_volume.md) | `rec.image` must be a `.nii` or `.nii.gz` file |

Any other value for `rec.kind` raises a `ValueError`.

---

## In the Data Pipeline

`default_load_volume()` is the default `load_fn` of [`VolumeDataset`](VolumeDataset.md). It is called lazily in `__getitem__()` when a training/validation/test batch is assembled.

```
VolumeDataset.__getitem__(idx)
  └─► load_fn(records[idx])
        └─► default_load_volume(rec)   ← here
              ├─► read_dicom_series()     (kind="dicom_series")
              ├─► load_numpy_volume()     (kind="numpy")
              └─► load_nifti_volume()     (kind="nifti" / "nii" / "nifti_gz")
                    └─► Volume(array, spacing_zyx, meta)
```

---

## Usage Example

```python
from pathlib import Path
from predict.dataset import SampleRecord, default_load_volume

# Load a DICOM series
rec = SampleRecord(
    subject_id="patient_001",
    image=Path("data/raw/patient_001"),
    label=0,
    kind="dicom_series",
)
volume = default_load_volume(rec)
print(volume.array.shape)

# Load a preprocessed NumPy file
rec_np = SampleRecord(
    subject_id="patient_001",
    image=Path("data/processed/patient_001.npy"),
    label=0,
    kind="numpy",
)
volume_np = default_load_volume(rec_np)
```

---

## Notes

- `default_load_volume()` is passed as the default `load_fn` when constructing a `VolumeDataset`. You can supply a custom loading function to `VolumeDataset` if you need different behaviour (e.g., loading from a remote store, applying on-the-fly augmentation at load time).
- The function does not apply any preprocessing; that is handled separately by `preprocess_fn` in `VolumeDataset`.

---

## Related

- [`SampleRecord`](SampleRecord.md) — the input record describing the data location and kind
- [`VolumeDataset`](VolumeDataset.md) — uses `default_load_volume` as its default `load_fn`
- [`read_dicom_series()`](../io/read_dicom_series.md) — called for DICOM data
- [`load_numpy_volume()`](../io/load_numpy_volume.md) — called for NumPy data
- [`load_nifti_volume()`](../io/load_nifti_volume.md) — called for NIfTI data
- [`Volume`](../io/Volume.md) — the returned data structure
