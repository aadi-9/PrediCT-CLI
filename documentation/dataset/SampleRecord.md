# `SampleRecord`

**Source:** `src/predict/dataset.py`

Frozen dataclass that represents a single subject entry in the dataset: an identifier, the path to the subject's imaging data, a class label, and a data-kind tag.

---

## Signature

```python
@dataclass(frozen=True)
class SampleRecord:
    subject_id: str
    image:      Path
    label:      int
    kind:       str = "dicom_series"
```

---

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `subject_id` | `str` | — | Unique string identifier for this subject (e.g., `"patient_001"`) |
| `image` | `Path` | — | Absolute path to the subject's imaging data (directory for DICOM, file path for NumPy/NIfTI) |
| `label` | `int` | — | Integer class label (e.g., `0` = control, `1` = disease) |
| `kind` | `str` | `"dicom_series"` | Data format tag; controls which loader is used at runtime |

---

## Supported `kind` Values

| Value | Loaded by |
|---|---|
| `"dicom_series"` | [`read_dicom_series()`](../io/read_dicom_series.md) |
| `"numpy"` | [`load_numpy_volume()`](../io/load_numpy_volume.md) |
| `"nifti"` / `"nii"` / `"nifti_gz"` | [`load_nifti_volume()`](../io/load_nifti_volume.md) |

---

## In the Data Pipeline

`SampleRecord` objects flow through every stage of the pipeline after metadata loading.

```
load_metadata_csv() or build_records_fallback()
  └─► list[SampleRecord]   ← created here
        └─► stratified_split()
              └─► oversample_minority()  (train only)
                    └─► VolumeDataset(records=...)
                          └─► __getitem__(idx) → default_load_volume(records[idx])
```

---

## Usage Example

```python
from pathlib import Path
from predict.dataset import SampleRecord

# DICOM subject
rec = SampleRecord(
    subject_id="patient_001",
    image=Path("/data/raw/patient_001"),
    label=1,
    kind="dicom_series",
)

# NumPy subject (after export_processed)
rec_np = SampleRecord(
    subject_id="patient_001",
    image=Path("/data/processed/patient_001.npy"),
    label=1,
    kind="numpy",
)

# NIfTI subject
rec_nii = SampleRecord(
    subject_id="patient_001",
    image=Path("/data/raw/patient_001/ct.nii.gz"),
    label=0,
    kind="nifti",
)
```

---

## Notes

- `SampleRecord` is `frozen=True`; fields cannot be modified after construction. To update a record (e.g., to point to an exported `.npy` file), construct a new `SampleRecord` with the desired values.
- The `image` path should be **absolute** for reliable resolution across different working directories.
- `subject_id` values must be unique within a dataset to prevent deduplication warnings in [`load_metadata_csv()`](../pipeline/load_metadata_csv.md).

---

## Related

- [`load_metadata_csv()`](../pipeline/load_metadata_csv.md) — constructs `SampleRecord` objects from a CSV
- [`build_records_fallback()`](../pipeline/build_records_fallback.md) — constructs `SampleRecord` objects by directory scanning
- [`default_load_volume()`](default_load_volume.md) — dispatches to the appropriate loader based on `rec.kind`
- [`VolumeDataset`](VolumeDataset.md) — holds a list of `SampleRecord` objects and loads them on demand
