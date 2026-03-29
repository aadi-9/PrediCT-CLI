# `generate_metadata_csv()`

**Source:** `src/predict/metadata.py`

Scans a raw data directory for subject subdirectories and writes a metadata CSV that serves as the input inventory for the rest of the PrediCT pipeline.

---

## Signature

```python
def generate_metadata_csv(
    raw_dir:       Path,
    out_csv:       Path,
    *,
    default_label: int = 0,
    kind:          str = "dicom_series",
) -> Path
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `raw_dir` | `Path` | — | Directory to scan; each immediate subdirectory is treated as one subject |
| `out_csv` | `Path` | — | File path where the CSV will be written; parent directories are created automatically |
| `default_label` | `int` | `0` | Integer class label assigned to every subject (used when ground-truth labels are not yet known) |
| `kind` | `str` | `"dicom_series"` | Data type of the subject images; controls which directories are included |

---

## Return Value

| Type | Description |
|---|---|
| `Path` | Absolute path to the written CSV file (`out_csv`) |

---

## Raises

| Exception | Condition |
|---|---|
| `ValueError` | `kind` is not one of the supported values |

---

## Supported `kind` Values

| Value | Inclusion rule |
|---|---|
| `"dicom_series"` | Only subdirectories that contain at least one `.dcm` file |
| `"numpy"` | All subdirectories |
| `"nifti"` / `"nii"` / `"nifti_gz"` | All subdirectories |

For `"dicom_series"`, the function uses an internal helper `_has_dicom_files(path)` to filter out subject directories that contain no `.dcm` files, avoiding empty entries in the CSV.

---

## Output CSV Schema

The generated CSV contains one row per subject:

| Column | Type | Description |
|---|---|---|
| `subject_id` | `str` | Name of the subject subdirectory |
| `image` | `str` | Absolute path to the subject's data directory |
| `label` | `int` | Class label (`default_label` for all rows at this stage) |
| `kind` | `str` | The `kind` argument passed to the function |

---

## In the Data Pipeline

`generate_metadata_csv()` is the **first** step: it transforms an unstructured collection of directories into a tabular inventory.

```
Raw DICOM data (data/raw/<subject>/)
  └─► generate_metadata_csv(raw_dir, out_csv)   ← here
        └─► metadata_all.csv
              └─► validate_metadata_csv(metadata_csv=metadata_all.csv, ...)
                    └─► metadata_clean.csv
                          └─► run_pipeline(metadata_csv=metadata_clean.csv, ...)
```

---

## Usage Example

```python
from pathlib import Path
from predict.metadata import generate_metadata_csv

csv_path = generate_metadata_csv(
    raw_dir=Path("data/raw"),
    out_csv=Path("outputs/metadata_all.csv"),
    default_label=0,
    kind="dicom_series",
)
print(f"Wrote metadata to {csv_path}")
```

Example output CSV:

```csv
subject_id,image,label,kind
patient_001,/project/data/raw/patient_001,0,dicom_series
patient_002,/project/data/raw/patient_002,0,dicom_series
patient_003,/project/data/raw/patient_003,0,dicom_series
```

### From the CLI

```bash
predict make-metadata \
  --raw-dir data/raw \
  --out-csv outputs/metadata_all.csv \
  --label 0 \
  --kind dicom_series
```

---

## Notes

> **Warning:** All subjects are assigned `default_label=0` at this stage. You must manually edit the CSV (or provide a labelled CSV from another source) to assign meaningful labels before running the pipeline with a classification task.

- The function creates parent directories of `out_csv` with `mkdir(parents=True, exist_ok=True)`.
- Files at the root of `raw_dir` (i.e., not in a subdirectory) are ignored; only first-level subdirectories are enumerated.
- The `image` column contains the **absolute** path to the subject directory at the time of generation. Moving the project directory will break these paths; use the `--raw-dir` override in subsequent steps as a workaround.

---

## Related

- [`validate_metadata_csv()`](../validate/validate_metadata_csv.md) — next step: validates each DICOM series listed in the CSV
- [`load_metadata_csv()`](../pipeline/load_metadata_csv.md) — reads the cleaned CSV into `SampleRecord` objects
- [`SampleRecord`](../dataset/SampleRecord.md) — the in-memory representation of each CSV row
- [`discover_subject_dirs()`](../io/discover_subject_dirs.md) — lower-level helper that lists subject subdirectories
