# `build_records_fallback()`

**Source:** `src/predict/pipeline.py`

Discovers subject directories under the raw data directory and synthesises a list of [`SampleRecord`](../dataset/SampleRecord.md) objects when no metadata CSV is available.

---

## Signature

```python
def build_records_fallback(
    paths: PathsConfig,
) -> tuple[list[SampleRecord], list[str]]
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `paths` | [`PathsConfig`](../config/PathsConfig.md) | The resolved filesystem paths configuration; specifically `paths.raw_dir` is scanned |

---

## Return Value

A 2-tuple:

| Position | Type | Description |
|---|---|---|
| `[0]` — `records` | `list[SampleRecord]` | One `SampleRecord` per subject subdirectory found |
| `[1]` — `warnings` | `list[str]` | Non-fatal warnings (e.g., empty raw directory notice) |

---

## Description

`build_records_fallback()` is used when [`run_pipeline()`](run_pipeline.md) is invoked without a `metadata_csv` argument and no CSV can be auto-discovered. It:

1. Calls [`discover_subject_dirs(paths.raw_dir)`](../io/discover_subject_dirs.md) to obtain a sorted list of all immediate subdirectories under `raw_dir`.
2. Assigns every discovered directory:
   - `subject_id` = directory name (stem)
   - `image` = absolute path to the subject directory
   - `label` = `0` (default; all subjects treated as the same class)
   - `kind` = `"dicom_series"` (assumes all data is DICOM)
3. Returns the list of records and any accumulated warnings.

If `raw_dir` is empty or does not exist, the returned records list will be empty and a warning will be included.

---

## In the Data Pipeline

```
run_pipeline(metadata_csv=None)
  └─► (no metadata CSV found)
        └─► build_records_fallback(paths)   ← here
              └─► list[SampleRecord]  (label=0, kind="dicom_series")
                    └─► stratified_split() → VolumeDataset → DataLoaders
```

---

## Usage Example

```python
from pathlib import Path
from predict.config import PathsConfig
from predict.pipeline import build_records_fallback

paths = PathsConfig.from_project_root(Path("/home/user/cardiac"))
records, warnings = build_records_fallback(paths)

for w in warnings:
    print("WARN:", w)

print(f"Found {len(records)} subjects")
for rec in records[:3]:
    print(rec.subject_id, rec.label, rec.kind)
```

Example output (with three subject directories):

```
Found 3 subjects
patient_001 0 dicom_series
patient_002 0 dicom_series
patient_003 0 dicom_series
```

---

## Notes

> **Warning:** All subjects are assigned `label=0`. This is appropriate for unsupervised or single-class pipelines but will produce meaningless stratification in classification tasks. For labelled datasets, always supply a metadata CSV.

> **Warning:** The fallback assumes `kind="dicom_series"`. If your data is in NumPy or NIfTI format, use a metadata CSV with the correct `kind` column instead.

- Unlike [`load_metadata_csv()`](load_metadata_csv.md), this function performs no path existence validation beyond what [`discover_subject_dirs()`](../io/discover_subject_dirs.md) provides.
- A warning is added if `raw_dir` does not exist, but the function still returns an empty list rather than raising.

---

## Related

- [`load_metadata_csv()`](load_metadata_csv.md) — preferred path; reads a curated metadata CSV
- [`discover_subject_dirs()`](../io/discover_subject_dirs.md) — the underlying directory scanner
- [`SampleRecord`](../dataset/SampleRecord.md) — dataclass populated for each discovered directory
- [`PathsConfig`](../config/PathsConfig.md) — provides `raw_dir` to this function
- [`run_pipeline()`](run_pipeline.md) — calls this function when no metadata CSV is present
