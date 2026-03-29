# `discover_subject_dirs()`

**Source:** `src/predict/io.py`

Returns a sorted list of all immediate subdirectories within a raw data directory, used to enumerate subjects when no metadata CSV is available.

---

## Signature

```python
def discover_subject_dirs(raw_dir: Path) -> list[Path]
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `raw_dir` | `Path` | Directory to scan for subject subdirectories |

---

## Return Value

| Type | Description |
|---|---|
| `list[Path]` | Sorted list of `Path` objects, one per immediate subdirectory of `raw_dir`; empty list if `raw_dir` does not exist or contains no subdirectories |

---

## Description

The function:

1. Checks whether `raw_dir` exists; returns `[]` immediately if not.
2. Iterates `raw_dir.iterdir()` and filters for entries where `entry.is_dir()`.
3. Returns the results sorted alphabetically.

Only **immediate** (first-level) subdirectories are returned. Files at the root of `raw_dir` and deeper nested directories are ignored.

---

## In the Data Pipeline

`discover_subject_dirs()` is the directory-scanning primitive used by [`build_records_fallback()`](../pipeline/build_records_fallback.md) when no metadata CSV exists.

```
build_records_fallback(paths)
  └─► discover_subject_dirs(paths.raw_dir)   ← here
        └─► [Path("data/raw/patient_001"), Path("data/raw/patient_002"), ...]
              └─► [SampleRecord(subject_id="patient_001", ...), ...]
```

---

## Usage Example

```python
from pathlib import Path
from predict.io import discover_subject_dirs

dirs = discover_subject_dirs(Path("data/raw"))
for d in dirs:
    print(d.name)
# patient_001
# patient_002
# patient_003

# Non-existent directory returns empty list
dirs = discover_subject_dirs(Path("/does/not/exist"))
print(dirs)  # []
```

Typical directory layout expected:

```
data/raw/
├── patient_001/
│   ├── slice_001.dcm
│   └── slice_002.dcm
├── patient_002/
│   └── ...
└── patient_003/
    └── ...
```

---

## Notes

- The function silently returns an empty list if `raw_dir` does not exist, rather than raising `FileNotFoundError`. Callers that need to distinguish "missing" from "empty" should check `raw_dir.exists()` separately.
- Sorting is lexicographic (Python's default `Path` sort), which works well for zero-padded numeric IDs (e.g., `patient_001` < `patient_002`).
- Hidden directories (names starting with `.`) are included unless the OS excludes them from `iterdir()`.

---

## Related

- [`build_records_fallback()`](../pipeline/build_records_fallback.md) — primary caller; converts discovered paths to `SampleRecord` objects
- [`generate_metadata_csv()`](../metadata/generate_metadata_csv.md) — also scans subject directories but applies additional filtering
- [`read_dicom_series()`](read_dicom_series.md) — reads the DICOM contents of each discovered directory
