# `validate_dicom_series_dir()`

**Source:** `src/predict/validate.py`

Checks whether a single DICOM series directory is readable, returning a structured result that indicates success, file count, and a human-readable reason string.

---

## Signature

```python
def validate_dicom_series_dir(
    series_dir: Path,
    *,
    mode: Literal["fast", "shallow", "deep"] = "shallow",
) -> tuple[bool, int, str]
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `series_dir` | `Path` | — | Path to the directory containing `.dcm` files for a single subject/series |
| `mode` | `str` | `"shallow"` | Depth of validation check; see modes table below |

---

## Return Value

A 3-tuple:

| Position | Type | Description |
|---|---|---|
| `[0]` — `ok` | `bool` | `True` if the series is considered valid, `False` otherwise |
| `[1]` — `num_files` | `int` | Number of `.dcm` files found in the directory |
| `[2]` — `reason` | `str` | Human-readable explanation; `"ok"` on success, error description on failure |

---

## Validation Modes

| Mode | What it checks | Speed | Requires SimpleITK |
|---|---|---|---|
| `"fast"` | Counts `.dcm` files; valid if count ≥ 1 | Very fast | No |
| `"shallow"` | Uses SimpleITK's GDCM series reader to enumerate DICOM series UIDs; valid if at least one series is found | Fast | **Yes** |
| `"deep"` | Fully reads the image array via SimpleITK; valid if the read completes without error | Slow | **Yes** |

### Mode selection guide

- Use `"fast"` for a quick sanity check on large datasets where read speed is critical.
- Use `"shallow"` (default) for a reliable check that the DICOM files form a valid series without loading pixel data.
- Use `"deep"` when you need to guarantee that pixel data is readable (e.g., detecting corrupted or incomplete files).

---

## Failure Reasons

Common `reason` strings returned on failure:

| Reason | Description |
|---|---|
| `"no .dcm files found"` | Directory contains no `.dcm` files |
| `"no DICOM series found by SimpleITK"` | `shallow` mode found files but SimpleITK could not construct a valid series |
| `"SimpleITK read failed: <error>"` | `deep` mode encountered an exception while reading pixel data |
| `"SimpleITK not available"` | `shallow`/`deep` mode requested but SimpleITK is not installed |
| `"directory does not exist"` | `series_dir` path does not exist on disk |

---

## In the Data Pipeline

`validate_dicom_series_dir()` is called once per DICOM row by [`validate_metadata_csv()`](validate_metadata_csv.md), which aggregates the results into the clean CSV and the validation report.

```
validate_metadata_csv(metadata_csv, ...)
  └─► for each row with kind == "dicom_series":
        validate_dicom_series_dir(series_dir, mode=mode)   ← here
          └─► (ok, num_files, reason)
                ├─► ok=True  → written to metadata_clean.csv
                └─► ok=False → written to dicom_validation_report.csv only
```

---

## Usage Example

```python
from pathlib import Path
from predict.validate import validate_dicom_series_dir

series = Path("data/raw/patient_001")

# Quick check
ok, n, reason = validate_dicom_series_dir(series, mode="fast")
print(ok, n, reason)  # True 312 ok

# Full shallow check
ok, n, reason = validate_dicom_series_dir(series, mode="shallow")
if not ok:
    print(f"Invalid series: {reason}")

# Deep read check
ok, n, reason = validate_dicom_series_dir(series, mode="deep")
```

---

## Notes

> **Warning:** `"shallow"` and `"deep"` modes require `SimpleITK` to be installed. If SimpleITK is not available, these modes return `(False, 0, "SimpleITK not available")` rather than raising an exception.

- The function is **pure** with respect to the filesystem — it never modifies any files.
- For large DICOM series (thousands of slices), `"deep"` mode can take several seconds per series. Consider `"shallow"` for initial dataset auditing.
- The `num_files` count reflects `.dcm` files only; other files in the directory (e.g., `.txt`, `.xml`) are not counted.

---

## Related

- [`validate_metadata_csv()`](validate_metadata_csv.md) — orchestrates calls to this function across an entire dataset
- [`read_dicom_series()`](../io/read_dicom_series.md) — actually loads pixel data (equivalent in scope to `"deep"` validation)
- [`generate_metadata_csv()`](../metadata/generate_metadata_csv.md) — produces the metadata CSV that feeds `validate_metadata_csv()`
