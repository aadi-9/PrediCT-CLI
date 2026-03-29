# `load_metadata_csv()`

**Source:** `src/predict/pipeline.py`

Reads a validated metadata CSV and converts each row into a [`SampleRecord`](../dataset/SampleRecord.md), resolving image paths to absolute filesystem paths and collecting non-fatal warnings.

---

## Signature

```python
def load_metadata_csv(
    path:         Path,
    project_root: Path,
    raw_dir:      Path,
) -> tuple[list[SampleRecord], list[str]]
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | Path to the metadata CSV file |
| `project_root` | `Path` | Absolute project root; used as the base for resolving relative image paths |
| `raw_dir` | `Path` | Raw DICOM base directory; used as a secondary base when an image path is relative |

---

## Return Value

A 2-tuple:

| Position | Type | Description |
|---|---|---|
| `[0]` — `records` | `list[SampleRecord]` | One `SampleRecord` per valid row |
| `[1]` — `warnings` | `list[str]` | Human-readable non-fatal issues discovered during loading |

---

## CSV Column Requirements

The CSV must contain these columns:

| Column | Type | Description |
|---|---|---|
| `subject_id` | `str` | Unique subject identifier |
| `image` | `str` | Path to subject data directory/file |
| `label` | `int` | Class label |
| `kind` | `str` | Data type (`dicom_series`, `numpy`, `nifti`, etc.) |

---

## Validation and Filtering

The function applies the following checks to each row and **skips** invalid rows (appending a warning instead of raising):

| Check | Warning text |
|---|---|
| Missing or empty `subject_id` | `"Row N: missing subject_id, skipping"` |
| Missing or empty `image` | `"Row N: missing image path, skipping"` |
| Non-integer `label` | `"Row N: label is not an integer, skipping"` |
| Unsupported `kind` | `"Row N: unsupported kind '<kind>', skipping"` |
| Duplicate `subject_id` | `"Duplicate subject_id '<id>', skipping later occurrence"` |
| Resolved path does not exist | `"Row N: resolved path '<path>' does not exist, skipping"` |

---

## Path Resolution Logic

For each `image` value in the CSV:

1. If the path is **absolute** and exists → use as-is.
2. If the path is **relative**, try joining to `project_root` → if that exists, use it.
3. If still not found, try joining to `raw_dir` → if that exists, use it.
4. If none of the above resolves to an existing path → skip with a warning.

---

## In the Data Pipeline

```
validate_metadata_csv()
  └─► metadata_clean.csv
        └─► load_metadata_csv(path, project_root, raw_dir)   ← here
              └─► list[SampleRecord]
                    └─► stratified_split(items=records, labels=...)
                          └─► VolumeDataset / run_pipeline()
```

---

## Usage Example

```python
from pathlib import Path
from predict.pipeline import load_metadata_csv

records, warnings = load_metadata_csv(
    path=Path("outputs/metadata_clean.csv"),
    project_root=Path("/home/user/cardiac"),
    raw_dir=Path("/home/user/cardiac/data/raw"),
)

for w in warnings:
    print("WARN:", w)

print(f"Loaded {len(records)} records")
for rec in records[:3]:
    print(rec.subject_id, rec.label, rec.kind)
```

---

## Notes

- Rows skipped due to validation issues are **not** included in the returned list; they only appear as warning strings.
- The function does not validate whether the DICOM series is actually readable — that is handled upstream by [`validate_metadata_csv()`](../validate/validate_metadata_csv.md).
- Duplicate `subject_id` values are deduplicated: the first occurrence is kept, subsequent ones are skipped.
- Label consistency (e.g., ensuring labels are non-negative integers) is enforced by the integer cast check.

---

## Related

- [`SampleRecord`](../dataset/SampleRecord.md) — the dataclass populated for each valid row
- [`build_records_fallback()`](build_records_fallback.md) — alternative used when no CSV is available
- [`validate_metadata_csv()`](../validate/validate_metadata_csv.md) — upstream step that produces the clean CSV
- [`run_pipeline()`](run_pipeline.md) — calls `load_metadata_csv()` as its first stage
