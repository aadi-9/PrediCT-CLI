# `validate_metadata_csv()`

**Source:** `src/predict/validate.py`

Reads a metadata CSV, validates each DICOM series directory for readability, and writes two output files: a cleaned CSV containing only valid rows, and a full validation report CSV covering every row.

---

## Signature

```python
def validate_metadata_csv(
    metadata_csv: Path,
    raw_dir:      Path,
    *,
    out_clean_csv:  Path,
    out_report_csv: Path,
    mode:   Literal["fast", "shallow", "deep"] = "shallow",
    resume: bool = True,
) -> tuple[Path, Path]
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metadata_csv` | `Path` | — | Input metadata CSV (produced by [`generate_metadata_csv()`](../metadata/generate_metadata_csv.md) or manually curated) |
| `raw_dir` | `Path` | — | Base directory for resolving relative `image` paths in the CSV |
| `out_clean_csv` | `Path` | — | Output path for the cleaned CSV (valid rows only) |
| `out_report_csv` | `Path` | — | Output path for the full validation report (all rows with status) |
| `mode` | `str` | `"shallow"` | Depth of DICOM check; passed to [`validate_dicom_series_dir()`](validate_dicom_series_dir.md) |
| `resume` | `bool` | `True` | If `True`, rows already present in `out_report_csv` are skipped |

---

## Return Value

A 2-tuple:

| Position | Type | Description |
|---|---|---|
| `[0]` — `clean_csv_path` | `Path` | Path to the written clean CSV |
| `[1]` — `report_csv_path` | `Path` | Path to the written validation report CSV |

---

## Behaviour

### Input CSV requirements

The input CSV must have at minimum these columns:

| Column | Description |
|---|---|
| `subject_id` | Unique identifier for the subject |
| `image` | Path to the subject's data directory (absolute, or relative to `raw_dir`) |
| `label` | Integer class label |
| `kind` | Data type: `dicom_series`, `numpy`, `nifti`, etc. |

### Validation logic

For each row in the input CSV:

- If `kind == "dicom_series"` → calls [`validate_dicom_series_dir(series_dir, mode=mode)`](validate_dicom_series_dir.md).
- If `kind` is anything else (e.g., `"numpy"`, `"nifti"`) → the row is **passed through without validation**, marked as `ok=True`.

### Resume support

When `resume=True`, the function reads any rows already written to `out_report_csv` and builds a set of `(subject_id, image)` keys that have been processed. Rows matching these keys are skipped, allowing interrupted validation runs to continue from where they left off.

### Output CSV schemas

**Clean CSV** (`out_clean_csv`) — same schema as the input, containing only rows where `ok=True`:

| Column | Description |
|---|---|
| `subject_id` | Subject identifier |
| `image` | Path to subject data |
| `label` | Class label |
| `kind` | Data kind |

**Report CSV** (`out_report_csv`) — all rows plus validation details:

| Column | Description |
|---|---|
| `subject_id` | Subject identifier |
| `image` | Path to subject data |
| `label` | Class label |
| `kind` | Data kind |
| `ok` | `True` or `False` |
| `num_files` | Count of `.dcm` files found |
| `reason` | `"ok"` or error description |

---

## In the Data Pipeline

```
generate_metadata_csv()
  └─► metadata_all.csv
        └─► validate_metadata_csv(metadata_csv=metadata_all.csv, ...)   ← here
              ├─► metadata_clean.csv       → load_metadata_csv() → run_pipeline()
              └─► dicom_validation_report.csv  (audit log)
```

---

## Usage Example

```python
from pathlib import Path
from predict.validate import validate_metadata_csv

clean, report = validate_metadata_csv(
    metadata_csv=Path("outputs/metadata_all.csv"),
    raw_dir=Path("data/raw"),
    out_clean_csv=Path("outputs/metadata_clean.csv"),
    out_report_csv=Path("outputs/dicom_validation_report.csv"),
    mode="shallow",
    resume=True,
)

print(f"Clean CSV:  {clean}")
print(f"Report CSV: {report}")
```

### From the CLI

```bash
predict validate-metadata \
  --metadata-csv outputs/metadata_all.csv \
  --raw-dir data/raw \
  --out-clean outputs/metadata_clean.csv \
  --out-report outputs/dicom_validation_report.csv \
  --mode shallow
```

To force a full re-validation (ignore previous results):

```bash
predict validate-metadata ... --no-resume
```

---

## Notes

> **Warning:** Invalid rows (where `ok=False`) are written to the report CSV but **excluded** from the clean CSV. Downstream pipeline stages (`run_pipeline`) will not see these subjects.

> **Warning:** `"shallow"` and `"deep"` modes require `SimpleITK`. Without it, all `dicom_series` rows will be marked as invalid with reason `"SimpleITK not available"`.

- Parent directories for both output files are created automatically (`mkdir(parents=True, exist_ok=True)`).
- If `resume=True` and `out_report_csv` does not yet exist, the function starts fresh (no error).
- The `resume` feature is especially useful for large datasets that may take hours to validate.

---

## Related

- [`validate_dicom_series_dir()`](validate_dicom_series_dir.md) — the per-series validation function
- [`generate_metadata_csv()`](../metadata/generate_metadata_csv.md) — produces the input CSV
- [`load_metadata_csv()`](../pipeline/load_metadata_csv.md) — reads the cleaned CSV into `SampleRecord` objects
- [`run_pipeline()`](../pipeline/run_pipeline.md) — consumes the cleaned CSV to build DataLoaders
