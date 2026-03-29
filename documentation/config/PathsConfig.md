# `PathsConfig`

**Source:** `src/predict/config.py`

Frozen dataclass that centralises all filesystem paths used by the PrediCT pipeline.

---

## Signature

```python
@dataclass(frozen=True)
class PathsConfig:
    project_root:  Path
    data_dir:      Path
    raw_dir:       Path
    processed_dir: Path
    outputs_dir:   Path

    @staticmethod
    def from_project_root(
        project_root: Path,
        raw_dir: Path | None = None,
    ) -> "PathsConfig": ...
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `project_root` | `Path` | Absolute path to the project root directory |
| `data_dir` | `Path` | `project_root / "data"` — parent of raw and processed data |
| `raw_dir` | `Path` | Resolved path to raw DICOM input data (see [`resolve_raw_dir()`](resolve_raw_dir.md)) |
| `processed_dir` | `Path` | `project_root / "data/processed"` — where `.npy` exports are written |
| `outputs_dir` | `Path` | `project_root / "outputs"` — where JSON/CSV/TXT artifacts are written |

All fields are read-only after construction (`frozen=True`).

---

## Factory Method: `from_project_root()`

```python
@staticmethod
def from_project_root(
    project_root: Path,
    raw_dir: Path | None = None,
) -> PathsConfig
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `project_root` | `Path` | — | Root of the project; all other paths are derived from this |
| `raw_dir` | `Path \| None` | `None` | Optional override for the raw DICOM directory; passed to [`resolve_raw_dir()`](resolve_raw_dir.md) |

### Return Value

A fully populated `PathsConfig` instance with all paths resolved to absolute values.

### Behaviour

```
from_project_root(project_root, raw_dir)
  ├─► data_dir      = project_root / "data"
  ├─► raw_dir       = resolve_raw_dir(project_root, raw_dir)
  ├─► processed_dir = project_root / "data" / "processed"
  └─► outputs_dir   = project_root / "outputs"
```

---

## In the Data Pipeline

`PathsConfig` is constructed at the start of [`run_pipeline()`](../pipeline/run_pipeline.md) and passed through to functions that need filesystem locations, eliminating scattered hardcoded paths.

```
run_pipeline(project_root, raw_dir=...)
  └─► PathsConfig.from_project_root(...)   ← here
        └─► PathsConfig(project_root, data_dir, raw_dir, processed_dir, outputs_dir)
              ├─► load_metadata_csv(... paths.raw_dir ...)
              ├─► build_records_fallback(paths)
              ├─► save_numpy_volume(... paths.processed_dir ...)
              └─► _write_json(paths.outputs_dir / "dataset_stats.json", ...)
```

---

## Usage Example

```python
from pathlib import Path
from predict.config import PathsConfig

paths = PathsConfig.from_project_root(Path("/home/user/cardiac"))

print(paths.project_root)   # /home/user/cardiac
print(paths.data_dir)       # /home/user/cardiac/data
print(paths.raw_dir)        # /home/user/cardiac/data/raw  (or PREDICT_RAW_DIR env)
print(paths.processed_dir)  # /home/user/cardiac/data/processed
print(paths.outputs_dir)    # /home/user/cardiac/outputs

# With an explicit raw directory override:
paths = PathsConfig.from_project_root(
    Path("/home/user/cardiac"),
    raw_dir=Path("/mnt/nfs/dicom"),
)
print(paths.raw_dir)  # /mnt/nfs/dicom
```

---

## Notes

- The dataclass is `frozen=True`, so fields cannot be mutated after construction. Create a new instance if different paths are needed.
- `processed_dir` and `outputs_dir` are not created automatically; pipeline functions call `mkdir(parents=True, exist_ok=True)` before writing files.

---

## Related

- [`resolve_raw_dir()`](resolve_raw_dir.md) — resolves the `raw_dir` field
- [`run_pipeline()`](../pipeline/run_pipeline.md) — primary consumer of `PathsConfig`
- [`build_records_fallback()`](../pipeline/build_records_fallback.md) — receives a `PathsConfig` to discover subject directories
- [`discover_subject_dirs()`](../io/discover_subject_dirs.md) — uses `paths.raw_dir`
