# `resolve_raw_dir()`

**Source:** `src/predict/config.py`

Resolves the raw DICOM directory to an absolute `Path`, following a three-level precedence chain.

---

## Signature

```python
def resolve_raw_dir(
    project_root: Path,
    raw_dir: Path | None = None,
) -> Path
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `project_root` | `Path` | — | Absolute path to the project root; used as the base when resolving relative paths |
| `raw_dir` | `Path \| None` | `None` | Explicit override for the raw DICOM directory |

---

## Return Value

| Type | Description |
|---|---|
| `Path` | Absolute path to the raw DICOM directory |

---

## Resolution Order

The function applies the following precedence, stopping at the first value that is not `None` / empty:

1. **Explicit `raw_dir` argument** — If `raw_dir` is provided and non-empty, it is used directly.
2. **`PREDICT_RAW_DIR` environment variable** — If the env var `PREDICT_RAW_DIR` is set (and non-empty), its value is used.
3. **`DEFAULT_RAW_DICOM_DIR` constant** — Falls back to the module-level constant (`data/raw` relative to `project_root`).

If the chosen path is relative (not an absolute path), it is joined to `project_root` to produce an absolute path.

---

## In the Data Pipeline

`resolve_raw_dir()` is called inside [`PathsConfig.from_project_root()`](PathsConfig.md) and, separately, inside [`run_pipeline()`](../pipeline/run_pipeline.md) when building the `PathsConfig` object.

```
run_pipeline(project_root, raw_dir=...)
  └─► PathsConfig.from_project_root(project_root, raw_dir)
        └─► resolve_raw_dir(project_root, raw_dir)   ← here
              └─► returns absolute Path for raw DICOM data
```

---

## Usage Example

```python
from pathlib import Path
from predict.config import resolve_raw_dir

root = Path("/home/user/cardiac_project")

# 1. Explicit path
p = resolve_raw_dir(root, raw_dir=Path("/mnt/nfs/dicom"))
# → Path("/mnt/nfs/dicom")

# 2. Relative explicit path
p = resolve_raw_dir(root, raw_dir=Path("my_dicom_data"))
# → Path("/home/user/cardiac_project/my_dicom_data")

# 3. From environment variable
import os
os.environ["PREDICT_RAW_DIR"] = "/data/dicom"
p = resolve_raw_dir(root)
# → Path("/data/dicom")

# 4. Default fallback
del os.environ["PREDICT_RAW_DIR"]
p = resolve_raw_dir(root)
# → Path("/home/user/cardiac_project/data/raw")
```

---

## Notes

> **Warning:** `resolve_raw_dir()` does **not** check whether the returned path exists. Callers are responsible for verifying existence before attempting to read files.

- The environment variable name is stored in the module-level constant `PREDICT_RAW_DIR_ENV = "PREDICT_RAW_DIR"`.
- The default fallback constant `DEFAULT_RAW_DICOM_DIR` is `Path("data/raw")`.

---

## Related

- [`PathsConfig`](PathsConfig.md) — dataclass that calls `resolve_raw_dir()` in its factory method
- [`run_pipeline()`](../pipeline/run_pipeline.md) — top-level consumer of the resolved path
- [`discover_subject_dirs()`](../io/discover_subject_dirs.md) — uses the resolved `raw_dir` to find subject subdirectories
