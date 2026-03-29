# `write_justification_report()`

**Source:** `src/predict/report.py`

Generates a human-readable preprocessing justification report and writes it to a text file. Accepts either a pre-loaded stats dictionary or a path to a `dataset_stats.json` file.

---

## Signature

```python
def write_justification_report(
    stats:      dict[str, Any] | None = None,
    stats_path: Path | None           = None,
    out_path:   Path | None           = None,
) -> Path
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stats` | `dict[str, Any] \| None` | `None` | Pre-loaded statistics dictionary (as returned in `PipelineOutputs.stats`) |
| `stats_path` | `Path \| None` | `None` | Path to a `dataset_stats.json` file to load; used if `stats` is `None` |
| `out_path` | `Path \| None` | `None` | Output path for the justification text file; defaults to `outputs/justification.txt` relative to the working directory |

---

## Return Value

| Type | Description |
|---|---|
| `Path` | Absolute path to the written justification text file |

---

## Raises

| Exception | Condition |
|---|---|
| `ValueError` | Both `stats` and `stats_path` are `None` |
| `FileNotFoundError` | `stats_path` is provided but does not exist |

---

## Description

`write_justification_report()` follows this logic:

1. If `stats` is provided, use it directly.
2. Else if `stats_path` is provided, load it with `json.load()`.
3. Else raise `ValueError`.
4. Call [`build_justification_text(stats)`](build_justification_text.md) to generate the report string.
5. Resolve `out_path`: if `None`, default to `Path("outputs/justification.txt")`.
6. Create parent directories with `mkdir(parents=True, exist_ok=True)`.
7. Write the string to `out_path` (UTF-8 encoding).
8. Return `out_path`.

---

## In the Data Pipeline

`write_justification_report()` is called as the final step of [`run_pipeline()`](../pipeline/run_pipeline.md), after statistics have been assembled and `dataset_stats.json` has been written.

```
run_pipeline()
  └─► stats = {num_subjects, split_sizes, ...}
        ├─► _write_json(stats_path, stats)   → outputs/dataset_stats.json
        └─► write_justification_report(stats=stats, out_path=...)   ← here
              └─► build_justification_text(stats)
                    └─► outputs/justification.txt
```

---

## Usage Example

### From a stats dict (in-memory)

```python
from pathlib import Path
from predict.report import write_justification_report

stats = {
    "num_subjects": 100,
    "split_sizes": {"train": 64, "val": 16, "test": 20},
    "class_counts_before_sampling": {"0": 60, "1": 4},
    "class_counts_after_sampling": {"0": 60, "1": 60},
    "resample": {
        "mode": "spacing",
        "target_spacing": [1.0, 1.0, 1.0],
        "interpolator": "linear",
    },
    "hu_window": {"lower": -200.0, "upper": 400.0},
    "paths": {},
    "warnings": [],
}

out = write_justification_report(stats=stats)
print(f"Report written to: {out}")
```

### From a saved JSON file

```python
from pathlib import Path
from predict.report import write_justification_report

out = write_justification_report(
    stats_path=Path("outputs/dataset_stats.json"),
    out_path=Path("outputs/justification.txt"),
)
```

### From the pipeline output

```python
from predict.pipeline import run_pipeline
from predict.report import write_justification_report
from pathlib import Path

outputs = run_pipeline(project_root=Path("."), dry_run=True)
# write_justification_report is called automatically by run_pipeline,
# but you can call it again with a custom path:
write_justification_report(
    stats=outputs.stats,
    out_path=Path("reports/preprocessing_report.txt"),
)
```

---

## Notes

- If `out_path` already exists, it is **silently overwritten**.
- The function creates all parent directories automatically, so `Path("deep/nested/dir/report.txt")` will work even if the intermediate directories don't yet exist.
- Both `stats` and `stats_path` can be provided simultaneously; `stats` takes priority.

---

## Related

- [`build_justification_text()`](build_justification_text.md) — generates the text content; called internally
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls this function automatically at the end of each pipeline run
- [`PipelineOutputs`](../pipeline/PipelineOutputs.md) — `stats` is available as `outputs.stats`
