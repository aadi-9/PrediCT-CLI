# `build_justification_text()`

**Source:** `src/predict/report.py`

Generates a human-readable, structured text summary of the preprocessing decisions made during a pipeline run, suitable for inclusion in research reports or experiment logs.

---

## Signature

```python
def build_justification_text(stats: dict[str, Any]) -> str
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `stats` | `dict[str, Any]` | Statistics dictionary produced by [`run_pipeline()`](../pipeline/run_pipeline.md) and written to `dataset_stats.json` |

---

## Return Value

| Type | Description |
|---|---|
| `str` | Multi-line formatted text covering all relevant preprocessing choices and their justifications |

---

## `stats` Dictionary Keys Used

| Key | Type | Used for |
|---|---|---|
| `num_subjects` | `int` | Dataset size summary |
| `split_sizes` | `dict` | Train/val/test count table |
| `class_counts_before_sampling` | `dict[str, int]` | Class balance before oversampling |
| `class_counts_after_sampling` | `dict[str, int]` | Class balance after oversampling |
| `resample` | `dict` | Resampling mode, target shape/spacing, interpolator |
| `hu_window` | `dict` | HU window lower and upper bounds |
| `paths` | `dict` | Key filesystem paths used in the run |
| `warnings` | `list[str]` | Any non-fatal warnings; listed at the end |

---

## Output Structure

The generated text is divided into labelled sections:

```
=== PrediCT Preprocessing Justification ===

Dataset
-------
Total subjects: N
  Train: X | Val: Y | Test: Z

HU Windowing
------------
Window: [lower, upper] HU
Rationale: ...

Resampling
----------
Mode: spacing | shape
Target spacing: Z x Y x X mm  (or target shape: Z x Y x Z)
Interpolator: linear | nearest
Rationale: ...

Class Balance
-------------
Before oversampling:
  Class 0: N
  Class 1: M
After oversampling (train set only):
  Class 0: N
  Class 1: N

Warnings
--------
  - <warning 1>
  - <warning 2>
  (or "None" if no warnings)
```

---

## In the Data Pipeline

`build_justification_text()` is called by [`write_justification_report()`](write_justification_report.md) immediately after [`run_pipeline()`](../pipeline/run_pipeline.md) writes `dataset_stats.json`.

```
run_pipeline()
  └─► stats = {...}
        └─► write_justification_report(stats=stats)
              └─► build_justification_text(stats)   ← here
                    └─► str
                          └─► outputs/justification.txt
```

---

## Usage Example

```python
import json
from predict.report import build_justification_text

with open("outputs/dataset_stats.json") as f:
    stats = json.load(f)

text = build_justification_text(stats)
print(text)
```

Or use the high-level wrapper directly:

```python
from predict.report import write_justification_report
write_justification_report(stats_path="outputs/dataset_stats.json")
```

---

## Notes

- The function is **pure** — it only reads from `stats` and returns a string; it has no side effects.
- If a key is missing from `stats` (e.g., because a very old `dataset_stats.json` was loaded), the function gracefully omits that section or falls back to `"N/A"` rather than raising a `KeyError`.
- The output is plain UTF-8 text, suitable for saving as `.txt` or embedding in a Markdown report.

---

## Related

- [`write_justification_report()`](write_justification_report.md) — calls this function and writes the result to disk
- [`run_pipeline()`](../pipeline/run_pipeline.md) — generates the `stats` dict consumed here
- [`PipelineOutputs`](../pipeline/PipelineOutputs.md) — `stats` is also accessible as `outputs.stats`
