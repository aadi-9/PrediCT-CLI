# `PipelineOutputs`

**Source:** `src/predict/pipeline.py`

Frozen dataclass that bundles every artifact produced by [`run_pipeline()`](run_pipeline.md) into a single return value.

---

## Signature

```python
@dataclass(frozen=True)
class PipelineOutputs:
    train_loader:              Any | None
    val_loader:                Any | None
    test_loader:               Any | None
    stats:                     dict[str, Any]
    processed_manifest_path:   Path | None
    split_manifest_path:       Path | None
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `train_loader` | `torch.utils.data.DataLoader \| None` | DataLoader for the training set (with augmentation and optional oversampling); `None` in `dry_run` mode |
| `val_loader` | `torch.utils.data.DataLoader \| None` | DataLoader for the validation set (no augmentation, no shuffle); `None` in `dry_run` mode |
| `test_loader` | `torch.utils.data.DataLoader \| None` | DataLoader for the test set (no augmentation, no shuffle); `None` in `dry_run` mode |
| `stats` | `dict[str, Any]` | Dictionary of pipeline statistics (see schema below) |
| `processed_manifest_path` | `Path \| None` | Path to `processed_manifest.csv` if `export_processed=True`; else `None` |
| `split_manifest_path` | `Path \| None` | Path to `splits.json` if written successfully; else `None` |

---

## `stats` Dictionary Schema

The `stats` dict is also serialised to `outputs/dataset_stats.json`. Its keys are:

| Key | Type | Description |
|---|---|---|
| `num_subjects` | `int` | Total number of valid subjects loaded from the metadata CSV |
| `split_sizes` | `dict` | Counts of subjects per split: `{"train": N, "val": N, "test": N}` |
| `class_counts_before_sampling` | `dict[str, int]` | Per-class subject counts in the training set **before** oversampling |
| `class_counts_after_sampling` | `dict[str, int]` | Per-class subject counts in the training set **after** oversampling |
| `resample` | `dict` | Resampling config: `{"mode": ..., "target_shape": ..., "target_spacing": ..., "interpolator": ...}` |
| `hu_window` | `dict` | HU window config: `{"lower": ..., "upper": ...}` |
| `paths` | `dict` | Key filesystem paths used during the run |
| `warnings` | `list[str]` | Any non-fatal warnings generated during the run |

---

## In the Data Pipeline

`PipelineOutputs` is the final return value of the pipeline. Consumers (training scripts, notebooks, tests) unpack it to access loaders and stats.

```
run_pipeline(...)
  └─► PipelineOutputs(          ← here
        train_loader=...,
        val_loader=...,
        test_loader=...,
        stats={...},
        processed_manifest_path=...,
        split_manifest_path=...,
      )
```

---

## Usage Example

```python
from pathlib import Path
from predict.pipeline import run_pipeline

outputs = run_pipeline(project_root=Path("."))

# Use the DataLoaders for training
for batch_tensors, batch_labels in outputs.train_loader:
    # batch_tensors shape: (B, 1, Z, Y, X)
    loss = model(batch_tensors)
    ...

# Inspect statistics
print(outputs.stats["num_subjects"])
print(outputs.stats["split_sizes"])

# Check for warnings
for w in outputs.stats["warnings"]:
    print("WARNING:", w)

# Path to the written splits file
if outputs.split_manifest_path:
    print(f"Splits saved to: {outputs.split_manifest_path}")
```

---

## Notes

- All three loaders are `None` when [`run_pipeline()`](run_pipeline.md) is called with `dry_run=True`. This mode is useful for validating configs without incurring the cost of DataLoader construction.
- `processed_manifest_path` is `None` unless `export_processed=True` was passed to `run_pipeline()`.
- The `stats` dict is always populated, even in `dry_run` mode.
- The dataclass is `frozen=True`.

---

## Related

- [`run_pipeline()`](run_pipeline.md) — the function that constructs and returns `PipelineOutputs`
- [`build_dataloader()`](../dataset/build_dataloader.md) — constructs each loader stored in this dataclass
- [`VolumeDataset`](../dataset/VolumeDataset.md) — the underlying dataset wrapped by each loader
- [`build_justification_text()`](../report/build_justification_text.md) — consumes `stats` to produce the justification report
