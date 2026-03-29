# `LoaderConfig`

**Source:** `src/predict/config.py`

Frozen dataclass that controls `torch.utils.data.DataLoader` construction parameters.

---

## Signature

```python
@dataclass(frozen=True)
class LoaderConfig:
    batch_size:  int  = 2
    num_workers: int  = 0
    pin_memory:  bool = False
    shuffle:     bool = True
```

---

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `2` | Number of samples per batch |
| `num_workers` | `int` | `0` | Number of worker processes for data loading (0 = main process) |
| `pin_memory` | `bool` | `False` | Whether to pin batches to CUDA page-locked memory for faster GPU transfer |
| `shuffle` | `bool` | `True` | Whether to shuffle the dataset each epoch (applied to training loader only) |

---

## Description

`LoaderConfig` is passed directly to [`build_dataloader()`](../dataset/build_dataloader.md), which wraps a [`VolumeDataset`](../dataset/VolumeDataset.md) in a `torch.utils.data.DataLoader`.

### Defaults rationale

| Field | Rationale |
|---|---|
| `batch_size=2` | 3D medical volumes are large; small batches reduce GPU memory pressure |
| `num_workers=0` | Safe default; multiprocessing can cause issues with SimpleITK on some platforms |
| `pin_memory=False` | Safe default; set to `True` when training on GPU for a speed boost |
| `shuffle=True` | Prevents the model from learning sample order; typically applied to training loaders |

### Per-split shuffle behaviour

[`run_pipeline()`](../pipeline/run_pipeline.md) creates separate `LoaderConfig` instances with `shuffle=False` for the validation and test loaders, ensuring deterministic evaluation order regardless of the `shuffle` field on the config passed by the user.

---

## In the Data Pipeline

```
build_parser() → --batch-size, --num-workers
  └─► LoaderConfig(batch_size, num_workers, ...)
        └─► run_pipeline(loader_cfg=...)
              ├─► build_dataloader(train_dataset, loader_cfg)          ← shuffle=True
              ├─► build_dataloader(val_dataset,   loader_cfg_no_shuf)  ← shuffle=False
              └─► build_dataloader(test_dataset,  loader_cfg_no_shuf)  ← shuffle=False
```

---

## Usage Example

```python
from predict.config import LoaderConfig

# Default: small batches, single process
cfg = LoaderConfig()

# GPU-optimised: larger batches, multiple workers, pinned memory
cfg = LoaderConfig(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
)

# Deterministic evaluation loader
eval_cfg = LoaderConfig(batch_size=4, shuffle=False)
```

---

## Notes

- When `num_workers > 0`, each worker spawns a new process. Ensure your OS allows the required number of file handles, especially when reading DICOM series.
- `pin_memory=True` is only beneficial when using a CUDA GPU; it has no effect (or may slow things down) on CPU-only machines.
- The dataclass is `frozen=True`.

---

## Related

- [`build_dataloader()`](../dataset/build_dataloader.md) — consumes `LoaderConfig` to construct a `DataLoader`
- [`VolumeDataset`](../dataset/VolumeDataset.md) — the dataset wrapped by the DataLoader
- [`SplitConfig`](SplitConfig.md) — companion config controlling train/val/test split proportions
- [`run_pipeline()`](../pipeline/run_pipeline.md) — assembles `LoaderConfig` from CLI args
