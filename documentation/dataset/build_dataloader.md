# `build_dataloader()`

**Source:** `src/predict/dataset.py`

Wraps a [`VolumeDataset`](VolumeDataset.md) in a `torch.utils.data.DataLoader` configured with [`LoaderConfig`](../config/LoaderConfig.md) settings and the [`pad_collate_fn()`](pad_collate_fn.md) collate function.

---

## Signature

```python
def build_dataloader(
    dataset: VolumeDataset,
    cfg:     LoaderConfig,
) -> torch.utils.data.DataLoader
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `dataset` | [`VolumeDataset`](VolumeDataset.md) | The dataset to wrap |
| `cfg` | [`LoaderConfig`](../config/LoaderConfig.md) | DataLoader configuration (batch size, workers, pin memory, shuffle) |

---

## Return Value

| Type | Description |
|---|---|
| `torch.utils.data.DataLoader` | Configured DataLoader ready for use in a training/evaluation loop |

---

## Description

`build_dataloader()` creates a `DataLoader` with:

```python
DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    shuffle=cfg.shuffle,
    collate_fn=pad_collate_fn,
)
```

The `pad_collate_fn` is always used, ensuring variable-size volumes in the same batch are zero-padded to a common shape before stacking.

---

## In the Data Pipeline

`build_dataloader()` is the final step before DataLoaders are returned to the caller via [`PipelineOutputs`](../pipeline/PipelineOutputs.md).

```
VolumeDataset(train_records)
  └─► build_dataloader(train_dataset, loader_cfg)   ← here
        └─► DataLoader → PipelineOutputs.train_loader
```

[`run_pipeline()`](../pipeline/run_pipeline.md) calls this three times, once each for train, validation, and test datasets. For the validation and test loaders, a modified `LoaderConfig` with `shuffle=False` is used regardless of the user-provided config.

---

## Usage Example

```python
from predict.dataset import VolumeDataset, build_dataloader, SampleRecord, default_load_volume
from predict.config import LoaderConfig
from pathlib import Path

records = [
    SampleRecord("p001", Path("data/processed/p001.npy"), label=0, kind="numpy"),
    SampleRecord("p002", Path("data/processed/p002.npy"), label=1, kind="numpy"),
]
dataset = VolumeDataset(records=records)
cfg = LoaderConfig(batch_size=2, num_workers=0, shuffle=True)

loader = build_dataloader(dataset, cfg)

for batch_tensors, batch_labels in loader:
    print(batch_tensors.shape)  # (2, 1, Z, Y, X)
    print(batch_labels)         # tensor([0, 1])
    break
```

---

## Notes

> **Warning:** Requires PyTorch to be installed. Raises `ImportError` if PyTorch is not available.

- Setting `num_workers > 0` spawns worker subprocesses. On Windows, this requires protecting the script entry point with `if __name__ == "__main__":`. On macOS/Linux with `fork`, be cautious of interactions between multiprocessing and SimpleITK.
- `pin_memory=True` is only beneficial when training on a CUDA GPU and using `num_workers > 0`.

---

## Related

- [`VolumeDataset`](VolumeDataset.md) — the dataset wrapped by this function
- [`pad_collate_fn()`](pad_collate_fn.md) — always used as the `collate_fn`
- [`LoaderConfig`](../config/LoaderConfig.md) — configuration dataclass
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls `build_dataloader()` for each split
- [`PipelineOutputs`](../pipeline/PipelineOutputs.md) — stores the returned loaders
