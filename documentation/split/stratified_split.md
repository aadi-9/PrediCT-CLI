# `stratified_split()`

**Source:** `src/predict/split.py`

Partitions a list of items into train, validation, and test subsets using stratified random splitting, preserving the class distribution in each split.

---

## Signature

```python
def stratified_split(
    items:  list[T],
    labels: list[int] | np.ndarray,
    cfg:    SplitConfig,
) -> SplitResult
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `items` | `list[T]` | The items to split (e.g., a list of [`SampleRecord`](../dataset/SampleRecord.md)); any list type |
| `labels` | `list[int] \| np.ndarray` | Integer class labels, one per item; must be the same length as `items` |
| `cfg` | [`SplitConfig`](../config/SplitConfig.md) | Split configuration: `test_size`, `val_size`, `random_state` |

---

## Return Value

| Type | Description |
|---|---|
| [`SplitResult`](SplitResult.md) | Three lists of indices (into the original `items`) for train, val, and test sets |

---

## Split Algorithm

The split is performed in two stages:

### Stage 1 — Test split

Splits `range(len(items))` indices into `(trainval_indices, test_indices)` using `cfg.test_size` as the fraction for the test set.

### Stage 2 — Val split

Splits the `trainval_indices` further into `(train_indices, val_indices)` using `cfg.val_size` as the fraction for the validation set.

Both stages are stratified by class label when possible.

### Stratification strategy

**Primary (with scikit-learn):** Uses `sklearn.model_selection.train_test_split(..., stratify=labels, random_state=cfg.random_state)`.

**Fallback (without scikit-learn):** A manual numpy-based split that groups indices by class, shuffles each group, and distributes them proportionally across splits. A warning is added to the pipeline warnings list if scikit-learn is unavailable.

**Edge-case fallback:** If stratification fails (e.g., a class has fewer than 2 samples), falls back to a non-stratified random split and emits a warning.

---

## In the Data Pipeline

```
load_metadata_csv() or build_records_fallback()
  └─► list[SampleRecord], list[int labels]
        └─► stratified_split(items, labels, cfg)   ← here
              └─► SplitResult(train=[...], val=[...], test=[...])
                    └─► run_pipeline() partitions records into 3 VolumeDatasets
```

The resulting split indices are written to `outputs/splits.json`:

```json
{
  "train": [0, 3, 5, ...],
  "val":   [1, 7, ...],
  "test":  [2, 4, ...]
}
```

---

## Usage Example

```python
from predict.split import stratified_split
from predict.config import SplitConfig

items = list(range(100))
labels = [0] * 60 + [1] * 40  # 60 class-0, 40 class-1

cfg = SplitConfig(test_size=0.2, val_size=0.2, random_state=42)
split = stratified_split(items, labels, cfg)

print(len(split.train))  # ~64
print(len(split.val))    # ~16
print(len(split.test))   # ~20

# Verify no overlap
all_idx = set(split.train) | set(split.val) | set(split.test)
assert len(all_idx) == 100
```

---

## Notes

> **Warning:** If any class has fewer than 2 samples, stratified splitting is impossible. The function will fall back to a random (non-stratified) split and emit a warning. Check `outputs.stats["warnings"]` after running the pipeline.

> **Warning:** For very small datasets (< 10 subjects), the stratified split may produce empty splits. Consider using a larger dataset or adjusting `test_size`/`val_size`.

- The function takes a generic `list[T]` but only uses it to determine the list length; the `items` themselves are not reordered.
- `cfg.val_size` applies to the **post-test** pool, not the whole dataset.

---

## Related

- [`SplitResult`](SplitResult.md) — the returned dataclass
- [`SplitConfig`](../config/SplitConfig.md) — configuration for fractions and seed
- [`oversample_minority()`](../sampling/oversample_minority.md) — applied to the training split after this step
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls `stratified_split()` and uses the result
