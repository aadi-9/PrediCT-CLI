# `SplitConfig`

**Source:** `src/predict/config.py`

Frozen dataclass that controls how subjects are partitioned into training, validation, and test sets.

---

## Signature

```python
@dataclass(frozen=True)
class SplitConfig:
    test_size:    float = 0.2
    val_size:     float = 0.2
    random_state: int   = 42
```

---

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `test_size` | `float` | `0.2` | Fraction of all subjects reserved for the test set |
| `val_size` | `float` | `0.2` | Fraction of the **remaining** (non-test) subjects reserved for validation |
| `random_state` | `int` | `42` | Random seed for reproducibility of splits |

---

## Description

The split is performed in two sequential stratified steps:

1. **Test split** — `test_size × N` subjects are held out as the test set.
2. **Val split** — `val_size × (N − N_test)` subjects are held out from the remaining subjects as the validation set.
3. **Train** — all remaining subjects form the training set.

With defaults `test_size=0.2` and `val_size=0.2` on 100 subjects:

```
Total:  100
Test:    20  (20 % of 100)
Val:     16  (20 % of 80 remaining)
Train:   64
```

Splits are **stratified by label**, meaning class proportions are approximately preserved in each split. If stratification is not possible (e.g., a class has only one sample), [`stratified_split()`](../split/stratified_split.md) falls back to a non-stratified random split with a warning.

---

## In the Data Pipeline

`SplitConfig` is passed to [`stratified_split()`](../split/stratified_split.md) inside [`run_pipeline()`](../pipeline/run_pipeline.md).

```
build_parser() → --test-size, --val-size, --random-state
  └─► SplitConfig(test_size, val_size, random_state)
        └─► run_pipeline(split_cfg=...)
              └─► stratified_split(items, labels, cfg=split_cfg)   ← consumed here
                    └─► SplitResult(train=[...], val=[...], test=[...])
```

Split indices are written to `outputs/splits.json` for reproducibility.

---

## Usage Example

```python
from predict.config import SplitConfig

# Defaults: 20 % test, 20 % of remainder for val
cfg = SplitConfig()

# Custom: 15 % test, 10 % of remainder for val, different seed
cfg = SplitConfig(test_size=0.15, val_size=0.10, random_state=7)
```

---

## Notes

- `test_size` and `val_size` must both be in the range `(0, 1)`. Values outside this range will cause `stratified_split()` to raise errors from scikit-learn or produce degenerate splits.
- `val_size` is applied to the **post-test** pool, not the whole dataset, so the effective validation fraction of the total dataset is `val_size × (1 − test_size)`.
- The split indices (not subject IDs) are what `SplitResult` stores; the pipeline maps indices back to `SampleRecord` lists.
- The dataclass is `frozen=True`.

---

## Related

- [`stratified_split()`](../split/stratified_split.md) — the function that consumes `SplitConfig`
- [`SplitResult`](../split/SplitResult.md) — the dataclass returned by `stratified_split()`
- [`run_pipeline()`](../pipeline/run_pipeline.md) — assembles `SplitConfig` from CLI args
- [`LoaderConfig`](LoaderConfig.md) — companion dataclass controlling DataLoader behaviour
