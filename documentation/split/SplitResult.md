# `SplitResult`

**Source:** `src/predict/split.py`

Frozen dataclass that holds the results of a stratified train/val/test split as three lists of **indices** into the original records list.

---

## Signature

```python
@dataclass(frozen=True)
class SplitResult:
    train: list[int]
    val:   list[int]
    test:  list[int]
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `train` | `list[int]` | Indices of records assigned to the training set |
| `val` | `list[int]` | Indices of records assigned to the validation set |
| `test` | `list[int]` | Indices of records assigned to the test set |

Indices refer to positions in the original `items` list passed to [`stratified_split()`](stratified_split.md). No record appears in more than one split.

---

## In the Data Pipeline

`SplitResult` is returned by [`stratified_split()`](stratified_split.md) and immediately used inside [`run_pipeline()`](../pipeline/run_pipeline.md) to partition records into three `VolumeDataset` instances.

```
stratified_split(records, labels, cfg)
  └─► SplitResult(train=[0,3,4,...], val=[1,6,...], test=[2,5,...])   ← here
        └─► run_pipeline():
              train_records = [records[i] for i in split.train]
              val_records   = [records[i] for i in split.val]
              test_records  = [records[i] for i in split.test]
```

The indices (not the records themselves) are also written to `outputs/splits.json` for reproducibility.

---

## Usage Example

```python
from predict.split import SplitResult

# Manually construct (usually returned by stratified_split)
split = SplitResult(
    train=[0, 2, 3, 5, 7, 8],
    val=[1, 6],
    test=[4, 9],
)

records = [...]  # list of SampleRecord
train_records = [records[i] for i in split.train]
val_records   = [records[i] for i in split.val]
test_records  = [records[i] for i in split.test]

print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
```

---

## Notes

- The indices are into the **original unsorted** records list, not a shuffled or sorted copy. The order of elements within each list matches the order produced by scikit-learn's `train_test_split` (or the fallback numpy split).
- All three lists together cover every index in `range(len(items))` with no overlap.
- The dataclass is `frozen=True`.

---

## Related

- [`stratified_split()`](stratified_split.md) — the function that constructs and returns `SplitResult`
- [`SplitConfig`](../config/SplitConfig.md) — configures the split fractions and random seed
- [`run_pipeline()`](../pipeline/run_pipeline.md) — uses `SplitResult` to partition records for each split's `VolumeDataset`
