# `oversample_minority()`

**Source:** `src/predict/sampling.py`

Balances class representation in the training set by duplicating minority-class samples until all classes have equal representation.

---

## Signature

```python
def oversample_minority(
    items:        list[T],
    labels:       list[int] | np.ndarray,
    random_state: int = 42,
) -> tuple[list[T], list[int]]
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `items` | `list[T]` | — | The samples to resample (e.g., a list of [`SampleRecord`](../dataset/SampleRecord.md)); must be the same length as `labels` |
| `labels` | `list[int] \| np.ndarray` | — | Integer class labels, one per item |
| `random_state` | `int` | `42` | Random seed for reproducibility |

---

## Return Value

A 2-tuple:

| Position | Type | Description |
|---|---|---|
| `[0]` — resampled items | `list[T]` | Resampled list with minority classes duplicated; length equals `N_majority × num_classes` |
| `[1]` — resampled labels | `list[int]` | Corresponding labels for the resampled items |

---

## Algorithm

**Primary (with imbalanced-learn):** Uses `imblearn.over_sampling.RandomOverSampler`:

1. Reshapes `items` indices into a 2D array `[[0], [1], ..., [N-1]]`.
2. Calls `RandomOverSampler(random_state=random_state).fit_resample(indices, labels)`.
3. Maps the resampled index array back to the original `items` list.

**Fallback (without imbalanced-learn):** Manual numpy-based oversampling:

1. Finds the majority class count `N_max`.
2. For each minority class, randomly samples (with replacement) enough items to reach `N_max`.
3. Concatenates all class groups together.

---

## Effect on Class Counts

Given training data with class imbalance:

```
Before oversampling:
  Class 0: 80 samples
  Class 1: 20 samples

After oversample_minority():
  Class 0: 80 samples
  Class 1: 80 samples  (60 duplicates added)
  Total:   160 samples
```

---

## In the Data Pipeline

`oversample_minority()` is applied **only to the training split** inside [`run_pipeline()`](../pipeline/run_pipeline.md). Validation and test splits are never oversampled to preserve unbiased evaluation.

```
stratified_split() → SplitResult.train (indices)
  └─► train_records = [records[i] for i in split.train]
        └─► oversample_minority(train_records, train_labels)   ← here
              └─► (resampled_records, resampled_labels)
                    └─► VolumeDataset(resampled_records)
```

Class counts before and after oversampling are recorded in `stats["class_counts_before_sampling"]` and `stats["class_counts_after_sampling"]`.

---

## Usage Example

```python
from predict.sampling import oversample_minority

items = list(range(10))
labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  # 7 class-0, 3 class-1

resampled_items, resampled_labels = oversample_minority(items, labels, random_state=42)

print(len(resampled_items))   # 14  (7 + 7 duplicated class-1 items)
from collections import Counter
print(Counter(resampled_labels))  # Counter({0: 7, 1: 7})
```

---

## Notes

> **Warning:** Oversampling duplicates existing `SampleRecord` objects in the list. If the `load_fn` in `VolumeDataset` reads from disk (e.g., [`read_dicom_series()`](../io/read_dicom_series.md)), each duplicate will trigger a separate disk read during training. This increases I/O but does not increase disk usage.

> **Note:** Only the training split should be oversampled. Applying oversampling to validation or test splits would give misleadingly optimistic evaluation metrics.

- The function works on any list type `T` (records, file paths, integers, etc.).
- To disable oversampling, pass `oversample_train=False` to [`run_pipeline()`](../pipeline/run_pipeline.md).

---

## Related

- [`stratified_split()`](../split/stratified_split.md) — produces the training subset that this function rebalances
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls `oversample_minority()` when `oversample_train=True`
- [`VolumeDataset`](../dataset/VolumeDataset.md) — receives the resampled records list
- [`SampleRecord`](../dataset/SampleRecord.md) — the typical type of items being resampled
