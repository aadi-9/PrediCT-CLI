from __future__ import annotations

from typing import TypeVar

import numpy as np

T = TypeVar("T")


def oversample_minority(
    items: list[T],
    labels: list[int] | np.ndarray,
    random_state: int = 42,
) -> tuple[list[T], list[int]]:
    """
    Random oversampling to address class imbalance.

    Example (from instructions):
        ros = RandomOverSampler()
        data_resampled, labels_resampled = ros.fit_resample(data, labels)

    Next steps:
    - Decide whether oversampling is acceptable for your evaluation (vs class-weighted loss).
    - Report class counts before/after to satisfy dataset statistics requirement.
    """
    n = len(items)
    if n == 0:
        return [], []
    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.shape[0] != n:
        raise ValueError("labels length must match items length")

    try:
        from imblearn.over_sampling import RandomOverSampler  # type: ignore[import-not-found]

        X = np.arange(n).reshape(-1, 1)
        ros = RandomOverSampler(random_state=random_state)
        X_res, y_res = ros.fit_resample(X, labels_arr)
        idx_res = X_res.reshape(-1).astype(int).tolist()
        return [items[i] for i in idx_res], y_res.astype(int).tolist()
    except Exception:
        rng = np.random.default_rng(random_state)
        unique, counts = np.unique(labels_arr, return_counts=True)
        max_count = int(counts.max())

        indices_by_class: dict[int, list[int]] = {int(u): [] for u in unique.tolist()}
        for i, y in enumerate(labels_arr.tolist()):
            indices_by_class[int(y)].append(i)

        resampled_indices: list[int] = []
        for y, idxs in indices_by_class.items():
            idxs = idxs.copy()
            if len(idxs) == 0:
                continue
            extra = rng.choice(idxs, size=max_count - len(idxs), replace=True).tolist()
            resampled_indices.extend(idxs + extra)

        rng.shuffle(resampled_indices)
        resampled_items = [items[i] for i in resampled_indices]
        resampled_labels = [int(labels_arr[i]) for i in resampled_indices]
        return resampled_items, resampled_labels
