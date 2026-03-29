from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from .config import SplitConfig

T = TypeVar("T")


@dataclass(frozen=True)
class SplitResult:
    train: list[int]
    val: list[int]
    test: list[int]


def stratified_split(
    items: list[T],
    labels: list[int] | np.ndarray,
    cfg: SplitConfig,
) -> SplitResult:
    """
    Stratified train/val/test split.

    Example (from instructions):
        train, test = train_test_split(data, test_size=0.2, stratify=labels)
        train, val = train_test_split(train, test_size=0.25, stratify=train_labels)

    Next steps:
    - Ensure labels reflect your class balance requirements.
    - Ensure you split by subject/patient, not by slice, to avoid leakage.
    """
    n = len(items)
    if n == 0:
        return SplitResult(train=[], val=[], test=[])

    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != n:
        raise ValueError("labels length must match items length")

    try:
        from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]

        idx = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=cfg.test_size,
            stratify=labels_arr,
            random_state=cfg.random_state,
        )

        train_labels = labels_arr[train_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=cfg.val_size,
            stratify=train_labels,
            random_state=cfg.random_state,
        )

        return SplitResult(
            train=sorted(map(int, train_idx)),
            val=sorted(map(int, val_idx)),
            test=sorted(map(int, test_idx)),
        )
    except Exception:
        rng = np.random.default_rng(cfg.random_state)
        indices_by_class: dict[int, list[int]] = {}
        for i, y in enumerate(labels_arr.tolist()):
            indices_by_class.setdefault(int(y), []).append(i)

        train: list[int] = []
        val: list[int] = []
        test: list[int] = []
        for _, cls_indices in indices_by_class.items():
            cls_indices = cls_indices.copy()
            rng.shuffle(cls_indices)
            n_cls = len(cls_indices)
            n_test = int(round(n_cls * cfg.test_size))
            remaining = cls_indices[n_test:]
            n_val = int(round(len(remaining) * cfg.val_size))

            test += cls_indices[:n_test]
            val += remaining[:n_val]
            train += remaining[n_val:]

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        return SplitResult(train=train, val=val, test=test)
