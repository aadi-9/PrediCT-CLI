# `build_monai_transforms()`

**Source:** `src/predict/augment.py`

Builds a MONAI augmentation pipeline for 3D medical volumes, returning a composed callable that applies random rotation, random flipping, and random Gaussian noise.

---

## Signature

```python
def build_monai_transforms(
    enable:         bool  = True,
    prob:           float = 0.5,
    rotate_degrees: float = 15.0,
) -> Callable[[Any], Any] | None
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `True` | If `False`, returns `None` immediately (no augmentation) |
| `prob` | `float` | `0.5` | Probability of applying each individual transform |
| `rotate_degrees` | `float` | `15.0` | Maximum rotation magnitude in degrees for `RandRotate` |

---

## Return Value

| Condition | Return value |
|---|---|
| `enable=True` and MONAI is installed | `monai.transforms.Compose` instance |
| `enable=False` | `None` |
| `enable=True` but MONAI is not installed | `None` (with a warning logged) |

---

## Transforms Applied

When enabled and MONAI is available, the returned `Compose` applies the following transforms **in order**, each independently gated by `prob`:

| Transform | Description | Controlled by |
|---|---|---|
| `RandRotate` | Random 3D rotation around all axes, up to ±`rotate_degrees` degrees | `prob`, `rotate_degrees` |
| `RandFlip(spatial_axis=0)` | Random flip along the Z (axial) axis | `prob` |
| `RandFlip(spatial_axis=1)` | Random flip along the Y (coronal) axis | `prob` |
| `RandFlip(spatial_axis=2)` | Random flip along the X (sagittal) axis | `prob` |
| `RandGaussianNoise` | Additive Gaussian noise with mean 0, std 0.01 | `prob` |

All transforms are configured for **3D tensors** with a channel dimension `(C, Z, Y, X)`.

---

## In the Data Pipeline

`build_monai_transforms()` is called once during [`run_pipeline()`](../pipeline/run_pipeline.md) and passed as the `transform` argument to the training `VolumeDataset`. Validation and test datasets always receive `transform=None`.

```
run_pipeline(enable_augmentation=True)
  └─► build_monai_transforms(enable=True)   ← here
        └─► Compose([RandRotate, RandFlip×3, RandGaussianNoise])
              └─► VolumeDataset(train_records, transform=augment_fn)
                    └─► __getitem__: transform(tensor) applied per sample
```

---

## Usage Example

```python
import torch
from predict.augment import build_monai_transforms

# Default augmentation
aug = build_monai_transforms()

if aug is not None:
    tensor = torch.rand(1, 128, 128, 128)  # (C, Z, Y, X)
    augmented = aug(tensor)
    print(augmented.shape)  # (1, 128, 128, 128)
else:
    print("MONAI not available; augmentation disabled")

# Aggressive augmentation (higher probability, larger rotation)
aug_strong = build_monai_transforms(enable=True, prob=0.7, rotate_degrees=30.0)

# Disable augmentation
aug_none = build_monai_transforms(enable=False)
assert aug_none is None
```

---

## Notes

> **Warning:** MONAI is an **optional** dependency. If it is not installed, this function returns `None` silently. Install with `pip install monai`.

> **Note:** Augmentation is applied only to the **training set**. [`run_pipeline()`](../pipeline/run_pipeline.md) always passes `transform=None` to the validation and test datasets.

- Rotation and flip augmentations are appropriate for cardiac CT, where the scanner orientation is relatively consistent but small angular variations exist between acquisitions.
- Gaussian noise augmentation (`std=0.01` on `[0, 1]`-normalised data) is mild by design; it helps prevent overfitting without degrading image quality.
- The transforms expect a `(C, Z, Y, X)` tensor (with a channel dimension), matching the output of `VolumeDataset.__getitem__()`.

---

## Related

- [`VolumeDataset`](../dataset/VolumeDataset.md) — receives the returned callable as its `transform` argument
- [`run_pipeline()`](../pipeline/run_pipeline.md) — calls `build_monai_transforms()` based on the `enable_augmentation` flag
- [`apply_hu_window()`](../preprocess/apply_hu_window.md) and [`resample_volume()`](../preprocess/resample_volume.md) — preprocessing steps applied before augmentation
