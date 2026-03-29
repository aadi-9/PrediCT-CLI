# `hu_windowing()`

**Source:** `src/predict/preprocess.py`

Clips a 3D CT volume to a specified Hounsfield Unit (HU) range and linearly normalises the result to a target floating-point interval.

---

## Signature

```python
def hu_windowing(
    image_zyx:   np.ndarray,
    lower_bound: float,
    upper_bound: float,
    out_min:     float = 0.0,
    out_max:     float = 1.0,
) -> np.ndarray
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_zyx` | `np.ndarray` | — | 3D input array in `(Z, Y, X)` axis order; typically HU values |
| `lower_bound` | `float` | — | Lower HU clip value; all values below this are set to `lower_bound` |
| `upper_bound` | `float` | — | Upper HU clip value; all values above this are set to `upper_bound` |
| `out_min` | `float` | `0.0` | Minimum value of the output range |
| `out_max` | `float` | `1.0` | Maximum value of the output range |

---

## Return Value

| Type | Description |
|---|---|
| `np.ndarray` | `float32` array of the same shape as `image_zyx`, with values in `[out_min, out_max]` |

---

## Algorithm

```
1. clipped = clip(image_zyx, lower_bound, upper_bound)
2. normalised = (clipped - lower_bound) / (upper_bound - lower_bound)
3. scaled = normalised * (out_max - out_min) + out_min
4. return scaled.astype(float32)
```

This is a linear min-max normalisation over the clipped window.

---

## In the Data Pipeline

`hu_windowing()` is the lowest-level preprocessing function. It is called by [`apply_hu_window()`](apply_hu_window.md), which wraps it to accept a [`Volume`](../io/Volume.md) and [`HUWindowConfig`](../config/HUWindowConfig.md).

```
run_pipeline()
  └─► apply_hu_window(volume, cfg)
        └─► hu_windowing(volume.array, cfg.lower, cfg.upper)   ← here
              └─► float32 ndarray in [0, 1]
```

---

## Usage Example

```python
import numpy as np
from predict.preprocess import hu_windowing

# Simulate a cardiac CT slice
ct = np.array([[[-500, 0, 100, 500]]], dtype=np.float32)

windowed = hu_windowing(ct, lower_bound=-200.0, upper_bound=400.0)
print(windowed)
# [[[ 0.     0.333  0.5    1.   ]]]
# -500 clipped to -200 → 0.0
#    0 → (0 - (-200)) / 600 = 0.333
#  100 → 300/600 = 0.5
#  500 clipped to 400 → 1.0

# Custom output range [−1, 1]
windowed = hu_windowing(ct, -200.0, 400.0, out_min=-1.0, out_max=1.0)
```

---

## Notes

- The output dtype is always `float32` regardless of the input dtype.
- If `lower_bound == upper_bound`, a division-by-zero will occur. Ensure the window has non-zero width.
- Values exactly at `lower_bound` map to `out_min`; values exactly at `upper_bound` map to `out_max`.
- The function is **pure** — it does not modify the input array.

---

## Related

- [`apply_hu_window()`](apply_hu_window.md) — higher-level wrapper that accepts a `Volume` and `HUWindowConfig`
- [`HUWindowConfig`](../config/HUWindowConfig.md) — stores the `lower` and `upper` bounds
- [`resample_volume()`](resample_volume.md) — typically applied after HU windowing
