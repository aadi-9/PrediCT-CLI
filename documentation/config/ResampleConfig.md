# `ResampleConfig`

**Source:** `src/predict/config.py`

Frozen dataclass that controls how 3D medical volumes are spatially resampled during preprocessing.

---

## Signature

```python
@dataclass(frozen=True)
class ResampleConfig:
    mode:           str                       = "spacing"
    target_shape:   tuple[int, int, int]      = (128, 128, 128)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolator:   str                       = "linear"
```

---

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"spacing"` | Resampling strategy: `"shape"` or `"spacing"` |
| `target_shape` | `tuple[int, int, int]` | `(128, 128, 128)` | Output voxel dimensions `(Z, Y, X)` used when `mode="shape"` |
| `target_spacing` | `tuple[float, float, float]` | `(1.0, 1.0, 1.0)` | Target voxel spacing in mm `(Z, Y, X)` used when `mode="spacing"` |
| `interpolator` | `str` | `"linear"` | Interpolation order: `"linear"` (order 1) or `"nearest"` (order 0) |

### `mode` values

| Value | Behaviour |
|---|---|
| `"spacing"` | Resamples the volume so that each voxel corresponds to `target_spacing` mm; output shape varies per subject |
| `"shape"` | Resizes the volume to exactly `target_shape` voxels regardless of physical spacing |

> **Note:** When resampling a **label/segmentation mask**, [`resample_volume()`](../preprocess/resample_volume.md) always forces `interpolator = "nearest"` regardless of this field to avoid introducing non-integer label values.

---

## In the Data Pipeline

`ResampleConfig` is constructed from CLI flags in `_cmd_pipeline` and passed to [`run_pipeline()`](../pipeline/run_pipeline.md), which passes it to [`resample_volume()`](../preprocess/resample_volume.md) for each subject.

```
build_parser() → --resample-mode, --resample-shape, --resample-spacing
  └─► ResampleConfig(mode, target_shape, target_spacing, interpolator)
        └─► run_pipeline(resample_cfg=...)
              └─► resample_volume(volume, cfg=resample_cfg)   ← consumed here
```

---

## Usage Example

```python
from predict.config import ResampleConfig

# Resample to isotropic 1 mm spacing (default)
cfg = ResampleConfig()

# Resample to fixed 128³ volume
cfg = ResampleConfig(mode="shape", target_shape=(128, 128, 64))

# Resample to 2 mm spacing with nearest-neighbour (e.g. for labels)
cfg = ResampleConfig(
    mode="spacing",
    target_spacing=(2.0, 2.0, 2.0),
    interpolator="nearest",
)
```

---

## Notes

- `target_shape` is only relevant when `mode="shape"`; it is ignored in `"spacing"` mode.
- `target_spacing` is only relevant when `mode="spacing"`; it is ignored in `"shape"` mode.
- All tuple values follow **(Z, Y, X)** axis order to match the internal `Volume.array` layout.
- The dataclass is `frozen=True`; create a new instance to change settings.

---

## Related

- [`resample_volume()`](../preprocess/resample_volume.md) — the function that consumes `ResampleConfig`
- [`HUWindowConfig`](HUWindowConfig.md) — companion preprocessing configuration
- [`run_pipeline()`](../pipeline/run_pipeline.md) — assembles `ResampleConfig` from CLI args
