# `HUWindowConfig`

**Source:** `src/predict/config.py`

Frozen dataclass that defines the Hounsfield Unit (HU) clipping window applied to cardiac CT volumes during preprocessing.

---

## Signature

```python
@dataclass(frozen=True)
class HUWindowConfig:
    lower: float = -200.0
    upper: float =  400.0
```

---

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `lower` | `float` | `-200.0` | Lower HU boundary; values below this are clipped to `lower` |
| `upper` | `float` | `400.0` | Upper HU boundary; values above this are clipped to `upper` |

---

## Description

HU windowing suppresses irrelevant tissue intensities and normalises the dynamic range before feeding volumes into a neural network. The defaults (`-200` to `400` HU) are chosen for **cardiac CT**:

| Tissue | Typical HU range |
|---|---|
| Lung / air | -1000 to -500 |
| Fat | -150 to -50 |
| Soft tissue / myocardium | 20 to 80 |
| Blood (enhanced) | 100 to 300 |
| Calcium / bone | 200 to 1000+ |

The window `[-200, 400]` retains soft tissue, blood pools, and moderate calcium while discarding air and dense bone, which are not informative for most cardiac tasks.

After clipping, [`hu_windowing()`](../preprocess/hu_windowing.md) normalises the window to `[0.0, 1.0]`.

---

## In the Data Pipeline

`HUWindowConfig` is passed to [`apply_hu_window()`](../preprocess/apply_hu_window.md) inside [`run_pipeline()`](../pipeline/run_pipeline.md).

```
build_parser() → --hu-bounds lower upper
  └─► HUWindowConfig(lower, upper)
        └─► run_pipeline(hu_cfg=...)
              └─► apply_hu_window(volume, cfg=hu_cfg)   ← consumed here
                    └─► hu_windowing(array, lower, upper)
```

The window bounds are also recorded in `dataset_stats.json` and [`justification.txt`](../report/build_justification_text.md) for reproducibility.

---

## Usage Example

```python
from predict.config import HUWindowConfig

# Default cardiac window
cfg = HUWindowConfig()

# Wider window to include dense calcium
cfg = HUWindowConfig(lower=-200.0, upper=1000.0)

# Soft-tissue only window
cfg = HUWindowConfig(lower=-100.0, upper=200.0)
```

---

## Notes

- `lower` must be strictly less than `upper`. No runtime check enforces this; violating it will produce an all-zero (or all-one) normalised volume.
- The dataclass is `frozen=True`; create a new instance to change settings.
- The `lower` and `upper` values are stored in `stats["hu_window"]` within the output `dataset_stats.json`.

---

## Related

- [`hu_windowing()`](../preprocess/hu_windowing.md) — core clipping and normalisation function
- [`apply_hu_window()`](../preprocess/apply_hu_window.md) — applies `HUWindowConfig` to a `Volume`
- [`ResampleConfig`](ResampleConfig.md) — companion preprocessing configuration
- [`run_pipeline()`](../pipeline/run_pipeline.md) — assembles `HUWindowConfig` from `--hu-bounds` CLI flag
- [`build_justification_text()`](../report/build_justification_text.md) — includes HU bounds in the written report
