# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PrediCT-CLI** is a Python-based medical imaging preprocessing and data loading pipeline for the Stanford COCA cardiac CT evaluation task. It provides end-to-end automation from raw DICOM data to PyTorch-ready DataLoaders.

## Setup & Installation

```bash
# Create virtual environment (PowerShell)
./setup_venv.ps1

# Or manually
python -m venv .venv
source .venv/Scripts/activate  # Windows bash
pip install -e ".[medical,ml]"
pip install -r requirements.txt
```

**Install with extras:**
```bash
pip install -e ".[medical,ml,coca_project1]"  # Project 1 (requires TotalSegmentator)
pip install -e ".[medical,ml,coca_project2]"  # Project 2 (requires pyradiomics)
```

## CLI Commands

The package exposes a `predict` CLI entry point (`predict.cli:main`):

```bash
# Generate metadata CSV from directory structure
predict make-metadata --project-root . --output data_example/metadata.csv

# Validate metadata (fast/shallow/deep modes)
predict validate-metadata --metadata data_example/metadata.csv

# Run full preprocessing pipeline
predict pipeline --project-root . --stats-path outputs/dataset_stats.json

# Dry-run (no file I/O, validates config only)
predict pipeline --dry-run --project-root . --stats-path outputs/dataset_stats.json
```

**Key CLI flags for pipeline:**
- `--spacing Z Y X` — target resampling spacing in mm (default: 1.0 1.0 1.0; Project 1: 3.0 0.7 0.7)
- `--hu-min / --hu-max` — HU window bounds (default: -200/400; Project 1: -200/800)
- `--oversample` — enable random oversampling for class imbalance
- `--augment` — enable MONAI augmentation transforms
- `--raw-dir` — override raw DICOM directory (or set `PREDICT_RAW_DIR` env var)

## Linting

```bash
ruff check src/
ruff format src/
```

## Architecture

### Data Flow

```
Raw DICOM → make-metadata → metadata_all.csv
          → validate-metadata → metadata_clean.csv
          → pipeline → SampleRecord objects
                     → stratified_split() → train/val/test
                     → oversample_minority() (optional)
                     → resample_volume() + apply_hu_window()
                     → VolumeDataset → build_dataloader()
                     → outputs: dataset_stats.json, splits.json,
                                processed_manifest.csv, justification.txt
```

### Core Modules (`src/predict/`)

| Module | Role |
|--------|------|
| `cli.py` | Argument parsing; commands: `pipeline`, `make-metadata`, `validate-metadata` |
| `pipeline.py` | Main orchestrator — loads metadata, assigns splits, calls preprocessing, builds loaders |
| `config.py` | Frozen dataclasses: `ResampleConfig`, `HUWindowConfig`, `SplitConfig`, `LoaderConfig`, `PathsConfig` |
| `dataset.py` | `VolumeDataset` (PyTorch Dataset), `SampleRecord`, `build_dataloader()` |
| `preprocess.py` | `resample_volume()` (scipy zoom), `apply_hu_window()` |
| `io.py` | DICOM/numpy/nifti reading via SimpleITK; volume saving |
| `split.py` | Stratified patient-level train/val/test split (no data leakage) |
| `augment.py` | MONAI dictionary transforms for paired image+mask augmentation |
| `sampling.py` | Random oversampling via imbalanced-learn |
| `validate.py` | DICOM series validation (fast/shallow/deep modes) |
| `project1_pipeline.py` | Training loop: MONAI UNet, DiceCE loss, cosine annealing LR, gradient clipping |
| `project1_eval.py` | Evaluation: Dice score, inference timing, `SmallUNet3D` architecture |
| `project2_radiomics.py` | Radiomics feature extraction via pyradiomics |

### Key Data Structures

- **`SampleRecord`** (frozen dataclass): `subject_id`, `image: Path`, `label: int`, `mask: Optional[Path]`, `kind: str`
- **`Volume`** (frozen dataclass): `array: np.ndarray`, `spacing_zyx: tuple`, `meta: dict`
- **`VolumeDataset.__getitem__`** returns dicts: `{"image": tensor, "mask": tensor}` (segmentation) or `{"image": tensor, "label": int}` (classification) — **not tuples**
- **`PipelineOutputs`**: `train_loader`, `val_loader`, `test_loader`, `stats: dict`, manifest paths

### Metadata CSV Format

Required columns: `subject_id`, `image` (path), `label` (int), `kind` (str, e.g. `"gated"`)
Optional column: `mask` (path, for segmentation tasks)

### Default Configuration Values

| Parameter | Default | Project 1 Override |
|-----------|---------|-------------------|
| Resampling mode | spacing | spacing |
| Target spacing (Z,Y,X mm) | 1.0, 1.0, 1.0 | 3.0, 0.7, 0.7 |
| HU window | -200 to 400 | -200 to 800 |
| Test split | 20% | 20% |
| Val split | 20% of train | 20% of train |
| Random state | 42 | 42 |
| Batch size | 2 | 2 |

### Project 1 Model

MONAI 3D UNet: channels `(32, 64, 128, 256)`, strides `(2, 2, 2)`. The `SmallUNet3D` in `project1_eval.py` is a lighter checkpoint-compatible variant for evaluation only.

## Raw Data Location

Default: `I:\My Drive\GSoC_PrediCT\data_raw\dicom\Gated_release_final\patient`
Override via: `PREDICT_RAW_DIR` environment variable or `--raw-dir` CLI flag.

## Output Artifacts

All written to `--stats-path` parent directory (default `outputs/`):
- `dataset_stats.json` — volume/label distribution stats
- `splits.json` — subject-level train/val/test assignments
- `processed_manifest.csv` — per-sample preprocessing record
- `justification.txt` — auto-generated preprocessing rationale
