# PrediCT COCA Preprocessing Pipeline

This repository provides a full preprocessing pipeline for the PrediCT common COCA task:

- patient-level metadata loading and validation
- deterministic CT preprocessing (resampling + HU windowing)
- stratified train/val/test split
- optional training oversampling
- optional export of processed `.npy` volumes
- automatic artifact generation (stats, split manifest, processed manifest, justification report)

The Stanford COCA dataset is not included in this repository.

## Evaluation Task Alignment (Common Task)

Goal: build a preprocessing and data loading pipeline tailored to your target project.

Tasks covered by this repo:

1. Download COCA and resample to a consistent geometry (either using the upstream COCA scripts, or this repo’s direct DICOM → resample → `.npy` export).
2. Implement preprocessing and data handling:
   - HU windowing for cardiac CT
   - task-appropriate augmentation hooks
   - stratified train/val/test split (patient-level)
   - optional oversampling for class imbalance
3. Create an efficient data loader (PyTorch Dataset/DataLoader).
4. Tailor to your Segmentation project

Deliverables produced by this repo:

- Pipeline code: [src/predict/pipeline.py](file:///c:/Users/aadya.AP-WIN11-DT/projects/PrediCT/src/predict/pipeline.py)
- Data loader: [src/predict/dataset.py](file:///c:/Users/aadya.AP-WIN11-DT/projects/PrediCT/src/predict/dataset.py)
- Written justification (1–2 paragraphs): `outputs/justification.txt` (generated via CLI)
- Dataset statistics: `outputs/dataset_stats.json` and `outputs/splits.json`

## Project 1 (Radiomics & Phenotyping) — Heart segmentation model

Specific task steps and deliverables are documented here:

- [project1_heart_segmentation.md](file:///c:/Users/aadya.AP-WIN11-DT/projects/PrediCT/docs/project1_heart_segmentation.md)

Quickstart (recommended order):

1. Validate and preprocess COCA (Common Task prerequisites):
   - Generate metadata for all patients:
     - `predict make-metadata --out-csv data\metadata_all.csv --default-label 0`
   - Validate DICOM readability (creates a clean list):
     - `predict validate-metadata --metadata-csv data\metadata_all.csv --out-clean-csv data\metadata_clean.csv --out-report-csv outputs\dicom_validation_report.csv --mode deep`
2. Generate “ground truth” heart masks on 30–50 scans with TotalSegmentator:
   - Install: `pip install TotalSegmentator`
   - Activate license (do not commit license numbers): `totalseg_set_license -l "<YOUR_LICENSE_NUMBER>"`
   - Run notebook: `notebooks/10_project1_totalseg_groundtruth.ipynb`
3. Train a lightweight coarse heart mask model (faster than TotalSegmentator):
   - Run notebook: `notebooks/11_project1_train_unet_coarse_mask.ipynb`
   - Outputs: `outputs/project1/model.pt`, `outputs/project1/train_log.csv`, `outputs/project1/model_justification.txt`
4. Evaluate Dice and inference time (target Dice > 0.85):
   - Run notebook: `notebooks/12_project1_evaluate_dice_and_time.ipynb`
   - Outputs: `outputs/project1/eval_metrics.csv`, `outputs/project1/eval_summary.json`

References:

- TotalSegmentator docs (installation, DICOM folder input, `--roi_subset`): https://github.com/wasserth/TotalSegmentator

## 1) Prerequisites

- Python 3.10+ (recommended 3.12)
- Windows PowerShell (commands below are PowerShell)
- Disk space for raw DICOM and processed `.npy` outputs

## 2) Setup

### Create and activate virtual environment

```powershell
cd c:\Users\aadya.AP-WIN11-DT\projects\PrediCT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS/Linux quick setup

Use the provided shell script to create a fresh venv, repair/upgrade pip, and install the project in editable mode:

```bash
cd /path/to/PrediCT
bash setup_venv.sh
source .venv/bin/activate
```

Script reference: [setup_venv.sh](file:///c:/Users/aadya.AP-WIN11-DT/projects/PrediCT/setup_venv.sh)

### Install package

```powershell
pip install -e .
```

Install medical/image extras (recommended for DICOM preprocessing):

```powershell
pip install -e ".[medical,ml]"
```

On macOS/Linux after `source .venv/bin/activate`, the same extras command applies:

```bash
pip install -e ".[medical,ml]"
```

Install COCA script dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## 3) Prepare data

### Raw COCA patient folder root (default)

Default raw root in this project:

Windows:

`I:\My Drive\GSoC_PrediCT\data_raw\dicom\Gated_release_final\patient`

Colab/Drive default raw root:

`/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient`

Override via:

- env var: `PREDICT_RAW_DIR`
- CLI: `predict pipeline --raw-dir "<path>"`

Windows PowerShell:

```powershell
$env:PREDICT_RAW_DIR = "I:\My Drive\GSoC_PrediCT\data_raw\dicom\Gated_release_final\patient"
```

Colab (Python):

```python
import os
os.environ["PREDICT_RAW_DIR"] = "/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient"
```

### Expected project folder layout

```text
data/
  metadata.csv
  processed/
outputs/
```

Create directories if they do not exist:

```powershell
New-Item -ItemType Directory -Force data\processed | Out-Null
New-Item -ItemType Directory -Force outputs | Out-Null
```

### Prepare metadata CSV

Use `data_example/metadata.csv` as your template.

Required columns:

- `subject_id`
- `image` (path to subject DICOM folder or `.npy` file)
- `label` (patient-level class label as integer)
- `kind` (`dicom_series`, `numpy`, or `nifti`; defaults to `dicom_series` if omitted)

Example:

```csv
subject_id,image,label,kind
0,0,0,dicom_series
1,1,1,dicom_series
```

Notes:

- Keep one row per subject for patient-level splitting.
- Keep labels consistent for each subject.
  - `image` can be absolute, relative to the repo, or relative to the configured raw root (`PREDICT_RAW_DIR` / default).

## 3A) COCA raw source workflow (already-downloaded dataset)

### Colab (Google Drive paths)

Use Drive-backed paths so outputs persist across sessions:

```python
from pathlib import Path
import os

# Raw COCA patient folder root
raw_dir = Path("/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient")
os.environ["PREDICT_RAW_DIR"] = str(raw_dir)

# Output locations on Drive
clean_csv = Path("/content/drive/MyDrive/GSoC_PrediCT/data/metadata_clean.csv")
report_csv = Path("/content/drive/MyDrive/GSoC_PrediCT/outputs/dicom_validation_report.csv")
report_csv.parent.mkdir(parents=True, exist_ok=True)
```

CLI examples with Drive paths:

```bash
# Validate and write to Drive
predict validate-metadata \
  --metadata-csv data/metadata_all.csv \
  --out-clean-csv /content/drive/MyDrive/GSoC_PrediCT/data/metadata_clean.csv \
  --out-report-csv /content/drive/MyDrive/GSoC_PrediCT/outputs/dicom_validation_report.csv \
  --mode deep

# Full pipeline with Drive-backed outputs
predict pipeline \
  --project-root . \
  --metadata-csv /content/drive/MyDrive/GSoC_PrediCT/data/metadata_clean.csv \
  --export-processed \
  --processed-dir /content/drive/MyDrive/GSoC_PrediCT/data/processed \
  --stats-path /content/drive/MyDrive/GSoC_PrediCT/outputs/dataset_stats.json \
  --split-manifest /content/drive/MyDrive/GSoC_PrediCT/outputs/splits.json \
  --processed-manifest /content/drive/MyDrive/GSoC_PrediCT/outputs/processed_manifest.csv \
  --justification-path /content/drive/MyDrive/GSoC_PrediCT/outputs/justification.txt
```

For your downloaded source at:

`/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient`

Run the extracted COCA scripts under `scripts/coca_upstream/COCA_scripts`.

## 3B) Step-by-step notebooks (recommended for debugging)

If you are seeing many per-patient DICOM errors, run these notebooks in order:

- `notebooks/00_Common_Task.ipynb` (single end-to-end wrapper)


Or use CLI equivalents:

```powershell
predict make-metadata --raw-dir "I:\My Drive\GSoC_PrediCT\data_raw\dicom\Gated_release_final\patient" --out-csv data\metadata_all.csv --default-label 0
predict validate-metadata --raw-dir "I:\My Drive\GSoC_PrediCT\data_raw\dicom\Gated_release_final\patient" --metadata-csv data\metadata_all.csv --out-clean-csv data\metadata_clean.csv --out-report-csv outputs\dicom_validation_report.csv --deep
predict pipeline --metadata-csv data\metadata_clean.csv --export-processed
```

Colab equivalents (Drive-backed outputs):

```bash
predict make-metadata \
  --raw-dir /content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient \
  --out-csv data/metadata_all.csv \
  --default-label 0

predict validate-metadata \
  --metadata-csv data/metadata_all.csv \
  --out-clean-csv /content/drive/MyDrive/GSoC_PrediCT/data/metadata_clean.csv \
  --out-report-csv /content/drive/MyDrive/GSoC_PrediCT/outputs/dicom_validation_report.csv \
  --mode deep

predict pipeline \
  --metadata-csv /content/drive/MyDrive/GSoC_PrediCT/data/metadata_clean.csv \
  --export-processed \
  --processed-dir /content/drive/MyDrive/GSoC_PrediCT/data/processed
```

### Step 1: (Optional) dry-run and apply gated unnester

```powershell
python scripts/coca_upstream/COCA_scripts/unnester.py `
  --patient-root "/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient"

python scripts/coca_upstream/COCA_scripts/unnester.py `
  --patient-root "/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient" `
  --apply
```

### Step 2: Run COCA processor + resampler + metadata build

This writes:
- canonical NIfTI + masks: `outputs/coca/canonical/`
- resampled NIfTI + masks: `outputs/coca/resampled/`
- PrediCT metadata CSV: `data/metadata_coca_resampled.csv`
- optional QC PNGs: `outputs/coca/qc_previews/`

```powershell
python scripts/coca_upstream/COCA_scripts/COCA_pipeline.py `
  --project-root . `
  --source-root "/content/drive/MyDrive/GSoC_PrediCT/data_raw/dicom/Gated_release_final/patient" `
  --mode full `
  --target-spacing 0.7 0.7 3.0 `
  --make-previews
```

### Step 3: Use generated metadata in PrediCT pipeline

```powershell
predict pipeline `
  --project-root . `
  --metadata-csv data/metadata_coca_resampled.csv `
  --stats-path outputs/coca/dataset_stats.json `
  --split-manifest outputs/coca/splits.json `
  --justification-path outputs/coca/justification.txt
```

3D Slicer viewing: load each `*_img.nii.gz` and matching `*_seg.nii.gz` together and mark the segmentation as a segmentation (not scalar volume).

## 4) Run pipeline

### Dry-run (sanity check)

This validates wiring and writes stats/split/report artifacts without full data loading.

```powershell
predict --help
predict pipeline `
  --dry-run `
  --project-root . `
  --stats-path outputs/dataset_stats.json `
  --split-manifest outputs/splits.json `
  --justification-path outputs/justification.txt
```

### Full preprocessing + export run

```powershell
predict pipeline `
  --project-root . `
  --metadata-csv data/metadata.csv `
  --export-processed `
  --processed-dir data/processed `
  --stats-path outputs/dataset_stats.json `
  --split-manifest outputs/splits.json `
  --processed-manifest outputs/processed_manifest.csv `
  --justification-path outputs/justification.txt `
  --resample-spacing 1.0 1.0 1.0 `
  --hu-bounds -200 400
```

Colab (Drive-backed) full run:

```bash
predict pipeline \
  --project-root . \
  --metadata-csv /content/drive/MyDrive/GSoC_PrediCT/data/metadata_clean.csv \
  --export-processed \
  --processed-dir /content/drive/MyDrive/GSoC_PrediCT/data/processed \
  --stats-path /content/drive/MyDrive/GSoC_PrediCT/outputs/dataset_stats.json \
  --split-manifest /content/drive/MyDrive/GSoC_PrediCT/outputs/splits.json \
  --processed-manifest /content/drive/MyDrive/GSoC_PrediCT/outputs/processed_manifest.csv \
  --justification-path /content/drive/MyDrive/GSoC_PrediCT/outputs/justification.txt \
  --resample-spacing 1.0 1.0 1.0 \
  --hu-bounds -200 400
```

## 5) Outputs produced

- `outputs/dataset_stats.json`
- `outputs/splits.json`
- `outputs/processed_manifest.csv`
- `outputs/justification.txt`
- `data/processed/*.npy` (when `--export-processed` is enabled)

`outputs/processed_manifest.csv` columns:

- `subject_id`
- `label`
- `kind`
- `processed_path`
- `split`

## 6) Generate report from stats only

If stats already exist, regenerate short justification:

```powershell
python scripts/generate_justification.py `
  --stats outputs/dataset_stats.json `
  --out outputs/justification.txt
```

Use `docs/justification_template.md` as the final submission template.

## 7) Common options

- `--resample-spacing Z Y X` sets target voxel spacing.
- `--hu-bounds LOWER UPPER` sets HU window.
- `--no-oversample-train` disables oversampling.
- `--no-augment` disables augmentation.
- `--processed-dir <path>` changes export location.
- `--split-manifest <path>` and `--processed-manifest <path>` customize manifest output paths.

## 8) Validation checklist

- `outputs/dataset_stats.json` has correct subject count and split sizes.
- `outputs/splits.json` contains every subject exactly once.
- `outputs/processed_manifest.csv` paths exist and match split assignments.
- `outputs/justification.txt` includes preprocessing choices and class/split counts.
