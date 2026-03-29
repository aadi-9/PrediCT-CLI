# Project 1 (Radiomics & Phenotyping) — Heart Segmentation Model

Goal: demonstrate medical image segmentation and model evaluation skills by building a lightweight heart mask / heart bounding-box model and comparing against TotalSegmentator.

## Deliverables (for evaluation)

- Model weights (your trained checkpoint)
- Evaluation notebook with Dice scores + inference time comparisons + visualizations
- 1-paragraph justification for model choice

## Step-by-step plan

### Step P1.0 — Prerequisites

- Install dependencies:

```powershell
pip install -e ".[medical,ml,coca_project1]"
```

- Ensure the raw COCA patient root is set:
  - env var: `PREDICT_RAW_DIR`
  - or CLI: `predict pipeline --raw-dir "<path>"`

### Step P1.1 — Create metadata + validate DICOM series

Create a full metadata list:

```powershell
predict make-metadata --out-csv data\metadata_all.csv --default-label 0
```

Validate and generate a clean CSV (recommended to avoid broken/empty folders):

```powershell
predict validate-metadata --metadata-csv data\metadata_all.csv --out-clean-csv data\metadata_clean.csv --out-report-csv outputs\dicom_validation_report.csv --mode deep
```

### Step P1.2 — Choose 30–50 scans for TotalSegmentator “ground truth”

Pick 30–50 subjects from `data/metadata_clean.csv`. Store the selected IDs in a file for reproducibility (e.g., `outputs/project1/selected_subjects.txt`).

### Step P1.3 — Run TotalSegmentator for heart masks (ground truth)

Install TotalSegmentator:

```powershell
pip install TotalSegmentator
```

TotalSegmentator supports using a DICOM folder as input and supports `--roi_subset` to only predict selected classes like `heart` (saves runtime) (see TotalSegmentator documentation)【https://github.com/wasserth/TotalSegmentator】.

License setup:

- Keep your license number out of git and notebooks.
- Set it via a local environment variable and run:

```powershell
totalseg_set_license -l "<YOUR_LICENSE_NUMBER>"
```

Run heart-only segmentation (example for one subject folder):

```powershell
TotalSegmentator -i "<patient_folder>" -o "outputs\project1\totalseg\<subject_id>" --roi_subset heart
```

Record:
- TotalSegmentator inference time per scan
- Output path to the `heart` mask

### Step P1.4 — Train a lightweight model (coarse mask / bbox)

Recommended approach:
- Train a small 3D U-Net on downsampled/resampled volumes (e.g., 1.0×1.0×3.0 mm or 1.5×1.5×3.0 mm) to predict a coarse heart mask.
- Derive a bounding box from the predicted mask for “heart ROI” cropping.

### Step P1.5 — Evaluate Dice and inference time

Evaluate on a held-out test set:
- Dice score against TotalSegmentator heart mask (target > 0.85)
- Compare inference time (your model vs TotalSegmentator)
- Provide qualitative visualizations (axial slices + mask overlay)

## Notebooks

Use the notebooks in `notebooks/`:

- `10_project1_totalseg_groundtruth.ipynb`
- `11_project1_train_unet_coarse_mask.ipynb`
- `12_project1_evaluate_dice_and_time.ipynb`

The training notebook auto-generates:

- `outputs/project1/model_justification.txt` (1-paragraph model choice justification)

Template for editing the wording:

- `docs/project1_model_choice_template.md`

## Notes

- TotalSegmentator is the reference “ground truth” for this evaluation task. It may fail on some subjects if the DICOM series is incomplete or not fully available offline. Validate your metadata first.
- Keep license numbers and any private keys out of git.

Below is a **Windows/PowerShell command-line runbook** to complete **Project 1 from Step P1.4 onward** (train + evaluate), assuming you already finished Step P1.3 (you have TotalSegmentator heart masks).

## Step P1.4 — Train (coarse heart mask model)

**1) Activate venv + install deps**
```powershell
cd c:\Users\aadya.AP-WIN11-DT\projects\PrediCT
.\.venv\Scripts\Activate.ps1
pip install -e ".[medical,ml,coca_project1]"
```

**2) Confirm you have the required ground-truth masks from Step P1.3**
You should have folders like:
- `outputs\project1\totalseg\<subject_id>\heart.nii.gz`

Quick check:
```powershell
dir outputs\project1\totalseg | Select-Object -First 5
dir outputs\project1\totalseg\0 | Select-Object -First 10
```

**3) Run training notebook (two options)**

### Option A (recommended): open Jupyter and run manually
```powershell
pip install notebook
jupyter notebook
```
Then open:
- `notebooks\11_project1_train_unet_coarse_mask.ipynb`

### Option B: run notebook headless from CLI (fully automated)
```powershell
pip install nbconvert
jupyter nbconvert --to notebook --execute notebooks\11_project1_train_unet_coarse_mask.ipynb --output notebooks\11_project1_train_unet_coarse_mask.executed.ipynb
```

**Expected outputs after training**
- `outputs\project1\model.pt`  (model weights)
- `outputs\project1\train_log.csv`
- `outputs\project1\model_justification.txt`  (auto-generated 1-paragraph model choice justification)

---

## Step P1.5 — Evaluate (Dice + inference time + visuals)

### Option A: open and run evaluation notebook
```powershell
jupyter notebook
```
Open:
- `notebooks\12_project1_evaluate_dice_and_time.ipynb`

### Option B: run evaluation notebook from CLI
```powershell
jupyter nbconvert --to notebook --execute notebooks\12_project1_evaluate_dice_and_time.ipynb --output notebooks\12_project1_evaluate_dice_and_time.executed.ipynb
```

**Expected outputs after evaluation**
- `outputs\project1\eval_metrics.csv` (per-subject Dice + times)
- `outputs\project1\eval_summary.json` (mean Dice, mean time, etc.)
- inline visualizations in the executed notebook

---

## GPU note (for training speed)
- Training/evaluation will use GPU **only if** your PyTorch build is CUDA-enabled and `torch.cuda.is_available()` is `True`.

Check:
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If you prefer the simplest GPU path: run the training + evaluation notebooks in **Google Colab with GPU runtime** (your notebooks already follow the same structure).

---
