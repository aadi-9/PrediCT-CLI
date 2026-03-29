# Issue: Zero Dice Score in 3D Cardiac Segmentation Despite High Inference Speed

## Background & Context
We are developing a 3D cardiac segmentation model intended to provide significant speedups over the TotalSegmentator baseline. 
Initial evaluations show that our custom `SmallUNet3D` model is highly efficient:
- **Model Inference Time (Mean):** ~0.40 seconds (vs ~51.65 seconds for TotalSegmentator)
- **Speedup Ratio:** ~130x faster
- **Dice Score:** **0.0**

Despite the massive inference speedup, the model fails to produce any valid foreground segmentation masks. The outputs are blank, indicating no overlap with the ground truth.

## Problem Statement
The current 3D segmentation model produces a Dice score of 0.0 because all output values fall below the required threshold (0.5). Visual inspections confirm that the model's predictions are empty compared to the ground truth masks. This failure prevents the model from being usable for any downstream tasks.

## Root Cause Analysis
A thorough review of the training pipeline, data preprocessing, and model architecture has identified 5 compounding issues:

1. 🔴 **Critical: Non-differentiable Dice Loss** 
   The loss function currently uses `(p > 0.5).float()`, which completely kills all gradients from the Dice component during backpropagation. Only Binary Cross-Entropy (BCE) is contributing to gradients, which is severely insufficient for datasets with extreme foreground/background class imbalance.
2. 🔴 **Insufficient Training Duration** 
   The model was only trained for 3 epochs. Medical 3D segmentation typically requires 50-200+ epochs to converge properly. *(Note: local GPU constraints make 50+ epochs difficult without overnight training).*
3. 🟡 **Inadequate Model Capacity** 
   The custom `SmallUNet3D` model is extremely lightweight (only 85K parameters with an 8 → 16 → 32 channel progression), which is fundamentally insufficient for the complex feature representations needed in 3D cardiac segmentation.
4. 🟡 **Lack of Batch Normalization** 
   The current architecture uses only convolutional layers without Batch Normalization (BN), making the training highly unstable across varying CT intensities.
5. 🟡 **Absence of Learning Rate Scheduling** 
   Using a fixed learning rate (`1e-3`) prevents the optimizer from converging to local minima efficiently.

## Proposed Solution & Action Plan
To resolve these issues and achieve a non-zero, competitive Dice score, the following architectural and pipeline updates must be implemented:

1. **Fix the Loss Function:** Remove the hard thresholding `(p > 0.5).float()` during training to ensure the Dice loss remains differentiable. Consider using `monai.losses.DiceLoss` or `monai.losses.DiceCELoss`.
2. **Upgrade the Architecture:** 
   - Replace the custom `SmallUNet3D` (85K params) with **MONAI's `UNet`** (≈2M params).
   - *Important:* Existing checkpoints (`model.pt`) will no longer be compatible with the new architecture.
3. **Add Batch Normalization:** Ensure the new architecture utilizes Batch/Instance Normalization to stabilize varying CT intensities.
4. **Implement LR Scheduling:** Add a learning rate scheduler (e.g., `ReduceLROnPlateau` or `CosineAnnealingLR`) to improve convergence.
5. **Retrain the Model:** Launch a full retraining from scratch on Google Colab for an adequate number of epochs (at least 50+ epochs). Ensure proper HU windowing and normalization align between training and inference.

## References
- **Evaluation Notebook Run:** [12_project1_evaluate_dice_and_time_v2.ipynb](https://colab.research.google.com/gist/vrAxiom/ff3c9b4a92fe22fd312923447ab2f065/12_project1_evaluate_dice_and_time_v2.ipynb)