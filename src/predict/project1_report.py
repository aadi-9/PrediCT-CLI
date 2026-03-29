from __future__ import annotations

from pathlib import Path
from typing import Any


def build_project1_model_choice_justification(payload: dict[str, Any] | None = None) -> str:  
    if payload is None:
        payload = {}
        
    spacing = payload.get("target_spacing_zyx", (0.7, 0.7, 3.0))
    hu = payload.get("hu_window", (-200, 800))
    
    parts = []
    parts.append("Preprocessing Strategy:")
    parts.append(f"We applied HU windowing ({hu[0]} to {hu[1]}) to retain clinically relevant cardiac structures while suppressing noise and extreme values. The dataset was resampled to anisotropic spacing ({spacing[2]}×{spacing[1]}×{spacing[0]} mm) to balance spatial resolution and computational efficiency, preserving axial detail critical for coronary calcium detection. Data augmentation (flips, rotations, zoom, noise) was used to improve model generalization across anatomical variability.")
    parts.append("\nModel Choice:")
    parts.append("A lightweight 3D U-Net was selected for its strong performance in volumetric medical segmentation with relatively low computational overhead. Compared to TotalSegmentator, this approach provides faster inference while maintaining high Dice similarity (>0.85 target). The model is suitable for scalable preprocessing pipelines and downstream radiomics integration.")

    text = "\n".join(parts).strip()
    return text + "\n"


def write_project1_model_choice_justification(payload: dict[str, Any], out_path: Path) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_project1_model_choice_justification(payload), encoding="utf-8")
    return out_path

