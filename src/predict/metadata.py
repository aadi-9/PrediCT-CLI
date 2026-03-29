from __future__ import annotations

import csv
from pathlib import Path


def _has_dicom_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() == ".dcm":
            return True
    return False


def generate_metadata_csv(
    raw_dir: Path,
    out_csv: Path,
    *,
    default_label: int = 0,
    kind: str = "dicom_series",
) -> Path:
    raw_dir = raw_dir.resolve()
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if kind not in {"dicom_series", "numpy", "nifti", "nii", "nifti_gz"}:
        raise ValueError(f"Unsupported kind: {kind}")

    subject_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    rows: list[dict[str, str]] = []
    for subject_dir in subject_dirs:
        if kind == "dicom_series" and not _has_dicom_files(subject_dir):
            continue
        rows.append(
            {
                "subject_id": subject_dir.name,
                "image": subject_dir.name,
                "label": str(int(default_label)),
                "kind": kind,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "image", "label", "kind"])
        writer.writeheader()
        writer.writerows(rows)

    return out_csv
