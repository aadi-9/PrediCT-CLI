from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def flatten_dicom_folders(root_dir: Path, apply_changes: bool = False) -> dict[str, int]:
    patient_folders = [p for p in root_dir.iterdir() if p.is_dir() and p.name.isdigit()]

    moved = 0
    collisions = 0
    touched_patients = 0

    for patient_dir in tqdm(patient_folders, desc="Flattening"):
        all_dcms = list(patient_dir.rglob("*.dcm"))
        patient_moved = 0

        for dcm_path in all_dcms:
            if dcm_path.parent == patient_dir:
                continue

            target_path = patient_dir / dcm_path.name
            if target_path.exists():
                collisions += 1
                target_path = patient_dir / f"{dcm_path.parent.name}_{dcm_path.name}"

            if apply_changes:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dcm_path), str(target_path))
            moved += 1
            patient_moved += 1

        if patient_moved > 0:
            touched_patients += 1

        if apply_changes:
            for subfolder in list(patient_dir.iterdir()):
                if subfolder.is_dir():
                    try:
                        shutil.rmtree(subfolder)
                    except Exception:
                        pass

    return {
        "patient_folders": len(patient_folders),
        "touched_patients": touched_patients,
        "moved_files": moved,
        "filename_collisions": collisions,
        "apply_changes": int(apply_changes),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flatten nested gated COCA patient folders")
    p.add_argument("--patient-root", required=True, help="Path to gated patient directory")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes in-place. Without this flag, only a dry-run summary is printed.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    patient_root = Path(args.patient_root)
    if not patient_root.exists():
        raise SystemExit(f"Patient root not found: {patient_root}")

    stats = flatten_dicom_folders(patient_root, apply_changes=bool(args.apply))
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] flatten summary: {stats}")


if __name__ == "__main__":
    main()
