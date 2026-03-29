from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from COCA_processor import COCAProcessor
from COCA_resampler import COCAResampler


def write_predict_metadata(resampled_index_csv: Path, out_csv: Path) -> Path:
    df = pd.read_csv(resampled_index_csv)
    rows = []
    for _, row in df.iterrows():
        subject_id = str(row.get("patient_id") or row.get("scan_id"))
        rows.append(
            {
                "subject_id": subject_id,
                "image": str(row["image_path"]),
                "label": int(row.get("label", 0)),
                "kind": "nifti",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def create_qc_previews(resampled_index_csv: Path, out_dir: Path, limit: int = 25) -> Path:
    df = pd.read_csv(resampled_index_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.head(limit).iterrows():
        scan_id = str(row["scan_id"])
        img = sitk.ReadImage(str(row["image_path"]))
        seg = sitk.ReadImage(str(row["seg_path"]))

        img_arr = sitk.GetArrayFromImage(img)
        seg_arr = sitk.GetArrayFromImage(seg)

        if img_arr.ndim != 3:
            continue

        z = int(img_arr.shape[0] // 2)
        img_u8 = _normalize_to_u8(img_arr[z])
        seg_slice = (seg_arr[z] > 0).astype(np.uint8)

        overlay = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        overlay[seg_slice > 0] = (0, 0, 255)

        side_by_side = np.concatenate([cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR), overlay], axis=1)
        cv2.imwrite(str(out_dir / f"{scan_id}_qc.png"), side_by_side)

    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COCA end-to-end processor + resampler + metadata builder")
    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument(
        "--source-root",
        default=r"I:\My Drive\GSoC_PrediCT\data_raw\dicom",
        help="COCA source root containing Gated_release_final and deidentified_nongated",
    )
    p.add_argument("--mode", choices=["full", "process", "resample"], default="full")
    p.add_argument("--dicom-root", default="")
    p.add_argument("--xml-root", default="")
    p.add_argument("--canonical-root", default="")
    p.add_argument("--scan-index-csv", default="")
    p.add_argument("--resampled-root", default="")
    p.add_argument("--target-spacing", nargs=3, type=float, default=[0.7, 0.7, 3.0], metavar=("X", "Y", "Z"))
    p.add_argument("--metadata-out", default="")
    p.add_argument("--make-previews", action="store_true")
    p.add_argument("--preview-dir", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(args.project_root).resolve()
    source_root = Path(args.source_root)

    dicom_root = Path(args.dicom_root) if args.dicom_root else source_root / "Gated_release_final" / "patient"
    xml_root = Path(args.xml_root) if args.xml_root else source_root / "Gated_release_final" / "calcium_xml"

    canonical_root = Path(args.canonical_root) if args.canonical_root else project_root / "outputs" / "coca" / "canonical"
    resampled_root = Path(args.resampled_root) if args.resampled_root else project_root / "outputs" / "coca" / "resampled"

    metadata_out = Path(args.metadata_out) if args.metadata_out else project_root / "data" / "metadata_coca_resampled.csv"
    preview_dir = Path(args.preview_dir) if args.preview_dir else project_root / "outputs" / "coca" / "qc_previews"

    scan_index_csv = Path(args.scan_index_csv) if args.scan_index_csv else canonical_root / "tables" / "scan_index.csv"

    if args.mode in {"full", "process"}:
        processor = COCAProcessor(
            project_root=str(project_root),
            dicom_root=str(dicom_root),
            xml_root=str(xml_root),
            output_root=str(canonical_root),
        )
        scan_index_csv = processor.process_all()

    if args.mode in {"full", "resample"}:
        resampler = COCAResampler(
            scan_index_csv=str(scan_index_csv),
            output_dir=str(resampled_root),
            target_spacing=args.target_spacing,
        )
        resampled_index_csv = resampler.run()

        metadata_csv = write_predict_metadata(resampled_index_csv, metadata_out)
        print(f"Wrote PrediCT metadata CSV: {metadata_csv}")

        if args.make_previews:
            out = create_qc_previews(resampled_index_csv, preview_dir)
            print(f"Wrote QC previews to: {out}")


if __name__ == "__main__":
    main()
