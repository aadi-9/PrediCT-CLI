from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


class COCAProcessor:
    def __init__(
        self,
        project_root: str,
        dicom_root: str,
        xml_root: str,
        output_root: str,
        min_dcm_files: int = 5,
    ):
        self.project_root = Path(project_root)
        self.dicom_root = Path(dicom_root)
        self.xml_root = Path(xml_root)
        self.output_root = Path(output_root)
        self.min_dcm_files = int(min_dcm_files)

        self.out_images_base = self.output_root / "images"
        self.out_tables = self.output_root / "tables"

        self.out_images_base.mkdir(parents=True, exist_ok=True)
        self.out_tables.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_stable_id(*parts: str, n: int = 12) -> str:
        h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
        return h[:n]

    @staticmethod
    def _patient_id_from_series(series_dir: Path) -> str:
        for parent in [series_dir, *series_dir.parents]:
            if parent.name.isdigit():
                return parent.name
        return series_dir.name

    def parse_plist_filled(self, xml_path: Path, image_shape: tuple[int, int, int]) -> tuple[np.ndarray, list[int]]:
        mask = np.zeros(image_shape, dtype=np.uint8)
        segmented_slices: set[int] = set()
        total_z, total_y, total_x = image_shape

        if not xml_path.exists():
            return mask, []

        try:
            with xml_path.open("rb") as f:
                data = plistlib.load(f)

            images = data.get("Images", [])
            for img_entry in images:
                z = int(img_entry.get("ImageIndex", -1))
                if z < 0 or z >= total_z:
                    continue

                rois = img_entry.get("ROIs", [])
                for roi in rois:
                    points_str = roi.get("Point_px", [])
                    if not points_str:
                        continue

                    poly_points: list[list[float]] = []
                    for p_str in points_str:
                        cleaned = p_str.replace("(", "").replace(")", "")
                        parts = cleaned.split(",")
                        if len(parts) == 2:
                            poly_points.append([float(parts[0]), float(parts[1])])

                    if not poly_points:
                        continue

                    pts = np.array(poly_points, dtype=np.int32)
                    temp_slice = np.zeros((total_y, total_x), dtype=np.uint8)

                    if len(pts) > 2:
                        cv2.fillPoly(temp_slice, [pts], 1)
                    else:
                        for p in pts:
                            if 0 <= p[0] < total_x and 0 <= p[1] < total_y:
                                temp_slice[int(p[1]), int(p[0])] = 1

                    if np.any(temp_slice):
                        mask[z, :, :] = np.logical_or(mask[z, :, :], temp_slice).astype(np.uint8)
                        segmented_slices.add(z)

        except Exception as e:  # noqa: BLE001
            print(f"[PARSING ERROR] {xml_path.name}: {e}")

        return mask, sorted(segmented_slices)

    def discover_series(self) -> list[Path]:
        print(f"Scanning {self.dicom_root} for DICOM series...")
        if not self.dicom_root.exists():
            return []

        candidates: list[Path] = []
        for d in self.dicom_root.rglob("*"):
            if not d.is_dir():
                continue
            count = sum(1 for _ in d.glob("*.dcm"))
            if count >= self.min_dcm_files:
                candidates.append(d)

        # Prefer leaf-most candidates to avoid duplicate parent/child series.
        series_dirs = []
        for cand in sorted(candidates):
            has_child_series = any(other != cand and cand in other.parents for other in candidates)
            if not has_child_series:
                series_dirs.append(cand)

        return series_dirs

    def process_all(self) -> Path:
        series_dirs = self.discover_series()
        print(f"Found {len(series_dirs)} valid series. Starting processing...")

        rows: list[dict[str, object]] = []
        for s_dir in tqdm(series_dirs, desc="Processing Scans"):
            patient_id = self._patient_id_from_series(s_dir)
            xml_path = self.xml_root / f"{patient_id}.xml"

            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(s_dir))
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                img_array = sitk.GetArrayFromImage(image)
                mask_array, seg_slices = self.parse_plist_filled(xml_path, img_array.shape)
                voxel_count = int(np.sum(mask_array))

                scan_id = self.generate_stable_id(str(s_dir.resolve()), patient_id)
                scan_folder = self.out_images_base / scan_id
                scan_folder.mkdir(parents=True, exist_ok=True)

                img_path = scan_folder / f"{scan_id}_img.nii.gz"
                seg_path = scan_folder / f"{scan_id}_seg.nii.gz"
                meta_path = scan_folder / f"{scan_id}_meta.json"

                sitk.WriteImage(image, str(img_path), useCompression=True)

                mask_image = sitk.GetImageFromArray(mask_array)
                mask_image.CopyInformation(image)
                sitk.WriteImage(mask_image, str(seg_path), useCompression=True)

                meta = {
                    "scan_id": scan_id,
                    "patient_id": patient_id,
                    "calcium_voxels": voxel_count,
                    "slices_with_calcium": seg_slices,
                    "original_path": str(s_dir),
                    "xml_path": str(xml_path),
                    "image_path": str(img_path),
                    "seg_path": str(seg_path),
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                rows.append(
                    {
                        "patient_id": patient_id,
                        "scan_id": scan_id,
                        "voxels": voxel_count,
                        "num_slices": len(seg_slices),
                        "folder_path": str(scan_folder),
                        "original_series_path": str(s_dir),
                        "image_path": str(img_path),
                        "seg_path": str(seg_path),
                    }
                )
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Patient {patient_id} ({s_dir}): {e}")

        out_csv = self.out_tables / "scan_index.csv"
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"Processing complete. Wrote {out_csv}")
        else:
            print("No scans processed successfully.")
        return out_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COCA DICOM/XML processor -> NIfTI images + masks")
    p.add_argument("--project-root", required=True)
    p.add_argument("--dicom-root", required=True)
    p.add_argument("--xml-root", required=True)
    p.add_argument("--output-root", default="")
    p.add_argument("--min-dcm-files", type=int, default=5)
    return p


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root)
    output_root = Path(args.output_root) if args.output_root else project_root / "outputs" / "coca" / "canonical"

    processor = COCAProcessor(
        project_root=str(project_root),
        dicom_root=str(Path(args.dicom_root)),
        xml_root=str(Path(args.xml_root)),
        output_root=str(output_root),
        min_dcm_files=args.min_dcm_files,
    )
    processor.process_all()


if __name__ == "__main__":
    main()
