from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


class COCAResampler:
    def __init__(self, scan_index_csv: str, output_dir: str, target_spacing: list[float] | tuple[float, float, float]):
        self.input_csv = Path(scan_index_csv)
        self.output_dir = Path(output_dir)
        self.target_spacing = [float(v) for v in target_spacing]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resample_volume(self, volume: sitk.Image, is_mask: bool = False) -> sitk.Image:
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()

        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(volume.GetDirection())
        resample.SetOutputOrigin(volume.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)

        if is_mask:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(volume)

    def run(self) -> Path:
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Could not find {self.input_csv}. Run the processor first.")

        df = pd.read_csv(self.input_csv)
        print(f"Starting resampling of {len(df)} scans to spacing XYZ={self.target_spacing} mm")

        rows: list[dict[str, object]] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Resampling"):
            scan_id = row["scan_id"]
            input_folder = Path(row["folder_path"])
            resampled_folder = self.output_dir / scan_id
            resampled_folder.mkdir(parents=True, exist_ok=True)

            try:
                img_path = input_folder / f"{scan_id}_img.nii.gz"
                seg_path = input_folder / f"{scan_id}_seg.nii.gz"

                img = sitk.ReadImage(str(img_path))
                seg = sitk.ReadImage(str(seg_path))

                res_img = self.resample_volume(img, is_mask=False)
                res_seg = self.resample_volume(seg, is_mask=True)

                out_img = resampled_folder / f"{scan_id}_img.nii.gz"
                out_seg = resampled_folder / f"{scan_id}_seg.nii.gz"
                sitk.WriteImage(res_img, str(out_img), useCompression=True)
                sitk.WriteImage(res_seg, str(out_seg), useCompression=True)

                rows.append(
                    {
                        "patient_id": row.get("patient_id", ""),
                        "scan_id": scan_id,
                        "voxels": int(row.get("voxels", 0)),
                        "num_slices": int(row.get("num_slices", 0)),
                        "folder_path": str(resampled_folder),
                        "image_path": str(out_img),
                        "seg_path": str(out_seg),
                        "label": int(int(row.get("voxels", 0)) > 0),
                    }
                )
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Failed to resample {scan_id}: {e}")

        out_csv = self.output_dir / "scan_index_resampled.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Resampling complete. Wrote {out_csv}")
        return out_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resample COCA NIfTI image/mask pairs")
    p.add_argument("--scan-index-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-spacing", nargs=3, type=float, default=[0.7, 0.7, 3.0], metavar=("X", "Y", "Z"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    COCAResampler(
        scan_index_csv=args.scan_index_csv,
        output_dir=args.output_dir,
        target_spacing=args.target_spacing,
    ).run()


if __name__ == "__main__":
    main()
