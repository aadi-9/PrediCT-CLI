from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HUWindowConfig, ResampleConfig, resolve_raw_dir
from .metadata import generate_metadata_csv
from .pipeline import run_pipeline
from .report import write_justification_report
from .validate import validate_metadata_csv


def _cmd_pipeline(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    stats_path = Path(args.stats_path) if args.stats_path else None
    if stats_path is not None and not stats_path.is_absolute():
        stats_path = project_root / stats_path

    split_manifest_path = Path(args.split_manifest) if args.split_manifest else None
    if split_manifest_path is not None and not split_manifest_path.is_absolute():
        split_manifest_path = project_root / split_manifest_path

    processed_manifest_path = Path(args.processed_manifest) if args.processed_manifest else None
    if processed_manifest_path is not None and not processed_manifest_path.is_absolute():
        processed_manifest_path = project_root / processed_manifest_path

    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    if processed_dir is not None and not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    if raw_dir is not None and not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    resample_cfg = ResampleConfig(
        mode="spacing",
        target_spacing=(
            float(args.resample_spacing[0]),
            float(args.resample_spacing[1]),
            float(args.resample_spacing[2]),
        ),
    )
    hu_cfg = HUWindowConfig(lower=float(args.hu_bounds[0]), upper=float(args.hu_bounds[1]))

    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    if metadata_csv is not None and not metadata_csv.is_absolute():
        metadata_csv = project_root / metadata_csv

    out = run_pipeline(
        project_root=project_root,
        metadata_csv=metadata_csv,
        stats_path=stats_path,
        split_manifest_path=split_manifest_path,
        processed_manifest_path=processed_manifest_path,
        processed_dir=processed_dir,
        raw_dir=raw_dir,
        resample_cfg=resample_cfg,
        hu_cfg=hu_cfg,
        dry_run=bool(args.dry_run),
        enable_augmentation=bool(args.augment),
        oversample_train=bool(args.oversample_train),
        export_processed=bool(args.export_processed),
    )

    stats_json = json.dumps(out.stats, indent=2, sort_keys=True)
    print(stats_json)

    if args.justification_path:
        justification_path = Path(args.justification_path)
        if not justification_path.is_absolute():
            justification_path = project_root / justification_path
        write_justification_report(stats=out.stats, out_path=justification_path)

    return 0


def _cmd_make_metadata(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    raw_dir_value = Path(args.raw_dir) if args.raw_dir else None
    raw_dir = resolve_raw_dir(project_root, raw_dir=raw_dir_value)

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = project_root / out_csv

    generate_metadata_csv(
        raw_dir=raw_dir,
        out_csv=out_csv,
        default_label=int(args.default_label),
        kind=str(args.kind),
    )
    print(str(out_csv))
    return 0


def _cmd_validate_metadata(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    raw_dir_value = Path(args.raw_dir) if args.raw_dir else None
    raw_dir = resolve_raw_dir(project_root, raw_dir=raw_dir_value)

    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else (project_root / "data" / "metadata.csv")
    if not metadata_csv.is_absolute():
        metadata_csv = project_root / metadata_csv

    out_clean_csv = Path(args.out_clean_csv)
    if not out_clean_csv.is_absolute():
        out_clean_csv = project_root / out_clean_csv

    out_report_csv = Path(args.out_report_csv)
    if not out_report_csv.is_absolute():
        out_report_csv = project_root / out_report_csv

    clean_path, report_path = validate_metadata_csv(
        metadata_csv=metadata_csv,
        raw_dir=raw_dir,
        out_clean_csv=out_clean_csv,
        out_report_csv=out_report_csv,
        mode=str(args.mode),
    )

    print(str(clean_path))
    print(str(report_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="predict", description="PrediCT scaffold CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pipe = sub.add_parser("pipeline", help="Run preprocessing + split + loader wiring")
    pipe.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Repository root (defaults to current working directory)",
    )
    pipe.add_argument(
        "--metadata-csv",
        default="",
        help="Optional path to metadata CSV (default: data/metadata.csv)",
    )
    pipe.add_argument(
        "--stats-path",
        default=str(Path("outputs") / "dataset_stats.json"),
        help="Where to write dataset statistics JSON",
    )
    pipe.add_argument(
        "--split-manifest",
        default=str(Path("outputs") / "splits.json"),
        help="Where to write the split manifest JSON",
    )
    pipe.add_argument(
        "--processed-manifest",
        default=str(Path("outputs") / "processed_manifest.csv"),
        help="Where to write the processed manifest CSV",
    )
    pipe.add_argument(
        "--processed-dir",
        default=str(Path("data") / "processed"),
        help="Directory to store exported processed numpy volumes",
    )
    pipe.add_argument(
        "--raw-dir",
        default="",
        help="Override raw dataset dir (default: PREDICT_RAW_DIR env var or config default)",
    )
    pipe.add_argument(
        "--export-processed",
        action="store_true",
        help="Export preprocessed volumes to data/processed and switch loaders to .npy files",
    )
    pipe.add_argument(
        "--resample-spacing",
        nargs=3,
        metavar=("Z", "Y", "X"),
        type=float,
        default=(1.0, 1.0, 1.0),
        help="Target spacing for resampling in (Z, Y, X)",
    )
    pipe.add_argument(
        "--hu-bounds",
        nargs=2,
        metavar=("LOWER", "UPPER"),
        type=float,
        default=(-200.0, 400.0),
        help="HU windowing bounds",
    )
    pipe.add_argument("--dry-run", action="store_true", help="Only compute stats and exit")
    pipe.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation")
    pipe.set_defaults(augment=True)
    pipe.add_argument(
        "--no-oversample-train",
        dest="oversample_train",
        action="store_false",
        help="Disable oversampling for training split",
    )
    pipe.set_defaults(oversample_train=True)
    pipe.add_argument(
        "--justification-path",
        default=str(Path("outputs") / "justification.txt"),
        help="Where to write the auto-generated short preprocessing justification",
    )
    pipe.add_argument(
        "--write-stats",
        dest="stats_path",
        help="Deprecated alias for --stats-path",
    )
    pipe.set_defaults(func=_cmd_pipeline)

    meta = sub.add_parser("make-metadata", help="Generate a metadata CSV by scanning raw patient folders")
    meta.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Repository root (defaults to current working directory)",
    )
    meta.add_argument(
        "--raw-dir",
        default="",
        help="Raw patient folder root (e.g. ...\\Gated_release_final\\patient)",
    )
    meta.add_argument(
        "--out-csv",
        default=str(Path("data") / "metadata.csv"),
        help="Output metadata CSV path",
    )
    meta.add_argument(
        "--default-label",
        type=int,
        default=0,
        help="Default label to assign to every subject",
    )
    meta.add_argument(
        "--kind",
        default="dicom_series",
        help="dicom_series|numpy|nifti|nii|nifti_gz",
    )
    meta.set_defaults(func=_cmd_make_metadata)

    val = sub.add_parser("validate-metadata", help="Validate DICOM readability and write a cleaned metadata CSV")
    val.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Repository root (defaults to current working directory)",
    )
    val.add_argument(
        "--raw-dir",
        default="",
        help="Raw patient folder root (default: PREDICT_RAW_DIR env var or config default)",
    )
    val.add_argument(
        "--metadata-csv",
        default=str(Path("data") / "metadata_all.csv"),
        help="Input metadata CSV to validate",
    )
    val.add_argument(
        "--out-clean-csv",
        default=str(Path("data") / "metadata_clean.csv"),
        help="Output CSV that excludes invalid series",
    )
    val.add_argument(
        "--out-report-csv",
        default=str(Path("outputs") / "dicom_validation_report.csv"),
        help="Output CSV with per-subject validation results",
    )
    val.add_argument(
        "--mode",
        default="shallow",
        help="fast|shallow|deep (deep reads each series; best for Drive placeholders)",
    )
    val.set_defaults(func=_cmd_validate_metadata)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
