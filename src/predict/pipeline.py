from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .augment import build_monai_transforms
from .config import HUWindowConfig, LoaderConfig, PathsConfig, ResampleConfig, SplitConfig
from .dataset import SampleRecord, VolumeDataset, build_dataloader, default_load_volume
from .io import discover_subject_dirs, save_numpy_volume
from .preprocess import apply_hu_window, resample_volume
from .sampling import oversample_minority
from .split import stratified_split


@dataclass(frozen=True)
class PipelineOutputs:
    train_loader: Any | None
    val_loader: Any | None
    test_loader: Any | None
    stats: dict[str, Any]
    processed_manifest_path: Path | None
    split_manifest_path: Path | None


def _resolve_data_path(project_root: Path, raw_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    parts = [p.lower() for p in path.parts]
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "raw":
        path = Path(*path.parts[2:])
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return candidate
    return (raw_dir / path).resolve()



def load_metadata_csv(path: Path, project_root: Path, raw_dir: Path) -> tuple[list[SampleRecord], list[str]]:
    """
    Expected columns:
      - subject_id
      - image (path to DICOM series folder or .npy file)
      - mask (path to ground truth mask, optional)
      - label (int)
      - kind (dicom_series|numpy|nifti) [optional; default dicom_series]

    Next steps:
    - Generate this file after you download COCA.
    - Ensure labels are patient-level and match your evaluation goal.
    """
    records: list[SampleRecord] = []
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"metadata_csv_missing:{path}")
        return records, warnings

    labels_by_subject: dict[str, int] = {}
    seen_subjects: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            subject_id = (row.get("subject_id") or "").strip()
            image = (row.get("image") or "").strip()
            mask = (row.get("mask") or "").strip()
            label_raw = (row.get("label") or "0").strip()
            kind = (row.get("kind") or "dicom_series").strip()

            if kind not in {"dicom_series", "numpy", "nifti", "nii", "nifti_gz"}:
                warnings.append(f"line_{line_no}:unsupported_kind:{kind}")
                continue
            if not subject_id or not image:
                warnings.append(f"line_{line_no}:missing_subject_or_image")
                continue

            try:
                label = int(label_raw)
            except ValueError:
                warnings.append(f"line_{line_no}:invalid_label:{label_raw}")
                continue

            existing_label = labels_by_subject.get(subject_id)
            if existing_label is not None and existing_label != label:
                warnings.append(
                    f"line_{line_no}:conflicting_label_for_subject:{subject_id}:{existing_label}!={label}"
                )
                continue

            if subject_id in seen_subjects:
                warnings.append(f"line_{line_no}:duplicate_subject_skipped:{subject_id}")
                continue

            labels_by_subject[subject_id] = label
            seen_subjects.add(subject_id)
            resolved_image = _resolve_data_path(project_root, raw_dir, image)
            if not resolved_image.exists():
                warnings.append(f"line_{line_no}:image_path_missing:{resolved_image}")
                
            resolved_mask = None
            if mask:
                resolved_mask = _resolve_data_path(project_root, raw_dir, mask)
                if not resolved_mask.exists():
                    warnings.append(f"line_{line_no}:mask_path_missing:{resolved_mask}")

            records.append(
                SampleRecord(
                    subject_id=subject_id,
                    image=resolved_image,
                    label=label,
                    mask=resolved_mask,
                    kind=kind,
                )
            )
    return records, warnings


def build_records_fallback(paths: PathsConfig) -> tuple[list[SampleRecord], list[str]]:
    """
    Fallback discovery if you don't have a metadata CSV yet.

    Assumes: each subdirectory under data/raw is a single subject's DICOM series folder.
    Assigns label=0 for all subjects (placeholder).

    Next steps:
    - Replace this with real metadata/labels once COCA is downloaded.
    """
    records: list[SampleRecord] = []
    warnings: list[str] = []
    for subject_dir in discover_subject_dirs(paths.raw_dir):
        records.append(
            SampleRecord(
                subject_id=subject_dir.name,
                image=subject_dir,
                label=0,
                kind="dicom_series",
            )
        )
    if records:
        warnings.append("metadata_fallback_used:assigned_label_0_for_all_subjects")
    return records, warnings


def _count_by_label(labels: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for y in labels:
        out[int(y)] = out.get(int(y), 0) + 1
    return out


def _rows_for_split(records: list[SampleRecord], split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "subject_id": r.subject_id,
                "label": int(r.label),
                "kind": r.kind,
                "image": str(r.image),
                "split": split_name,
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_processed_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "label", "kind", "processed_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    project_root: Path,
    metadata_csv: Path | None = None,
    stats_path: Path | None = None,
    split_manifest_path: Path | None = None,
    processed_manifest_path: Path | None = None,
    processed_dir: Path | None = None,
    raw_dir: Path | None = None,
    resample_cfg: ResampleConfig | None = None,
    hu_cfg: HUWindowConfig | None = None,
    split_cfg: SplitConfig | None = None,
    loader_cfg: LoaderConfig | None = None,
    enable_augmentation: bool = True,
    oversample_train: bool = True,
    export_processed: bool = False,
    dry_run: bool = False,
) -> PipelineOutputs:
    """
    End-to-end pipeline that wires preprocessing -> split -> sampling -> DataLoader.

    Next steps:
    - Project 1 (segmentation): add mask paths + paired transforms + UNet training loop.
    - Project 2 (radiomics): export windowed/resampled volumes in a radiomics-friendly format.
    - Project 3 (template extraction): add task-specific feature preprocessing here.
    """
    paths = PathsConfig.from_project_root(project_root, raw_dir=raw_dir)
    resample_cfg = resample_cfg or ResampleConfig()
    hu_cfg = hu_cfg or HUWindowConfig()
    split_cfg = split_cfg or SplitConfig()
    loader_cfg = loader_cfg or LoaderConfig()

    if metadata_csv is None:
        metadata_csv = paths.data_dir / "metadata.csv"
    if stats_path is None:
        stats_path = paths.outputs_dir / "dataset_stats.json"
    if split_manifest_path is None:
        split_manifest_path = paths.outputs_dir / "splits.json"
    if processed_manifest_path is None:
        processed_manifest_path = paths.outputs_dir / "processed_manifest.csv"
    if processed_dir is None:
        processed_dir = paths.processed_dir

    warnings: list[str] = []
    if not paths.raw_dir.exists():
        warnings.append(f"raw_dir_missing:{paths.raw_dir}")

    records, metadata_warnings = load_metadata_csv(metadata_csv, project_root=project_root, raw_dir=paths.raw_dir)
    warnings.extend(metadata_warnings)
    if not records:
        records, fallback_warnings = build_records_fallback(paths)
        warnings.extend(fallback_warnings)

    labels = [r.label for r in records]

    split = stratified_split(records, labels, split_cfg)
    train_recs = [records[i] for i in split.train]
    val_recs = [records[i] for i in split.val]
    test_recs = [records[i] for i in split.test]

    train_labels = [r.label for r in train_recs]
    val_labels = [r.label for r in val_recs]
    test_labels = [r.label for r in test_recs]

    before_counts = {
        "train": _count_by_label(train_labels),
        "val": _count_by_label(val_labels),
        "test": _count_by_label(test_labels),
    }

    if oversample_train and len(train_recs) > 0:
        train_recs, train_labels = oversample_minority(train_recs, train_labels, split_cfg.random_state)

    after_counts = {
        "train": _count_by_label(train_labels),
        "val": _count_by_label(val_labels),
        "test": _count_by_label(test_labels),
    }

    split_rows = (
        _rows_for_split(train_recs, "train")
        + _rows_for_split(val_recs, "val")
        + _rows_for_split(test_recs, "test")
    )

    stats: dict[str, Any] = {
        "num_subjects": len(records),
        "split_sizes": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
        "class_counts_before_sampling": before_counts,
        "class_counts_after_sampling": after_counts,
        "resample": {
            "mode": resample_cfg.mode,
            "target_shape": resample_cfg.target_shape,
            "target_spacing": resample_cfg.target_spacing,
        },
        "hu_window": {"lower": hu_cfg.lower, "upper": hu_cfg.upper},
        "paths": {
            "metadata_csv": str(metadata_csv),
            "stats": str(stats_path),
            "split_manifest": str(split_manifest_path),
            "processed_manifest": str(processed_manifest_path),
            "processed_dir": str(processed_dir),
            "raw_dir": str(paths.raw_dir),
        },
        "warnings": warnings,
    }

    _write_json(split_manifest_path, split_rows)
    _write_json(stats_path, stats)

    if dry_run:
        return PipelineOutputs(
            train_loader=None,
            val_loader=None,
            test_loader=None,
            stats=stats,
            processed_manifest_path=None,
            split_manifest_path=split_manifest_path,
        )

    def preprocess_fn(vol, is_label=False):
        vol = resample_volume(vol, cfg=resample_cfg, is_label=is_label)
        if not is_label:
            vol = apply_hu_window(vol, hu_cfg)
        return vol

    final_train_recs = train_recs
    final_val_recs = val_recs
    final_test_recs = test_recs
    final_processed_manifest_path: Path | None = None
    export_failures: list[str] = []

    if export_processed:
        processed_rows: list[dict[str, Any]] = []

        def export_split_records(split_name: str, split_records: list[SampleRecord]) -> list[SampleRecord]:
            out_records: list[SampleRecord] = []
            it = split_records
            try:
                from tqdm import tqdm  # type: ignore[import-not-found]

                it = tqdm(split_records, desc=f"export:{split_name}", unit="subject")
            except Exception:
                pass

            for rec in it:
                try:
                    vol = default_load_volume(rec)
                    vol = preprocess_fn(vol)
                    out_path = processed_dir / f"{rec.subject_id}.npy"
                    save_numpy_volume(vol, out_path)
                    out_records.append(
                        SampleRecord(
                            subject_id=rec.subject_id,
                            image=out_path,
                            label=rec.label,
                            kind="numpy",
                        )
                    )
                    processed_rows.append(
                        {
                            "subject_id": rec.subject_id,
                            "label": int(rec.label),
                            "kind": "numpy",
                            "processed_path": str(out_path),
                            "split": split_name,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    export_failures.append(f"{split_name}:{rec.subject_id}:{e}")
            return out_records

        final_train_recs = export_split_records("train", train_recs)
        final_val_recs = export_split_records("val", val_recs)
        final_test_recs = export_split_records("test", test_recs)
        _write_processed_manifest(processed_manifest_path, processed_rows)
        final_processed_manifest_path = processed_manifest_path

    if export_failures:
        stats["warnings"] = list(stats.get("warnings", [])) + export_failures
        _write_json(stats_path, stats)

    train_loader: Any | None = None
    val_loader: Any | None = None
    test_loader: Any | None = None
    try:
        train_loader_cfg = loader_cfg
        eval_loader_cfg = replace(loader_cfg, shuffle=False)

        transform = build_monai_transforms(enable=enable_augmentation)
        dataset_preprocess_fn = None if export_processed else preprocess_fn
        train_ds = VolumeDataset(final_train_recs, transform=transform, preprocess_fn=dataset_preprocess_fn)
        val_ds = VolumeDataset(final_val_recs, transform=None, preprocess_fn=dataset_preprocess_fn)
        test_ds = VolumeDataset(final_test_recs, transform=None, preprocess_fn=dataset_preprocess_fn)

        if len(train_ds) > 0:
            train_loader = build_dataloader(train_ds, train_loader_cfg)
        if len(val_ds) > 0:
            val_loader = build_dataloader(val_ds, eval_loader_cfg)
        if len(test_ds) > 0:
            test_loader = build_dataloader(test_ds, eval_loader_cfg)
    except Exception as e:  # noqa: BLE001
        stats["warnings"] = list(stats.get("warnings", [])) + [f"loader_build_skipped:{e}"]
        _write_json(stats_path, stats)

    return PipelineOutputs(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        stats=stats,
        processed_manifest_path=final_processed_manifest_path,
        split_manifest_path=split_manifest_path,
    )
