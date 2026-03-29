from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ValidationResult:
    subject_id: str
    series_dir: Path
    ok: bool
    num_files: int
    reason: str


def _count_dcm_files(series_dir: Path) -> int:
    if not series_dir.exists() or not series_dir.is_dir():
        return 0
    return sum(1 for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")


def _processed_key(subject_id: str, image: str) -> str:
    subject_id = subject_id.strip()
    if subject_id:
        return subject_id
    image = image.strip()
    if not image:
        return ""
    try:
        return Path(image).name
    except Exception:
        return image


def _read_processed_keys(report_csv: Path) -> set[str]:
    if not report_csv.exists():
        return set()
    try:
        with report_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            keys: set[str] = set()
            for row in reader:
                subject_id = (row.get("subject_id") or "").strip()
                image = (row.get("image") or "").strip()
                key = _processed_key(subject_id, image)
                if key:
                    keys.add(key)
            return keys
    except Exception:
        return set()


def _ensure_csv_with_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _normalize_csv_header(path: Path, desired_fieldnames: list[str]) -> None:
    if not path.exists():
        return
    if path.stat().st_size <= 0:
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        first_line = (f.readline() or "").strip()
        if not first_line:
            return
        existing = [c.strip() for c in first_line.split(",") if c.strip()]
        if existing == desired_fieldnames:
            return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=desired_fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {k: (row.get(k) or "") for k in desired_fieldnames}
            writer.writerow(out_row)



def validate_dicom_series_dir(
    series_dir: Path,
    *,
    mode: Literal["fast", "shallow", "deep"] = "shallow",
) -> tuple[bool, int, str]:
    if mode == "fast":
        n = _count_dcm_files(series_dir)
        if n <= 0:
            return False, 0, "no_dcm_files"
        return True, n, "ok"

    try:
        import SimpleITK as sitk
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("SimpleITK is required for DICOM validation. Install with: pip install SimpleITK") from e

    reader = sitk.ImageSeriesReader()
    file_names = list(reader.GetGDCMSeriesFileNames(str(series_dir)))
    if not file_names:
        return False, 0, "no_dicom_series_files"

    missing = [fn for fn in file_names if not Path(fn).exists()]
    if missing:
        return False, len(file_names), f"missing_files:{len(missing)}"

    if mode != "deep":
        return True, len(file_names), "ok"

    try:
        reader.SetFileNames(file_names)
        _ = reader.Execute()
        return True, len(file_names), "ok"
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\r", " ").replace("\n", " ").strip()
        msg = msg[:240]
        return False, len(file_names), f"read_failed:{msg}"


def validate_metadata_csv(
    metadata_csv: Path,
    raw_dir: Path,
    *,
    out_clean_csv: Path,
    out_report_csv: Path,
    mode: Literal["fast", "shallow", "deep"] = "shallow",
    resume: bool = True,
) -> tuple[Path, Path]:
    metadata_csv = metadata_csv.resolve()
    raw_dir = raw_dir.resolve()
    out_clean_csv = out_clean_csv.resolve()
    out_report_csv = out_report_csv.resolve()
    out_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report_csv.parent.mkdir(parents=True, exist_ok=True)

    with metadata_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seen_input_keys: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        subject_id = (row.get("subject_id") or "").strip()
        image = (row.get("image") or "").strip()
        key = _processed_key(subject_id, image)
        if key and key in seen_input_keys:
            continue
        if key:
            seen_input_keys.add(key)
        unique_rows.append(row)
    rows = unique_rows

    clean_fields = ["subject_id", "image", "label", "kind"]
    report_fields = ["subject_id", "image", "kind", "ok", "num_files", "reason"]
    _ensure_csv_with_header(out_clean_csv, clean_fields)
    _ensure_csv_with_header(out_report_csv, report_fields)
    _normalize_csv_header(out_clean_csv, clean_fields)
    _normalize_csv_header(out_report_csv, report_fields)

    processed_keys: set[str] = set()
    if resume:
        processed_keys = _read_processed_keys(out_report_csv)
        if out_report_csv.exists() and out_report_csv.stat().st_size < 128:
            processed_keys = set()

    it = rows
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        it = tqdm(rows, desc=f"validate:{mode}", unit="subject")
    except Exception:
        pass

    with out_clean_csv.open("a", newline="", encoding="utf-8") as clean_f, out_report_csv.open(
        "a", newline="", encoding="utf-8"
    ) as report_f:
        clean_writer = csv.DictWriter(clean_f, fieldnames=clean_fields)
        report_writer = csv.DictWriter(report_f, fieldnames=report_fields)

        for row in it:
            subject_id = (row.get("subject_id") or "").strip()
            image = (row.get("image") or "").strip()
            label = (row.get("label") or "0").strip()
            kind = (row.get("kind") or "dicom_series").strip()

            key = _processed_key(subject_id, image)
            if key and key in processed_keys:
                continue

            series_dir = Path(image)
            if not series_dir.is_absolute():
                series_dir = raw_dir / series_dir

            if kind != "dicom_series":
                clean_writer.writerow(
                    {
                        "subject_id": subject_id,
                        "image": image,
                        "label": label,
                        "kind": kind,
                    }
                )
                report_writer.writerow(
                    {
                        "subject_id": subject_id,
                        "image": str(series_dir),
                        "kind": kind,
                        "ok": "true",
                        "num_files": "",
                        "reason": "skipped_non_dicom",
                    }
                )
                if key:
                    processed_keys.add(key)
                clean_f.flush()
                report_f.flush()
                continue

            ok, n_files, reason = validate_dicom_series_dir(series_dir, mode=mode)
            if ok:
                clean_writer.writerow(
                    {
                        "subject_id": subject_id,
                        "image": image,
                        "label": label,
                        "kind": kind,
                    }
                )

            report_writer.writerow(
                {
                    "subject_id": subject_id,
                    "image": str(series_dir),
                    "kind": kind,
                    "ok": "true" if ok else "false",
                    "num_files": str(n_files),
                    "reason": reason,
                }
            )
            if key:
                processed_keys.add(key)
            clean_f.flush()
            report_f.flush()

    return out_clean_csv, out_report_csv
