from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    parts: list[str] = []
    for key in sorted(counts, key=lambda x: str(x)):
        parts.append(f"class {key}: {counts[key]}")
    return ", ".join(parts)


def build_justification_text(stats: dict[str, Any]) -> str:
    resample = stats.get("resample", {})
    hu_window = stats.get("hu_window", {})
    split_sizes = stats.get("split_sizes", {})
    before = stats.get("class_counts_before_sampling", {})
    after = stats.get("class_counts_after_sampling", {})
    warnings = stats.get("warnings", [])

    lines: list[str] = []
    lines.append("COCA preprocessing justification")
    lines.append("")
    lines.append(
        "We use deterministic preprocessing with a fixed order: resampling first, then HU windowing."
    )
    lines.append(
        "Resampling is configured with "
        f"mode={resample.get('mode')}, target_spacing={resample.get('target_spacing')}, "
        f"target_shape={resample.get('target_shape')}."
    )
    lines.append(
        "HU window bounds are fixed across all splits at "
        f"lower={hu_window.get('lower')} and upper={hu_window.get('upper')}."
    )
    lines.append(
        "This keeps intensity normalization and geometry handling consistent across train, validation, and test."
    )
    lines.append("")
    lines.append(
        "Split sizes: "
        f"train={split_sizes.get('train', 0)}, val={split_sizes.get('val', 0)}, test={split_sizes.get('test', 0)}."
    )
    lines.append(
        "Class counts before sampling: "
        f"train[{_fmt_counts(before.get('train', {}))}], "
        f"val[{_fmt_counts(before.get('val', {}))}], "
        f"test[{_fmt_counts(before.get('test', {}))}]."
    )
    lines.append(
        "Class counts after sampling: "
        f"train[{_fmt_counts(after.get('train', {}))}], "
        f"val[{_fmt_counts(after.get('val', {}))}], "
        f"test[{_fmt_counts(after.get('test', {}))}]."
    )
    lines.append("")
    lines.append(
        "Augmentation should be applied to training data only; validation and test are kept unaugmented."
    )
    lines.append(
        "Stratified splitting is performed at subject level using the provided patient label."
    )
    if warnings:
        lines.append("")
        lines.append("Warnings and exclusions:")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def write_justification_report(
    stats: dict[str, Any] | None = None,
    stats_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    if stats is None:
        if stats_path is None:
            raise ValueError("Either stats or stats_path must be provided.")
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        payload = stats

    if out_path is None:
        out_path = Path("outputs") / "justification.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_justification_text(payload), encoding="utf-8")
    return out_path
