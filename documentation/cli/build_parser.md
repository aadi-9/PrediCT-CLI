# `build_parser()`

**Source:** `src/predict/cli.py`

Builds and returns the top-level `predict` `ArgumentParser`, complete with all three subcommands and their arguments.

---

## Signature

```python
def build_parser() -> argparse.ArgumentParser
```

---

## Parameters

None.

---

## Return Value

| Type | Description |
|---|---|
| `argparse.ArgumentParser` | Fully configured parser with `pipeline`, `make-metadata`, and `validate-metadata` subcommands |

---

## Description

`build_parser()` constructs the entire CLI argument tree for the `predict` command. It registers three subcommands via an `add_subparsers()` call, each with its own set of flags:

### Subcommand: `pipeline`

Runs the full end-to-end preprocessing pipeline.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--project-root` | `str` | `cwd` | Root directory of the project |
| `--metadata-csv` | `str` | `""` | Path to metadata CSV; empty string triggers auto-discovery |
| `--stats-path` | `str` | `"outputs/dataset_stats.json"` | Output path for statistics JSON |
| `--split-manifest` | `str` | `"outputs/splits.json"` | Output path for train/val/test split indices |
| `--processed-manifest` | `str` | `"outputs/processed_manifest.csv"` | Output path for processed volume manifest |
| `--processed-dir` | `str` | `"data/processed"` | Directory to save exported `.npy` files |
| `--raw-dir` | `str` | `""` | Override for raw DICOM directory |
| `--resample-spacing` | `float float float` | `1.0 1.0 1.0` | Target voxel spacing in mm (Z Y X) |
| `--hu-bounds` | `float float` | `-200.0 400.0` | HU clip window (lower upper) |
| `--justification-path` | `str` | `"outputs/justification.txt"` | Output path for the auto-generated justification report |
| `--dry-run` | flag | off | Parse and validate only; skip DataLoader construction |
| `--no-augment` | flag | off | Disable training augmentation |
| `--no-oversample-train` | flag | off | Disable minority-class oversampling on training set |
| `--export-processed` | flag | off | Save preprocessed volumes as `.npy` files |

### Subcommand: `make-metadata`

Scans a raw directory and writes a metadata CSV.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--project-root` | `str` | `cwd` | Root directory of the project |
| `--raw-dir` | `str` | `""` | Path to directory containing subject subdirectories |
| `--out-csv` | `str` | `"data/metadata.csv"` | Output path for the generated CSV |
| `--default-label` | `int` | `0` | Default label to assign to all subjects |
| `--kind` | `str` | `"dicom_series"` | Data kind: `dicom_series`, `numpy`, `nifti`, `nii`, `nifti_gz` |

### Subcommand: `validate-metadata`

Validates DICOM readability for every series listed in a metadata CSV.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--project-root` | `str` | `cwd` | Root directory of the project |
| `--raw-dir` | `str` | `""` | Base raw directory for resolving relative paths |
| `--metadata-csv` | `str` | `"data/metadata_all.csv"` | Path to the input metadata CSV to validate |
| `--out-clean-csv` | `str` | `"data/metadata_clean.csv"` | Output path for the cleaned (valid-only) CSV |
| `--out-report-csv` | `str` | `"outputs/dicom_validation_report.csv"` | Output path for the full validation report CSV |
| `--mode` | `str` | `"shallow"` | Validation depth: `fast`, `shallow`, or `deep` |

---

## In the Data Pipeline

`build_parser()` is called once, at startup, by [`main()`](main.md). It has no effect on data itself — it only defines the CLI surface.

```
main()
  └─► build_parser()   ← here
        └─► parser.parse_args()
              └─► dispatch to _cmd_pipeline / _cmd_make_metadata / _cmd_validate_metadata
```

---

## Usage Example

```python
from predict.cli import build_parser

parser = build_parser()
args = parser.parse_args([
    "pipeline",
    "--project-root", "/data/cardiac",
    "--metadata-csv", "outputs/metadata_clean.csv",
    "--export-processed",
])
print(args.metadata_csv)  # "outputs/metadata_clean.csv"
```

To print help for a subcommand:

```bash
predict pipeline --help
predict make-metadata --help
predict validate-metadata --help
```

---

## Notes

- If no subcommand is given, the parser prints help and exits.
- Relative paths for `--project-root`, `--metadata-csv`, etc. are resolved against the current working directory inside the respective `_cmd_*` handler, not inside `build_parser()` itself.
- The `--resample-shape` and `--resample-spacing` flags both accept three space-separated numbers (Z Y X order).

---

## Related

- [`main()`](main.md) — calls `build_parser()` and dispatches subcommands
- [`run_pipeline()`](../pipeline/run_pipeline.md) — the function `_cmd_pipeline` eventually calls
- [`generate_metadata_csv()`](../metadata/generate_metadata_csv.md) — called by `_cmd_make_metadata`
- [`validate_metadata_csv()`](../validate/validate_metadata_csv.md) — called by `_cmd_validate_metadata`
