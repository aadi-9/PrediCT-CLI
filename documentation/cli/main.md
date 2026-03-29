# `main()`

**Source:** `src/predict/cli.py`

Entry point for the `predict` CLI. Parses command-line arguments and dispatches to the appropriate subcommand handler.

---

## Signature

```python
def main() -> None
```

---

## Parameters

None. Arguments are read from `sys.argv` via `argparse`.

---

## Return Value

`None`. The function terminates the process with `sys.exit(code)` where `code` is the integer return value of the dispatched subcommand handler (`0` on success, non-zero on error).

---

## Raises

| Exception | Condition |
|---|---|
| `SystemExit(0)` | Subcommand handler returned `0` (success) |
| `SystemExit(1)` | Subcommand handler returned a non-zero exit code |
| `SystemExit(2)` | `argparse` detected invalid arguments or missing required options |

---

## Description

`main()` is the registered `console_scripts` entry point defined in `pyproject.toml`. Its responsibilities are:

1. Call [`build_parser()`](build_parser.md) to obtain the fully configured `ArgumentParser`.
2. Call `parser.parse_args()` to process `sys.argv[1:]`.
3. Read the `func` attribute set on the parsed namespace by the subcommand registration (each subcommand registers itself with `set_defaults(func=_cmd_*)`).
4. Call `args.func(args)` and pass the integer return code to `sys.exit()`.

If no subcommand is provided, `argparse` prints help and exits with code `2`.

### Dispatch table

| Subcommand | Handler |
|---|---|
| `pipeline` | `_cmd_pipeline(args)` |
| `make-metadata` | `_cmd_make_metadata(args)` |
| `validate-metadata` | `_cmd_validate_metadata(args)` |

---

## In the Data Pipeline

`main()` is the outermost shell that initiates every pipeline stage. All data-transformation logic lives in the handlers and the modules they call.

```
CLI invocation (predict ...)
  └─► main()                        ← here
        ├─► build_parser()
        ├─► parse_args()
        └─► args.func(args)
              ├─► _cmd_pipeline        → run_pipeline()
              ├─► _cmd_make_metadata   → generate_metadata_csv()
              └─► _cmd_validate_metadata → validate_metadata_csv()
```

---

## Usage Example

Typically invoked from the shell:

```bash
predict pipeline --project-root . --export-processed
predict make-metadata --raw-dir data/raw --out-csv outputs/metadata_all.csv
predict validate-metadata --metadata-csv outputs/metadata_all.csv \
    --out-clean outputs/metadata_clean.csv \
    --out-report outputs/dicom_validation_report.csv
```

`main()` can also be called programmatically (e.g. in tests), but it will call `sys.exit()`, so wrap with `pytest.raises(SystemExit)` if needed:

```python
import sys
from predict.cli import main

sys.argv = ["predict", "pipeline", "--dry-run", "--project-root", "/tmp/test"]
main()  # exits with sys.exit(0) on success
```

---

## Notes

- `main()` never raises exceptions directly; all error handling is delegated to the subcommand handlers, which return non-zero exit codes on failure.
- The `sys.exit()` call means any code after `main()` in the same process will not run on an error path.

---

## Related

- [`build_parser()`](build_parser.md) — constructs the argument parser used by `main()`
- [`run_pipeline()`](../pipeline/run_pipeline.md) — ultimate target of `predict pipeline`
- [`generate_metadata_csv()`](../metadata/generate_metadata_csv.md) — called via `predict make-metadata`
- [`validate_metadata_csv()`](../validate/validate_metadata_csv.md) — called via `predict validate-metadata`
