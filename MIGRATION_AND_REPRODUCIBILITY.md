# RouteCast migration and reproducibility record

Date: 2026-08-26

## Outcome

`F:\MOEresearch` is the canonical, self-contained RouteCast project. Runtime
code no longer depends on `E:\Chip Research\Predictor` or on a fixed F-drive
absolute path. Project-relative paths are derived from each script location.

## Migrated from the initial E-drive project

- Source papers: `reference/`
- Initial experiment records, scripts and checkpoints:
  `archive/initial_e_design/`
- Three legacy Qwen records used by extended manuscript figures:
  `record/legacy_qwen/`
- Working CUDA environment: `.venv/`

The initial raw Qwen trace was not copied a second time because every one of
the 47,633 E-drive files exists under `data/moe_trace` with the same relative
path and byte size. The F-drive copy contains two additional Hugging Face cache
metadata files. The 38.62-GB E-drive `fusion_cache` belongs to the superseded
first-generation predictor and is not referenced by any final executable.

## Validation performed before E-drive cleanup

- Archive comparisons: 0 missing files and 0 byte-size differences for the
  initial Reference, experiment-record, script, model, paper and Research trees.
- Python syntax: all 46 project Python files parsed successfully.
- Absolute dependency scan: no E-drive or fixed `F:\MOEresearch` path remains
  in executable `.py`, `.ps1`, `.bat`, or `.cmd` files.
- Environment: PyTorch 2.13.0+cu126, CUDA 12.6, NumPy 2.5.2; CUDA available.
- Packed cache: 3,000 manifest items loaded.
- Final RouteCast checkpoint loaded successfully on CPU.
- CLI smoke checks passed for training, TCN, calibration, dynamic budget,
  adaptive cache, paper heatmap and cache-building entry points.
- Three-model forward/backward numerical stability smoke test passed.

Smoke-test output: `record/migration_smoke_stability.json`.

## Key SHA-256 digests

| Artifact | SHA-256 |
|---|---|
| Final RouteCast checkpoint | `BF2CE60AECFB59327575E2FEA0C2A7D35E67C752A9A449AF00D38C2CB433D50F` |
| GRU baseline checkpoint | `7D879EA623654D9A8BD9B3144DD9D3F01B5FC47250592D0AD3074B9F7528B22A` |
| TCN baseline checkpoint | `75921B655EFFF38BB8425785B137B759682AD193AFCDEB7DB90DD82F63D0BBF5` |
| Frozen protocol | `E72D5D9809A43244A123920A0D97BC9B3BB7970331702253722E5D1363553180` |
| Final manuscript source | `4D01762EAF95BE8B79325FD50FBCD01ECBAC2F023FC385922D894E7E2301E211` |
| Primary cache manifest | `787A68AEB794778D4A534749BB2501E590A6ADC9C86F6DD8DBED3FB4E8EE919A` |

## Re-run audit

```powershell
cd "F:\MOEresearch"
& ".\.venv\Scripts\python.exe" ".\code\verify_reproducibility.py"
```

Expected final line:

```text
reproducibility_audit=PASS
```

## E-drive cleanup scope

After the above checks passed, the contents of
`E:\Chip Research\Predictor` were authorized for deletion. The cleanup is not
recoverable from the E drive itself. Required research artifacts remain under
the F-drive paths documented above.

