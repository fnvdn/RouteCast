# RouteCast reproducible project

For the paper-order execution path, use `../workflow/README.md`. For the
manuscript-to-code ownership map, use `CODE_MAP.md`.

The project is self-contained under the directory that contains this `code`
folder. Executable scripts derive `data`, `record`, `models`, and `.venv` paths
from their own location; no E-drive or fixed F-drive dependency remains.

```text
MOEresearch/
|-- .venv/                     # verified local CUDA environment
|-- code/                      # RouteCast, baselines, evaluation and figures
|   |-- models/                # final and baseline checkpoints
|   |-- routecast_rc/          # reusable model/data/cache modules
|   `-- legacy/                # minimal first-generation compatibility scripts
|-- data/                      # public traces and packed RouteCast caches
|-- record/                    # protocol, results, manuscript and figures
|   `-- legacy_qwen/           # legacy records used by extended figures
|-- reference/                 # source papers
|-- archive/initial_e_design/  # compact archive of the first E-drive design
|-- requirements-core.txt
|-- requirements-figures.txt
|-- requirements-current-lock.txt
`-- setup_environment.ps1
```

## Verify the existing environment

```powershell
cd "F:\MOEresearch"
& ".\.venv\Scripts\python.exe" ".\code\verify_reproducibility.py"
```

## Recreate the environment on another machine

```powershell
powershell -ExecutionPolicy Bypass -File ".\setup_environment.ps1" `
  -BasePython "python"
```

The exact local environment used for the paper has PyTorch
`2.13.0+cu126`, CUDA `12.6`, and NumPy `2.5.2`. The portable requirements use
a compatible PyTorch 2.x lower bound because the exact development build may
not be available from every public package index.

## Main artifacts

- Final RouteCast checkpoint:
  `code/models/routecast_cross_token_limit1000_v3_stable.pt`
- Primary packed cache:
  `data/routecast_cache/cross_token_limit1000_h16_packed`
- Frozen protocol:
  `record/paper/protocol/final_protocol.json`
- Final manuscript source:
  `record/paper/routecast_v2_consistent_pre_experimental_design.tex`

