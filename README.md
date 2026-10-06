# RouteCast

RouteCast is a route-only predictor and selective-prefetching research
prototype for sparse mixture-of-experts inference. It combines statistical
route priors, cross-token transitions, request-prefill evidence, recurrent
history, and multiscale temporal features, then applies calibration, adaptive
candidate selection, and cache-aware admission.

This repository contains the implementation, evaluation scripts, compact
checkpoints, paper sources, and summarized experimental results. The public
trace dataset and generated caches are intentionally excluded from Git because
the local data directory is approximately 194 GB.

## Repository map

```text
MOEresearch/
|-- workflow/             numbered paper-order reproduction entry points
|-- code/                 stable implementation and development utilities
|   |-- models/           compact final/baseline checkpoints
|   `-- routecast_rc/     reusable RouteCast modules
|-- data/README.md        dataset source and expected local layout
|-- record/               protocols, result tables, manuscript and figures
|-- requirements-*.txt    Python dependency specifications
|-- setup_environment.ps1 environment bootstrap for Windows
`-- HANDOFF.md            team handoff and reproduction guide
```

## Quick start

```powershell
git clone <repository-url> RouteCast
cd RouteCast
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1 -BasePython python
```

Place the datasets and packed caches according to [data/README.md](data/README.md),
then run:

```powershell
& ".\.venv\Scripts\python.exe" ".\code\verify_reproducibility.py"
```

The expected final line is `reproducibility_audit=PASS`.

## Primary artifacts

- Model: `code/models/routecast_cross_token_limit1000_v3_stable.pt`
- Protocol: `record/paper/protocol/final_protocol.json`
- Results: `record/paper/results/`
- Manuscript: `record/paper/routecast_v2_consistent_pre_experimental_design.tex`
- Compiled paper: `RouteCast.pdf`

See [HANDOFF.md](HANDOFF.md) for the complete handoff procedure and ownership
boundaries.

## Reproduce in paper order

Start from [workflow/README.md](workflow/README.md). It maps the complete
architecture and evidence chain to numbered scripts from environment audit,
through prediction and calibration, to cache replay and paper figures. The
implementation ownership map is in [code/CODE_MAP.md](code/CODE_MAP.md).

