# RouteCast team handoff

## 1. What is being handed over

The canonical project is this repository. It contains the final RouteCast
implementation, matched baselines, evaluation/plotting scripts, compact model
checkpoints, frozen experiment protocol, summarized results, and manuscript
sources. No executable script should depend on the former E-drive workspace or
on an absolute `F:\MOEresearch` path.

Large local assets are deliberately outside Git:

- `data/`: public route traces and generated packed caches (~194 GB locally);
- `.venv/`: machine-specific CUDA Python environment;
- `python_packages/`: local dependency mirror;
- `archive/`: superseded first-generation project snapshot;
- `reference/`: local paper library;
- TIFF figure exports and disposable smoke-test caches.

## 2. Recommended handoff package

Share two things separately:

1. a **private GitHub repository** containing the tracked files;
2. the dataset/cache directory through a lab server, NAS, portable disk, or a
   separately documented download-and-build workflow.

Do not commit Hugging Face credentials, `.env` files, virtual environments, or
the 194-GB data directory. Before making the repository public, also confirm
that the manuscript author list and unpublished results may be disclosed.

## 3. Reproduce on a new machine

```powershell
git clone <repository-url> RouteCast
cd RouteCast
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1 -BasePython python
```

Restore `data/` using `data/README.md`. Then verify the checkout:

```powershell
& ".\.venv\Scripts\python.exe" ".\code\verify_reproducibility.py"
```

The audit checks required manifests/checkpoints, parses every executable Python
file, rejects legacy absolute E/F path dependencies, loads the main cache
manifest, and loads the final checkpoint.

## 4. Experiment entry points

| Purpose | Entry point |
|---|---|
| Ordered end-to-end workflow | `workflow/README.md` |
| Main RouteCast training | `code/train_universal_routecast_v3.py` |
| TCN baseline | `code/train_tcn_baseline.py` |
| Paper heatmap baseline | `code/evaluate_paper_heatmap.py` |
| Strong baseline suite | `code/run_strong_baselines.ps1` |
| Calibration | `code/calibrate_routecast_v3.py` |
| Adaptive budget/cache | `code/evaluate_adaptive_cache_v3.py` |
| Main metrics | `code/evaluate_routecast_metrics.py` |
| Paper figures | `code/generate_paper_figures.py` |

Run `python <script> --help` before an expensive experiment. The frozen data
split and metric definitions are in `record/paper/protocol/final_protocol.json`.

## 5. Canonical outputs

- Final RouteCast checkpoint:
  `code/models/routecast_cross_token_limit1000_v3_stable.pt`
- GRU baseline checkpoint:
  `code/models/gru_only_cross_token_limit1000_stable.pt`
- TCN baseline checkpoint:
  `code/models/tcn_cross_token_limit1000.pt`
- Primary packed cache:
  `data/routecast_cache/cross_token_limit1000_h16_packed`
- Main result records: `record/paper/results/`
- Final protocol: `record/paper/protocol/final_protocol.json`
- Manuscript source:
  `record/paper/routecast_v2_consistent_pre_experimental_design.tex`

## 6. Collaboration rules

- Protect the `main` branch and develop through short-lived branches.
- Keep one issue per experiment or paper change.
- Never change the request-level split, target definition, or metric semantics
  without updating the frozen protocol and documenting the change.
- Save machine-readable outputs (`CSV`/`JSON`) together with a short Markdown
  interpretation; do not rely on terminal screenshots as evidence.
- Do not commit generated caches. Commit compact final checkpoints only when
  needed for reproduction.
- Record the random seed, command line, software versions, and model/data limit
  for every reported experiment.

## 7. Current evidence boundary

The project establishes trace-level forecasting and idealized cache-replay
results. It does not yet demonstrate end-to-end latency, throughput, or energy
improvements in a deployed MoE serving system. Preserve this distinction in
all presentations and manuscript revisions.

