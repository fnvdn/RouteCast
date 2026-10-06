# RouteCast ordered workflow

This directory is the canonical entry point for reproducing the paper. The
numbered stages follow the manuscript's argument and keep the implementation
scripts under `code/` stable.

## Pipeline at a glance

| Stage | Paper role | Entry point | Main implementation |
|---:|---|---|---|
| 00 | Environment and artifact audit | `00_verify.ps1` | `code/verify_reproducibility.py` |
| 01 | Request-disjoint trace preprocessing | `01_build_cache.ps1` | `code/build_cross_token_cache_v3.py` |
| 02 | Multi-source RouteCast training | `02_train_routecast.ps1` | `code/train_universal_routecast_v3.py` |
| 03 | Matched forecasting baselines | `03_train_baselines.ps1` | heatmap, request-aware heatmap and TCN scripts |
| 04 | Marginal calibration | `04_calibrate.ps1` | `code/calibrate_routecast_v3.py` |
| 05 | Fixed/adaptive candidate evaluation | `05_evaluate_prediction.ps1` | metrics and mass-sweep scripts |
| 06 | Selective cache admission | `06_replay_cache.ps1` | `code/evaluate_adaptive_cache_v3.py` |
| 07 | Tables, statistics and figures | `07_generate_paper_artifacts.ps1` | summaries, bootstrap and plotting scripts |

## Two reproduction modes

### Fast verification

Use the committed checkpoints and summarized results:

```powershell
cd <RouteCast repository>
powershell -ExecutionPolicy Bypass -File .\workflow\00_verify.ps1
powershell -ExecutionPolicy Bypass -File .\workflow\04_calibrate.ps1
powershell -ExecutionPolicy Bypass -File .\workflow\05_evaluate_prediction.ps1
powershell -ExecutionPolicy Bypass -File .\workflow\06_replay_cache.ps1
powershell -ExecutionPolicy Bypass -File .\workflow\07_generate_paper_artifacts.ps1
```

### Full reproduction

After restoring the public traces described in `data/README.md`, run stages
00--07 in order. Stages 01--03 are the expensive steps. Outputs produced by
the wrappers are written under `record/reproduction/` or use a
`*_reproduced` filename so the frozen paper artifacts are not overwritten.

## Evidence chain

1. **Trace protocol.** Requests are assigned wholly to train, validation, or
   test. Stage 01 materializes the shared history-16 representation.
2. **Forecaster.** Stage 02 trains the six-branch conditional mixture described
   in the RouteCast Design section.
3. **Comparators.** Stage 03 evaluates statistical heatmaps and trains the
   matched TCN baseline under the same split.
4. **Decision interface.** Stage 04 learns per-model temperatures; Stage 05
   reports ranking quality and the score-mass/average-width frontier.
5. **System-facing analysis.** Stage 06 performs idealized cache replay. These
   results must not be described as measured end-to-end latency.
6. **Paper evidence.** Stage 07 rebuilds summaries and submission figures from
   machine-readable results.

For the implementation-level ownership of every script, see
[`../code/CODE_MAP.md`](../code/CODE_MAP.md).

