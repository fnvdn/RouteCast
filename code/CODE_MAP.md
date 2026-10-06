# RouteCast implementation map

Use `workflow/00_verify.ps1` through `workflow/07_generate_paper_artifacts.ps1`
for normal reproduction. This file maps the manuscript concepts to their
implementation and distinguishes canonical code from development utilities.

## Canonical implementation

### Data and protocol

- `routecast_rc/registry.py`: model-family registry, expert-pool and native-K metadata.
- `routecast_rc/trace_io.py`: trace discovery, request-level split and tensor loading.
- `build_cross_token_cache_v3.py`: history-16 feature cache used by final experiments.
- `audit_cache_numerics.py`: packed-cache finite-value and shape audit.

### RouteCast forecaster

- `routecast_rc/universal_model_v3.py`: six evidence branches, descriptors,
  conditional gate and residual correction.
- `train_universal_routecast_v3.py`: balanced losses, optional Group-DRO,
  ablations and shared/per-model training.
- `test_routecast_stability.py`: numerical forward/backward stability smoke test.

### Baselines

- `evaluate_paper_heatmap.py`: paper-derived cross-token heatmap baseline.
- `evaluate_request_aware_heatmap.py`: request-conditioned heatmap baseline.
- `routecast_rc/tcn_baseline.py`: causal dilated TCN model.
- `train_tcn_baseline.py`: matched TCN training and evaluation.
- `evaluate_moe_infinity_eam.py`: trace-level system-policy comparison utility.

### Calibration and adaptive selection

- `calibrate_routecast_v3.py`: per-model marginal temperature scaling.
- `sweep_dynamic_topk_v3.py`: calibrated score-mass to dynamic-width sweep.
- `evaluate_budget_curve.py`: fixed candidate-width coverage/precision frontier.
- `routecast_rc/conformal.py`: confidence/set-selection helpers.

### Cache-aware admission

- `routecast_rc/cache_sim.py`: cache state and replay primitives.
- `evaluate_adaptive_cache_v3.py`: LRU, fixed, adaptive-mass and selective
  cache-aware replay policies.
- `compute_deadline_transfer_budget.py`: optional deadline-budget sensitivity.

### Metrics, statistics and reporting

- `evaluate_routecast_metrics.py`: Recall, Precision, MRR and NDCG evaluation.
- `bootstrap_paired_requests.py`: paired request-level bootstrap.
- `run_cross_model_ablation.py`: shared/per-model and no-Group-DRO experiment.
- `summarize_limit1000_main.py`: frozen primary-result table.
- `generate_paper_figures.py`: main and extended paper figures.
- `generate_split_main_figures.py`: column-width figure variants.
- `inspect_paper_pdf.py`: PDF layout/text diagnostic.

## Development or compatibility code

The following files are retained for provenance but are not the final paper
pipeline:

- `legacy/`: first-generation Qwen-only gated-fusion implementation.
- `build_universal_cache.py`, `train_universal_routecast.py`: V1 shared model.
- `build_universal_cache_v3.py`: earlier non-cross-token V3 cache builder.
- `train_multik_routecast.py`: exploratory multi-K training branch.
- framework/compact plotting scripts: figure-layout development utilities.

Do not use a development script to regenerate a primary table unless the
protocol and manuscript are explicitly revised.

