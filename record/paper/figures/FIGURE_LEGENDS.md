# RouteCast figure legends

## Main figures

**Figure 1 | RouteCast couples calibrated route forecasting to selective expert-prefetch decisions.** Recent cross-token routes and request context feed three complementary predictors: route history, expert-transition statistics and popularity priors. A context-dependent gate combines their expert scores. Per-model temperature scaling converts the fused scores into calibrated probabilities, after which cumulative probability mass, confidence rejection and cache-aware admission determine which experts are actually prefetched. This schematic describes the algorithmic workflow and contains no measured performance values.

**Figure 2 | RouteCast improves native-budget expert forecasting across heterogeneous MoE models.** **a–c,** Native-budget recall on the request-disjoint limit=1000 test split for Qwen3 (K=8), DeepSeek-R1 (K=8) and Llama4 Maverick (K=1). All methods use the same request split and target definition. **d,** Recall improvement over GRU-only with 95% confidence intervals from 10,000 paired bootstrap resamples at the request level (seed 2026). Intervals exclude zero for all three models.

**Figure 3 | Enlarging the expert budget trades higher coverage for more wasted prefetches.** Recall and wasted-prefetch ratio are plotted against the number of prefetched experts for Qwen3, DeepSeek-R1 and Llama4 Maverick. Vertical dotted lines mark each model's native routing width. These limit=100 development curves motivate budget-aware rather than recall-only evaluation.

**Figure 4 | RouteCast components have model-dependent effects.** Heat-map entries report the change in native-budget recall after removing cross-token history, two-hop transition statistics or prefill context from the full model on the limit=100 development split. Negative values indicate degradation after removal. Cross-token history contributed consistently, whereas two-hop and prefill features were not uniformly beneficial, motivating context-dependent fusion.

**Figure 5 | Calibration makes confidence actionable and probability mass exposes the budget-quality frontier.** **a–c,** Reliability diagrams before and after per-model temperature scaling on the limit=100 test split; dotted diagonals denote ideal calibration. **d–f,** Validation recall as a function of the resulting average expert budget as the cumulative probability-mass threshold varies. Red markers denote validation-selected mass values. Curves are development evidence and are not measurements of end-to-end inference speed.

**Figure 6 | Cache-aware scheduling changes the system-level consequence of expert prediction.** **a–c,** Cache hit rate under LRU, fixed native-budget prefetching, adaptive-mass prefetching and adaptive cache-aware scheduling at three cache capacities. **d–f,** Corresponding total expert transfers. Results are trace-driven replay on the limit=100 development traces, not latency measurements from a deployed MoE server. The plots retain failure boundaries, including settings where a higher hit rate requires substantially more transfers.

## Extended Data figures

**Extended Data Figure 1 | Composition of the limit=1000 cross-model cache.** Token-layer samples and request counts are shown by train, validation and test split for the three model traces. Requests, rather than token-layer samples, define the disjoint split unit.

**Extended Data Figure 2 | Development-stage sensitivity to route-history length.** Qwen-only Recall@8 across history lengths 1–16 using the earlier gated predictor at limit=100. This legacy experiment motivates temporal-context selection but is not a cross-model RouteCast V3 result.

**Extended Data Figure 3 | RouteCast performance across development and scale-confirmation settings.** Native-budget recall at limit=100 and stable limit=1000. The comparison summarizes empirical scale behaviour; because the smaller result is also used during method development, it should not be interpreted as an independent learning-curve experiment.

**Extended Data Figure 4 | Three-seed reproducibility of the cross-model development experiment.** Points show mean native-budget recall and error bars show one sample standard deviation across seeds 2024, 2025 and 2026 at limit=100.

**Extended Data Figure 5 | Layer-wise heterogeneity in the legacy Qwen-only predictor.** Recall@8 is reported separately across MoE layers, with the best and worst layers marked. The result establishes layer heterogeneity but is not a current cross-model RouteCast V3 layer analysis.

**Extended Data Figure 6 | Per-model calibration metrics.** BCE, Brier score and ECE before and after temperature scaling on the cross-model limit=100 test split. Lower values indicate better calibration.

**Extended Data Figure 7 | Legacy predictor-overhead microbenchmark.** Parameter count, serialized size, per-sample inference time and peak GPU memory are displayed using explicitly labelled scale factors. The benchmark was measured for the earlier Qwen-only gate and must not be reported as RouteCast V3 overhead.

**Extended Data Figure 8 | Complete cache-replay outcome matrix.** Cache hit rate, total transfers and prefetch precision for every model, cache capacity and replay policy. Values are recorded directly in each cell. Zero prefetch precision for LRU reflects the absence of issued prefetches rather than failed prefetch predictions.

