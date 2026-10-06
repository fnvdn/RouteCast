# MOE4 layout and manuscript polish audit

## Scope

- Source reviewed: `D:\MOE4.pdf` (7 pages).
- Revised source: `F:\MOEresearch\record\paper\routecast_draft_v1.tex`.
- Paper type: algorithmic systems/ML paper.
- Language and venue style: concise English for a generic IEEE conference manuscript.

## Layout findings

1. Splitting the original composite images fixed the main legibility problem.
2. Pages 5--7 still contained a long sequence of independent floats. The calibration and mass figures have therefore been reordered by model so that each reliability diagram is followed by its operating curve.
3. The source retains atomic figures rather than rebuilding them as one raster composite. This keeps labels readable and permits later venue-specific float tuning.
4. The revised manuscript contains more method equations than MOE4, so the final page count may increase. This is preferable to omitting definitions, but the Overleaf build must be inspected again.

## Canonical terminology

| Concept | Canonical form | Rule |
|---|---|---|
| Proposed method | RouteCast | Preserve capitalization everywhere. |
| Models | Qwen3; DeepSeek-R1; Llama4 Maverick | `Llama4` is allowed only as a compact table label. |
| Forecast target | next-token expert set | Do not alternate with next-layer prediction. |
| Main quality metric | native-budget Recall | Use Recall@8 for Qwen3/DeepSeek-R1 and Recall@1 for Llama4 Maverick when K must be explicit. |
| Candidate control | cumulative mass; adaptive candidate width | Do not call mass a hardware parameter. |
| Downstream evaluation | trace-driven cache replay | Do not describe it as measured system speedup. |

## Formula coverage after revision

The revised source explicitly defines:

1. task target, Recall@K, and Precision@K;
2. popularity, one-hop, two-hop, and request-prefill branches;
3. GRU temporal and multiscale branches;
4. per-branch normalization, confidence descriptors, context gate, and probability-mixture fusion;
5. BCE, listwise, ranking, multi-budget, and group-DRO objectives;
6. sigmoid/BCE temperature fitting and the separate softmax distribution used by cumulative mass;
7. dynamic K, validation mass selection, confidence rejection utility, cache-value decay, and admission rule;
8. MRR, NDCG, cache hit rate, total transfers, and prefetch precision.

## Important consistency correction

The implementation fits temperature with multilabel sigmoid binary cross-entropy, while cumulative-mass selection applies the learned temperature to a softmax distribution. MOE4 described only the latter. The revised equations now distinguish these two uses and match the code.

## Evidence allocation

| Evidence | Class | Location |
|---|---|---|
| limit=1000 main comparison and paired bootstrap | core discovery | Main text |
| budget expansion and waste | necessary support | Main text |
| cross-model ablation | mechanism/qualification | Main text |
| calibration and mass curves | decision-interface support | Main text |
| cache replay and failure cases | qualification | Main text |
| legacy Qwen-only diagnostics and extended matrices | robustness/provenance | Supplementary package |

## Remaining blocking checks

- Compile the revised source on Overleaf twice.
- Inspect the new PDF for overfull equations, float backlog, and references moving beyond the expected page budget.
- Verify the exact released-code definition of MRR/NDCG before submission if a separate evaluator, rather than the current report generator, produced Table I.
- Replace placeholder or unverified bibliographic metadata only after checking the cited primary sources.

## Figure-wall correction

The six independent calibration/mass floats and three cache floats created a figure-only page in the compiled draft. They have been removed from the main-text LaTeX and replaced by two claim-driven composites assembled from recorded data:

- `Fig5_compact_calibration_budget`: calibration-error change plus the three selected mass operating points;
- `Fig6_compact_cache_pareto`: one aligned Pareto panel per model with a shared legend.

The complete per-model reliability diagrams and mass curves remain in the supplementary figure bundle. Main-text interpretation now reports the relevant ECE changes, selected masses, average candidate widths, Recall values, transfer reductions, and negative model/capacity cases.
