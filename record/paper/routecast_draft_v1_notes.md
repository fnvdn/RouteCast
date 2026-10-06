# RouteCast draft v1 notes

## One-sentence argument

Across heterogeneous MoE routing traces, RouteCast improves next-token expert forecasting by conditionally combining statistical and temporal route evidence, then uses calibrated confidence to prevent ranking gains from being translated blindly into wasteful expert transfers; the evidence is request-disjoint and trace driven, not an end-to-end deployment claim.

## Terminology ledger

| Canonical term | Use |
|---|---|
| RouteCast V3 | Current cross-model forecaster |
| native routing width / native-K | Number of experts selected by the original router |
| route-only | Uses observed discrete expert selections, not hidden states |
| adaptive mass | Cumulative calibrated-probability threshold controlling candidate count |
| confidence rejection | Decision to issue fewer or zero predicted prefetches |
| cache-aware admission | Residency-aware decision that converts candidates into transfers |
| trace-driven cache replay | Offline replay; never call it deployed acceleration |

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| RouteCast improves native-K forecasting across three MoEs | Stable limit=1000 Table 1 / Fig. 2 | Supported |
| Improvement over GRU-only is request-level robust | 10,000 paired bootstrap resamples | Supported for seed 2026 |
| Cross-token history is broadly useful | Three-model limit=100 ablation | Supported as development evidence |
| Every statistical branch is universally useful | DeepSeek/Llama ablation | Rejected; manuscript reports heterogeneity |
| Calibration provides a usable mass interface | Per-model reliability and validation mass curves | Supported as development evidence |
| Selective scheduling can reduce unfiltered adaptive-mass transfers | Trace-driven cache replay | Supported |
| RouteCast universally dominates LRU | Cache replay failure regions | Not supported and not claimed |
| RouteCast accelerates deployed MoE inference | No end-to-end server experiment | Not supported |

## Evidence allocation

- Main text: stable limit=1000 forecasting result, paired CI, essential budget trade-off, component heterogeneity, calibration mechanism, cache replay benefit and failure boundary.
- Appendix/extended data: complete cache matrix, dataset composition, history sweep, scale comparison, seed screening, layer heterogeneity, detailed calibration and legacy overhead.
- Pending before a stronger submission: matched TCN/Transformer baseline, limit=1000 multi-seed RouteCast, current V3 overhead, and real-system latency/throughput if resources permit.

## Items requiring author verification

1. Verify every bibliography entry against the original publisher/arXiv record before submission; the current bibliography preserves the supplied reference set.
2. Confirm whether the target conference permits an appendix within the main page limit or requires supplementary material.
3. Decide whether to retain all three future-work items in the main Discussion after the target venue and page limit are known.

