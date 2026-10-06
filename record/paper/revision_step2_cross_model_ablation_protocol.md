# Revision Step 2: Cross-model Training Ablation Protocol

## Scientific questions

1. Does shared training across the three observed MoE families improve
   native-budget Recall relative to three independently trained predictors?
2. Within shared training, does adaptive Group-DRO improve Recall relative to
   fixed uniform group weights?

## Compared conditions

| Condition | Training data | Static sample balancing | Group-DRO | Checkpoint rule |
|---|---|---:|---:|---|
| Shared + Group-DRO | All three model families | Yes | Yes | Validation macro native-budget Recall |
| Shared, no Group-DRO | All three model families | Yes | No, uniform weights | Validation macro native-budget Recall |
| Per-model | One family per predictor | Trivial weight 1 | Not applicable | That model's validation native-budget Recall |

All conditions use the same packed limit=1000 cache, request-disjoint splits,
RouteCast V3 architecture, six branches, multi-budget objective, history 16,
seed 2026, batch size 256, learning rate 6e-4, weight decay 1e-4, three epochs,
and native-budget test metrics.

## Reported effects

- Group-DRO contribution = Recall(shared + Group-DRO) - Recall(shared, no DRO)
- Shared-training contribution = Recall(shared + Group-DRO) - Recall(per-model)
- Both effects are reported separately for Qwen3, DeepSeek-R1, and Llama4
  Maverick, plus their unweighted macro-average.

## Interpretation boundary

This experiment tests shared supervised training across three observed model
families. It does not test zero-shot transfer to an unseen MoE architecture.
