# Revision Step 2: Cross-model training ablation results

## Protocol

- Dataset: packed `limit=1000`, request-disjoint split.
- Seed: 2026.
- Compared configurations:
  1. one shared RouteCast predictor with static sample balancing and Group-DRO;
  2. one shared predictor with static balancing and uniform group weights;
  3. one independently trained predictor per model family.
- Metric: native-budget Recall.

## Verified results

| Model | Shared + Group-DRO | Shared, no DRO | Per-model | Full minus no DRO | Full minus per-model |
|---|---:|---:|---:|---:|---:|
| Qwen3 | 0.555837 | 0.555895 | 0.558416 | -0.000059 | -0.002579 |
| DeepSeek-R1 | 0.317227 | 0.316581 | 0.312466 | +0.000646 | +0.004762 |
| Llama 4 Maverick | 0.360242 | 0.361107 | 0.364363 | -0.000865 | -0.004121 |
| Macro average | 0.411102 | 0.411195 | 0.411748 | -0.000093 | -0.000646 |

## Supported interpretation

Group-DRO did not produce a measurable numerical advantage in this run. Its largest model-level change was below 0.0009 Recall, and macro-average Recall was 0.000093 lower than uniform group weighting.

Independent per-model training achieved only 0.000646 higher macro-average Recall than the shared predictor. The direction varied by architecture: shared learning helped DeepSeek-R1 but slightly reduced Qwen3 and Llama 4 Maverick Recall. The evidence therefore supports a unified shared predictor with little aggregate accuracy loss, not a claim that shared training or Group-DRO uniformly improves accuracy.

## Manuscript placement

- Methods: matched ablation definitions were added to Cross-Model Learning.
- Results: a dedicated subsection and compact table report all values.
- Discussion: the portability interpretation is bounded explicitly.
- Limitations: the limit=1000 coverage statement now includes this ablation.
