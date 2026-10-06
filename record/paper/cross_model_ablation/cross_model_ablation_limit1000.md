# Cross-model training ablation (limit=1000)

| Model | Shared + Group-DRO | Shared, no DRO | Per-model | DRO delta | Shared-training delta |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.555837 | 0.555895 | 0.558416 | -0.000059 | -0.002579 |
| deepseek_r1 | 0.317227 | 0.316581 | 0.312466 | +0.000646 | +0.004762 |
| llama4_maverick | 0.360242 | 0.361107 | 0.364363 | -0.000865 | -0.004121 |
| macro_average | 0.411102 | 0.411195 | 0.411748 | -0.000093 | -0.000646 |

Positive DRO delta means adaptive Group-DRO improved Recall over uniform group weighting.
Positive shared-training delta means the shared model improved Recall over independent per-model training.
