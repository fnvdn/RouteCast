# Cross-token 强基线比较（limit=100）

| 模型 | 原生K | 方法 | Recall | 相对RouteCast |
|---|---:|---|---:|---:|
| qwen3 | 8 | Paper Heatmap | 0.461987 | -0.093834 |
| qwen3 | 8 | Request-aware Heatmap | 0.443498 | -0.112322 |
| qwen3 | 8 | GRU-only | 0.526260 | -0.029560 |
| qwen3 | 8 | RouteCast V3 | 0.555820 | +0.000000 |
| deepseek_r1 | 8 | Paper Heatmap | 0.187861 | -0.099748 |
| deepseek_r1 | 8 | Request-aware Heatmap | 0.187840 | -0.099769 |
| deepseek_r1 | 8 | GRU-only | 0.272574 | -0.015035 |
| deepseek_r1 | 8 | RouteCast V3 | 0.287609 | +0.000000 |
| llama4_maverick | 1 | Paper Heatmap | 0.313287 | -0.012831 |
| llama4_maverick | 1 | Request-aware Heatmap | 0.317898 | -0.008219 |
| llama4_maverick | 1 | GRU-only | 0.303060 | -0.023058 |
| llama4_maverick | 1 | RouteCast V3 | 0.326118 | +0.000000 |
