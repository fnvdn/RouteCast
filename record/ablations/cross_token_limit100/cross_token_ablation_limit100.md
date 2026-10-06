# Cross-token RouteCast V3 模块消融（limit=100）

| 模型 | 原生K | 版本 | Recall | 相对完整模型变化 |
|---|---:|---|---:|---:|
| qwen3 | 8 | full | 0.555820 | +0.000000 |
| qwen3 | 8 | no_history | 0.541027 | -0.014793 |
| qwen3 | 8 | no_two_hop | 0.546326 | -0.009494 |
| qwen3 | 8 | no_prefill | 0.551244 | -0.004577 |
| deepseek_r1 | 8 | full | 0.287609 | +0.000000 |
| deepseek_r1 | 8 | no_history | 0.265053 | -0.022556 |
| deepseek_r1 | 8 | no_two_hop | 0.294057 | +0.006449 |
| deepseek_r1 | 8 | no_prefill | 0.294554 | +0.006946 |
| llama4_maverick | 1 | full | 0.326118 | +0.000000 |
| llama4_maverick | 1 | no_history | 0.320475 | -0.005642 |
| llama4_maverick | 1 | no_two_hop | 0.327148 | +0.001031 |
| llama4_maverick | 1 | no_prefill | 0.325792 | -0.000326 |
