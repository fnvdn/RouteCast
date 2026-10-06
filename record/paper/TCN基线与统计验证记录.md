# TCN 基线与统计验证记录

## 结论

TCN-only 是比 GRU-only 略强的纯时序神经基线，但 RouteCast 在三个模型上均稳定超过 TCN。limit=1000 主实验的 Recall 绝对提升分别为 Qwen3 `+0.028147`、DeepSeek-R1 `+0.045478`、Llama4 Maverick `+0.057871`。以完整 request 为重采样单位的 10,000 次配对 bootstrap 的 95% 置信区间均不跨越 0，说明提升不是由少量 token-layer 样本或单个请求偶然造成。

## limit=1000 主结果

| Model | Native K | TCN-only Recall | RouteCast Recall | RouteCast - TCN | 95% paired request bootstrap CI | Test requests |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | 8 | 0.527690 | 0.555837 | +0.028147 | [0.026459, 0.029859] | 157 |
| DeepSeek-R1 | 8 | 0.271749 | 0.317227 | +0.045478 | [0.040339, 0.050988] | 153 |
| Llama4 Maverick | 1 | 0.302371 | 0.360242 | +0.057871 | [0.054598, 0.060975] | 147 |

Bootstrap 协议：固定测试 request 集，成对保留 RouteCast 与 TCN 在同一 request 上的结果；每次以 request 为单位有放回抽样，再计算总体 Recall 差值。该设计保留同一 request 内 token-layer 样本的相关性。

## 三随机种子稳定性筛查

该实验使用 limit=100 development setting，种子为 2024、2025、2026；所有方法使用相同 request 划分、目标、训练轮次与验证集选优规则。它用于检查优化稳定性，不替代 limit=1000 主实验。

| Model | K | RouteCast mean ± SD | TCN mean ± SD | Delta mean ± SD | RouteCast wins |
|---|---:|---:|---:|---:|---:|
| Qwen3 | 8 | 0.558412 ± 0.002246 | 0.526686 ± 0.000637 | +0.031725 ± 0.001660 | 3/3 |
| DeepSeek-R1 | 8 | 0.290884 ± 0.005325 | 0.271300 ± 0.001062 | +0.019583 ± 0.006218 | 3/3 |
| Llama4 Maverick | 1 | 0.328125 ± 0.002363 | 0.306252 ± 0.002298 | +0.021873 ± 0.004138 | 3/3 |

## 论文可用表述

TCN-only slightly outperformed GRU-only, establishing a stronger temporal neural baseline. RouteCast nevertheless improved native-budget Recall over TCN-only by 0.0281, 0.0455, and 0.0579 on Qwen3, DeepSeek-R1, and Llama4 Maverick, respectively. Paired request-level bootstrap intervals excluded zero for all three models. In a matched three-seed development screening, RouteCast also outperformed TCN in all nine model–seed comparisons. Together, these results indicate that the gains arise from multi-source conditional fusion rather than from temporal encoding alone.

## 产物位置

- 主表：`results/main_limit1000/main_forecasting_limit1000.csv`
- TCN 统一指标：`results/tcn_limit1000_unified_metrics.json`
- request-level bootstrap：`results/main_limit1000/bootstrap_routecast_vs_tcn_limit1000.json`
- 三种子汇总：`results/seed_repro_tcn_limit100/tcn_routecast_seed_summary.json`
- 三种子逐次结果：`results/seed_repro_tcn_limit100/tcn_routecast_seed_runs.csv`
- 主基线图：`figures/split/Fig2a_main_grouped_comparison.pdf`
- bootstrap 图：`figures/split/Fig2b_paired_bootstrap.pdf`
- 三种子补充图：`figures/split/FigS_tcn_seed_stability.pdf`
