# RouteCast 最终实验协议 v1.0

状态：**冻结，用于 limit=1000 NaN 修复及后续最终复现。**

机器可读版本：`final_protocol.json`。

## 1. 任务

使用截至 token `t` 已观测的专家路由（包括当前 token 路由），预测 token `t+1` 在同一 MoE 层的真实专家集合。历史长度固定为 `H=16`。数据划分单位为 request，评价基本单位为 token-layer。

## 2. 固定 request 划分

划分由 request 相对路径唯一确定：

```text
key = relative_path.replace("\\", "/")
value = int.from_bytes(SHA1(key)[0:4], "big") / 2^32
train: value < 0.70
val:   0.70 <= value < 0.85
test:  value >= 0.85
```

因此数据成员资格与训练随机种子无关。随机种子只影响参数初始化、训练文件顺序、batch采样和CUDA随机状态。

## 3. 模型结构与原生预算

| Model | Experts | Active MoE layers | Native K |
|---|---:|---:|---:|
| Qwen3-235B-A22B | 128 | 94 | 8 |
| DeepSeek-R1 | 256 | 58 | 8 |
| Llama4 Maverick | 128 | 24 | 1 |

## 4. 数据规模

### limit=100

| Model | Train requests/examples | Val requests/examples | Test requests/examples |
|---|---:|---:|---:|
| Qwen3 | 69 / 823,534 | 14 / 168,448 | 17 / 203,510 |
| DeepSeek-R1 | 69 / 512,120 | 14 / 103,876 | 17 / 124,505 |
| Llama4 | 70 / 215,040 | 18 / 55,296 | 12 / 36,864 |

### limit=1000

| Model | Train requests/examples | Val requests/examples | Test requests/examples |
|---|---:|---:|---:|
| Qwen3 | 698 / 8,343,910 | 145 / 1,733,736 | 157 / 1,885,170 |
| DeepSeek-R1 | 708 / 5,199,528 | 139 / 1,022,736 | 153 / 1,123,638 |
| Llama4 | 708 / 2,171,400 | 145 / 445,440 | 147 / 450,672 |

## 5. 训练协议

- Seeds：2024、2025、2026；单次主表默认展示2026，最终报告 mean±std。
- Epochs：3。
- Batch size：256。
- Optimizer：AdamW，`lr=6e-4`，`weight_decay=1e-4`。
- Gradient clipping：1.0。
- Group-DRO eta：0.8。
- Checkpoint：验证集三模型 native-K Recall 的宏平均最高者。
- Loss：`0.20 BCE + 0.25 listwise + 0.35 ranking + 0.20 multi-budget`。
- AMP：CUDA FP16 autocast。

测试集不得用于模型、温度、mass、拒绝阈值或缓存超参数选择。

## 6. 校准、预算与拒绝

1. 冻结预测器；每个模型在验证集学习单一温度。
2. mass 网格固定为0.01–1.00，步长0.01。
3. 逐样本硬约束：`1 <= K_t <= K_native`。
4. 目标 Recall retention：0.98。
5. `mass=1.0` 明确退化至完整原生 K。
6. 拒绝阈值仅在验证集按 `useful - wasted` 选择，`waste_penalty=1.0`。
7. 阈值1.01允许完全不预取。

## 7. 缓存回放

- 每个 request 独立重置缓存。
- 使用首个预测前已观测的当前路由进行 warm start。
- 容量：`active_layers × native_K × {1,2,4}` 个专家条目。
- Cache decay：0.90。
- Eviction margin：0.01。
- LRU 必须是纯 recency-only，不得读取预测分数。
- 报告 LRU、fixed-native、adaptive-mass、完整 selective cache-aware 四种策略。

## 8. 必报指标

- 预测：Recall@K、Precision@K、MRR、NDCG@K、平均命中专家数。
- 校准：BCE、Brier、ECE。
- 决策：平均候选K、候选Recall、issued-prefetch coverage/precision、abstention rate。
- 缓存：hit rate、on-demand loads、prefetch loads、useful/wasted prefetch、prefetch precision、total transfers。
- 统计：以 request 为单位的 paired bootstrap 95% CI。

## 9. 论文主张边界

本协议支持专家预测质量与轨迹驱动缓存/传输结果。没有真实部署前，不声称端到端 latency、throughput、energy 或 PCIe overlap speedup。

## 10. 完成判据

该协议从现在起冻结。任何修改必须：

1. 新增 protocol version；
2. 写明修改原因；
3. 重新运行受影响的方法；
4. 不允许只对 RouteCast 改协议而不重跑 baseline。
