# RouteCast A会冲刺执行清单

> 执行原则：严格按编号推进；每完成一项，必须保存 JSON/CSV/Markdown 结果，并同步更新 `RouteCast_论文写作总档.md`。没有结果文件的终端截图不算完成。

## 当前状态

- 核心算法：已冻结为 RouteCast V3 + Calibration + Budget mode + Abstention + Cache-aware scheduling。
- 开发规模：limit=100 已形成完整证据链。
- 最大风险：limit=1000 的 Group-DRO 权重出现 NaN；强神经 baseline、统计置信区间和完整开销尚未补齐。
- 系统边界：只做 trace-driven cache replay，不声称真实端到端加速。

---

# 阶段一：冻结协议与修复大规模实验

## 1.1 冻结最终实验协议

- [x] 固定 request-level train/validation/test 划分。
- [x] 固定随机种子集合：2024、2025、2026。
- [x] 固定历史长度：H=16。
- [x] 固定原生预算：Qwen K=8、DeepSeek K=8、Llama K=1。
- [x] 固定预测目标：下一 token、同一 MoE 层的真实专家集合。
- [x] 固定最终算法超参数来源：只允许验证集选择，不读取测试标签。

完成标准：生成 `protocol/final_protocol.json` 和 `protocol/final_protocol.md`。

## 1.2 修复 limit=1000 Group-DRO NaN

- [x] 定位 NaN 污染阶段；旧日志未保留逐 batch 信息，无法无证据虚构精确 batch，新代码已支持首次异常精确定位。
- [x] 检查 group loss、softmax logits、权重归一化和梯度。
- [x] 加入 finite 检查、FP32 loss、BF16/GradScaler、稳定 Group-DRO softmax 和梯度裁剪。
- [x] 完成 limit=100 与 limit=1000 三模型短回归测试，正常路径均通过。
- [x] 稳定版 limit=1000 三轮全量复训完成；三轮 Group-DRO 权重均有限且归一化，最终 JSON 无 NaN。

完成标准：所有 epoch 的 group weights 有限且和为1；训练 loss 有限；模型和完整日志保存。

## 1.3 limit=1000 主预测复现

- [x] 运行最终 RouteCast V3（stable，seed=2026，best epoch=3）。
- [x] 运行 Paper Heatmap。
- [x] 运行 Request-aware Heatmap。
- [x] 运行 GRU-only。
- [x] 保存并汇总四种方法在三模型上的 Recall/Precision/MRR/NDCG；RouteCast 与 GRU-only 已保存 request-level 原始计数。

完成标准：已生成 `results/main_limit1000/main_forecasting_limit1000.json/.csv/.md`，可进入论文主表。

---

# 阶段二：强 baseline 与公平性

## 2.1 实现一个强神经 baseline

- [ ] 优先实现轻量 TCN；如文献证据更强，可替换为轻量 Transformer Encoder。
- [ ] 使用与 RouteCast 相同的 H=16 历史和相同 request split。
- [ ] 不额外使用 RouteCast 专属统计特征，或分别报告 history-only 与 matched-input 两种版本。
- [ ] 记录参数量、训练时间和推理时间。

完成标准：强 baseline 在三模型、三种随机种子上有完整结果。

## 2.2 公平输入与计算预算表

- [ ] 记录每个方法使用的输入信息。
- [ ] 记录历史长度。
- [ ] 记录参数量。
- [ ] 记录训练数据量。
- [ ] 记录单 token-layer 推理开销。
- [ ] 标出学习型与非学习型方法。

完成标准：生成论文 `Table: Baseline fairness and complexity`。

## 2.3 Oracle upper bound

- [ ] Oracle route predictor。
- [ ] Oracle cache admission/eviction。
- [ ] 计算 RouteCast 与理论上界的差距。

完成标准：给出预测和缓存两个层面的上界，不把 Oracle 当作可部署 baseline。

---

# 阶段三：指标、统计与复现

## 3.1 补齐预测指标

- [x] Recall@K。
- [x] Precision@K。
- [x] MRR。
- [x] NDCG@K。
- [x] 平均命中专家数。
- [x] 所有指标在同一评价实现中计算，并汇总至 limit=1000 主表。

完成标准：三模型、所有 baseline 使用完全相同的指标实现。

## 3.2 Request-level bootstrap 置信区间

- [x] 以 request 为重采样单位，不以 token-layer 为独立样本。
- [x] 计算 RouteCast 与当前最强 baseline（GRU-only）的 paired delta。
- [x] 计算 95% bootstrap CI，并额外保存与 Paper Heatmap 的比较。
- [x] 固定 bootstrap seed=2026、repeats=10000。

完成标准：主结果全部带 mean/delta/95% CI。

## 3.3 最终算法多随机种子

- [ ] 三个种子重新训练最终预测器。
- [ ] 各种子重新校准温度。
- [ ] 各种子重新选择 mass 与 abstention threshold。
- [ ] 各种子运行最终缓存回放。

完成标准：预测 Recall、平均 K、hit rate、total transfers 均报告 mean±std。

## 3.4 issued-prefetch 指标

- [ ] 候选集合 Recall。
- [ ] 拒绝后的 issued-prefetch coverage/recall。
- [ ] issued-prefetch precision。
- [ ] abstention rate。
- [ ] 每 token-layer 实际预取数量。

完成标准：能清楚区分“预测到了”“进入候选”“真正搬运”三个阶段。

---

# 阶段四：算法解释与敏感性

## 4.1 完整模块消融

- [ ] No history。
- [ ] No one-step transition。
- [ ] No two-hop transition。
- [ ] No prefill prior。
- [ ] No popularity prior。
- [ ] Fixed fusion 替换动态门控。
- [ ] No calibration。
- [ ] Fixed native K 替换 adaptive mass。
- [ ] No abstention。
- [ ] No cache-aware admission/eviction。

完成标准：预测模块与决策模块各有一张消融表。

## 4.2 超参数敏感性

- [ ] history H。
- [ ] mass。
- [ ] Recall retention rho。
- [ ] abstention threshold tau。
- [ ] waste penalty lambda_w。
- [ ] cache decay。
- [ ] eviction margin。

完成标准：主结论不依赖单一偶然参数；详细曲线放SI。

## 4.3 异质性与失败模式

- [ ] 按模型分析。
- [ ] 按层分析。
- [ ] 按请求长度分析。
- [ ] 按 Prefill 长度分析。
- [ ] 按路由熵/置信度分析。
- [ ] 分析 DeepSeek 2×/4×为何不如 LRU。

完成标准：Discussion 的限制与解释都有实验支撑。

---

# 阶段五：开销与可复现性

## 5.1 预测器开销

- [ ] 参数量与模型大小。
- [ ] CPU单样本/批量延迟。
- [ ] GPU单样本/批量延迟。
- [ ] 峰值CPU/GPU内存。
- [ ] 不同 batch size 的吞吐。

## 5.2 决策器与缓存调度开销

- [ ] Temperature、mass、abstention 计算时间。
- [ ] Cache admission/eviction 每决策耗时。
- [ ] 与 GRU、TCN/Transformer、EAM 检索开销比较。
- [ ] 报告 Python 原型开销与算法复杂度，避免把原型耗时当部署耗时。

## 5.3 可复现包

- [ ] 固定 requirements/environment。
- [ ] 记录 Python、PyTorch、CUDA、GPU、CPU、OS。
- [ ] 一键构建缓存脚本。
- [ ] 一键训练脚本。
- [ ] 一键评估脚本。
- [ ] 每个脚本支持 seed、进度输出、summary 和 resume。
- [ ] README 给出从数据到论文表格的完整命令。

完成标准：新环境可按 README 重建核心结果。

---

# 阶段六：论文图表

## 6.1 主图

- [ ] Fig. 1 RouteCast 四层框架。
- [ ] Fig. 2 三模型强 baseline 对比。
- [ ] Fig. 3 Calibration + mass + abstention。
- [ ] Fig. 4 Cache hit–transfer Pareto。
- [ ] Fig. 5 模块消融与模型异质性。

## 6.2 主表

- [ ] Table 1 主预测结果。
- [ ] Table 2 缓存与传输结果。
- [ ] Table 3 核心消融。
- [ ] Table 4 公平性与开销。

## 6.3 补充图表

- [ ] Reliability diagrams。
- [ ] 完整 K-budget curves。
- [ ] 随机种子与 bootstrap CI。
- [ ] 层级、长度和熵分组。
- [ ] 全部敏感性曲线。

完成标准：SVG/PDF/PNG三种格式，统一字体、颜色、尺寸和图注。

---

# 阶段七：论文写作

## 7.1 Background / Introduction

- [ ] 梳理 MoE专家驻留与预取问题。
- [ ] 梳理统计预测、请求级预测和学习型预测。
- [ ] 明确“预测准确不等于系统收益”的研究缺口。
- [ ] 写出三项贡献，不使用无证据的 first/SOTA。

## 7.2 Related Work

- [ ] Expert offloading and caching。
- [ ] Statistics-based route prediction。
- [ ] Request-level activation prediction。
- [ ] Learning-based route forecasting。
- [ ] Calibration/selective prediction/cache admission。
- [ ] 所有引用逐条核验原文、DOI/arXiv和发表状态。

## 7.3 Method

- [ ] Task formulation。
- [ ] Multi-source predictor。
- [ ] Dynamic gating。
- [ ] Training objectives。
- [ ] Temperature calibration。
- [ ] Budget-preserving mass。
- [ ] Abstention。
- [ ] Cache-aware scheduling。
- [ ] Complexity analysis。

## 7.4 Experiments / Results

- [ ] Setup and protocol。
- [ ] Main forecasting results。
- [ ] Generalization and robustness。
- [ ] Ablation。
- [ ] Calibration and budgeting。
- [ ] Cache replay Pareto analysis。
- [ ] Overhead and failure modes。

## 7.5 Discussion / Conclusion

- [ ] 解释三模型差异。
- [ ] 解释 DeepSeek失败边界。
- [ ] 明确 trace replay 与真实部署差距。
- [ ] 不把传输降低直接写成延迟加速。
- [ ] 给出可验证的未来真实部署路线。

---

# 阶段八：投稿前审计

- [ ] 标题、摘要和贡献中的每个数字可追溯到最终 JSON。
- [ ] 图表数据与正文数字一致。
- [ ] 没有使用被淘汰的旧缓存回放结果。
- [ ] 没有使用存在 NaN 的 limit=1000 结果。
- [ ] 每个 baseline 协议公平。
- [ ] 每项主要主张都有统计证据。
- [ ] 失败结果没有被藏入SI。
- [ ] 代码、数据和随机种子可复现。
- [ ] 引用全部核验。
- [ ] 进行一次模拟审稿与逐条修订。

---

# 推荐执行顺序

1. **1.1 冻结实验协议**
2. **1.2 修复 limit=1000 NaN**
3. **1.3 重新生成 limit=1000 主结果**
4. **2.1 强神经 baseline**
5. **3.1–3.4 指标与统计**
6. **4.1–4.3 消融与失败模式**
7. **5.1–5.3 开销和复现包**
8. **6.1–6.3 最终图表**
9. **7.1–7.5 完整论文**
10. **8 投稿前审计**

## 当前下一项

> **Task 1.2：修复 limit=1000 Group-DRO NaN。**
