# 无服务器版 MoE 专家预取研究路线

## 研究定位

本项目暂不声称真实 GPU/PCIe/网络环境下的端到端加速，而定位为：基于公开 MoE 路由轨迹的风险可控、预算约束专家预取算法与可复现实验基准。

核心目标是：在真实专家路由序列已知、但未来路由未知的条件下，预测下一层专家集合，并在给定预取预算下控制漏取风险、减少无效传输和缓存污染。

## 当前成果与保留版本

- RouteCast 基础预测器：Global Popularity、Cross-layer Transition、GRU。
- Gated Fusion：根据上下文动态融合三个分支。
- Temperature Scaling：校准专家概率。
- Dynamic Top-K：根据累计概率 mass 生成可变预取集合。
- Hardware-aware mass scheduler：仅作为归一化硬件 profile 的敏感性分析，不作为真实系统加速证据。
- 旧版本模型、实验结果和代码全部保留，不覆盖原始预测器。

## 最终算法路线

### 1. 置换等变专家打分器

对每个候选专家使用共享打分网络，使专家编号置换不改变模型逻辑，并支持不同专家数量的 MoE 模型：

\[
s_{t,l,e}=f_\theta(\mathbf a^{(H)}_{t,l,e},q^{\mathrm{trans}}_{t,l,e},q^{\mathrm{pop}}_{t,l,e},\mathbf c_{t,l}).
\]

其中历史激活、跨层转移、流行度和层/token 上下文共同决定专家分数。

### 2. 概率校准

对融合分数进行温度校准：

\[
p_{t,l,e}=\frac{\exp(s_{t,l,e}/T)}{\sum_j\exp(s_{t,l,j}/T)}.
\]

温度参数只在验证请求上学习，测试请求严格隔离。

### 3. 请求级风险控制

把一个 request 作为校准单位，定义专家漏取风险：

\[
R_i(\lambda)=\frac{1}{N_i}\sum_{t,l}\left(1-\frac{|\mathcal S_\lambda(x_{i,t,l})\cap\mathcal Y_{i,t,l}|}{8}\right).
\]

在校准请求上选择最小阈值，使期望漏取风险满足目标：

\[
\mathbb E[R(\lambda^*)]\leq \alpha.
\]

该步骤采用 conformal risk control，报告实际 coverage、risk violation 和置信区间。

### 4. 预算约束集合选择

在满足风险约束的前提下，最小化预取成本：

\[
\min_\lambda\;\mathbb E[C_{\mathrm{transfer}}+\beta C_{\mathrm{waste}}+\gamma C_{\mathrm{pollution}}],
\quad\mathrm{s.t.}\quad \mathbb E[R(\lambda)]\leq\alpha,
\quad |\mathcal S_\lambda|\leq K_{\max}.
\]

不使用虚构的毫秒延迟；传输量、错误预取、缓存污染和专家切换均直接从 trace 计算。

## 数据与实验计划

1. 使用 Qwen3-235B 现有数据完成算法回归测试。
2. 分层下载 DeepSeek-R1、Llama-4-Maverick、Kimi-K2-Thinking 的代表性请求。
3. 采用 request-level 固定 train/validation/test 划分，避免 token 泄漏。
4. 统一不同模型的层数、专家数、Top-K 和 prefill/decode 格式。
5. 比较 Random、Popularity、Transition、Markov、GRU、Gated Fusion、MoE-Infinity-style trace cache、RouteCast-RC 和 Oracle。
6. 建立 trace-driven cache/bandwidth simulator，报告 hit rate、demand miss、wasted prefetch、cache pollution、transfer volume、oracle regret 和 Recall–Cost Pareto 曲线。
7. 完成跨任务、跨模型、OOD 请求、不同校准集大小、不同随机种子和逐层失败分析。
8. 采用 request-level bootstrap 95% CI，并报告效应量和最差 10% 请求。

## 论文边界

论文可以声称：算法在公开多模型 trace 上具有更好的风险—成本折中、跨工作负载泛化和可复现的离线决策性能。

论文不能声称：已经在真实 GPU/PCIe/网络系统上获得端到端吞吐或延迟加速。真实硬件实验作为后续工作明确说明。

## 执行顺序

1. 迁移并核对 E 盘项目到 `F:\MOEresearch\Predictor`。
2. 先完成当前论文的交差版润色和编译检查。
3. 统一多模型 trace adapter。
4. 实现置换等变打分器。
5. 实现 request-level conformal risk control。
6. 实现缓存/带宽 trace simulator。
7. 完成多模型和 OOD 实验。
8. 用统一总表、图和统计结果重写论文。
