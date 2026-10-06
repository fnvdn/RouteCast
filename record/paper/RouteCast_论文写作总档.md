---
title: "RouteCast 论文写作总档"
project: "RouteCast: Universal Cross-token Expert Forecasting and Selective Prefetching for Heterogeneous MoE Models"
document_type: "论文母稿、证据台账、图表规划与复现清单"
status: "算法与实验协议 v1.0 已冻结；论文证据整理中"
updated: "2026-08-24"
language: "中文证据母稿 + 英文论文接口"
---

# RouteCast 论文写作总档

> 本文档是后续论文写作的唯一母稿。背景、方法定义、实验协议、结果、图表、证据边界和待办事项均在此维护。原始 JSON/CSV/模型文件保持不动，并在文末建立来源索引。

## 0. 当前版本与不可越过的边界

### 0.1 一句话论证

在异构 MoE 路由轨迹上，RouteCast 通过多源跨 token 预测、概率校准、自适应预算与选择性缓存调度，提高下一 token 的专家预测质量，并在轨迹驱动缓存回放中减少固定 Top-K 预取的无效传输。

### 0.2 当前可支持的主张

1. RouteCast V3 在 Qwen3、DeepSeek-R1 和 Llama4 Maverick 上均优于 Paper Heatmap、Request-aware Heatmap、GRU-only 和轨迹级 MoE-Infinity EAM。
2. Temperature Scaling 在不改变专家排序和 Recall 的情况下改善 Qwen3 与 DeepSeek-R1 的概率校准；Llama4 的校准收益有限。
3. Budget mode 保证逐样本 `K_t <= K_native`，并在验证集上选择满足 Recall 保留目标的最小预算；无法缩减时退化至固定原生 K。
4. 选择性缓存感知调度通过置信度拒绝和价值保护，大幅减少固定 Top-K 的预取次数与总专家传输。
5. 在正确的纯 LRU 对照下，完整策略在 Qwen3 的全部缓存容量、DeepSeek-R1 的 1×容量以及 Llama4 的 2×/4×容量同时获得更高命中率与更少总传输。

### 0.3 当前不能声称

- 尚未在真实 MoE 推理服务器上测量端到端 latency、throughput 或 energy。
- 缓存实验是 trace-driven replay，不是完整专家卸载系统部署。
- 不能将总传输降低直接等同于等比例推理加速。
- 不能声称在所有模型和缓存容量下全面优于 LRU；DeepSeek-R1 2×/4×为明确失败边界。
- 旧 limit=1000 报告的 `group_weights=NaN` 已定位并由稳定版全量复训取代；旧文件仅作失效证据，不得进入论文表格。

---

# 1. 术语与符号台账

| 标准术语 | 首次定义 | 禁止混用或备注 |
|---|---|---|
| RouteCast | 本文提出的通用跨 token 专家预测与选择性预取框架 | 不再使用 RouteCast V1/V2 指代最终算法 |
| RouteCast V3 predictor | 冻结的多模型通用专家预测器 | 与完整 RouteCast 系统区分 |
| cross-token forecasting | 使用已观测 token 路由预测下一 token 同层专家 | 不写成 cross-layer prediction |
| native K | 模型路由器真实激活专家数；Qwen/DeepSeek 为8，Llama为1 | 与动态平均 K 区分 |
| adaptive mass budgeting | 由累计概率质量确定动态 K 的预算机制 | 当前版本不称 hardware-aware mass |
| selective prefetching | 低置信度时允许不执行预取 | 可令实际预取数为0 |
| cache-aware scheduling | 基于驻留状态、候选价值与淘汰价值决定实际预取 | 不等于缓存回放本身 |
| trace-driven cache replay | 根据真实专家路由轨迹模拟缓存命中与专家传输 | 不称 end-to-end deployment |
| Recall@K | 预测 Top-K 与真实专家集合交集数除以真实专家数 | 原生 K 时 Qwen/DeepSeek 的 Precision 数值等于 Recall |
| prefetch precision | 有效预取次数除以全部预取次数 | 与预测 Precision@K 区分 |
| total transfers | 按需加载次数与预取加载次数之和 | 不直接等价于延迟 |
| EAM/EAMC | Expert Activation Matrix / Collection，MoE-Infinity 轨迹级复现组件 | 不声称复现整个 MoE-Infinity 系统 |

核心符号：

- `t`：当前已观测 token；目标为预测 token `t+1`。
- `l`：MoE 层索引。
- `e`：专家索引。
- `E`：专家总数。
- `K_native`：模型原生激活专家数。
- `s_{t,l,e}`：RouteCast 输出的专家 logit/score。
- `p_{t,l,e}`：经过温度缩放后的归一化排序概率。
- `m`：累计概率质量阈值。
- `K_{t,l}`：动态候选专家数。
- `C_t`：当前专家缓存集合。
- `P_t`：最终实际预取集合。

---

# 2. 背景与问题链

## 2.1 背景漏斗

1. **领域背景**：MoE 模型通过稀疏激活扩展参数容量，但专家权重的驻留和搬运成为有限 GPU 内存条件下的重要瓶颈。
2. **具体困难**：下一 token 激活哪些专家由路由器动态决定；若专家不在加速器缓存中，则可能触发按需加载。
3. **现有路线**：统计热度、跨 token/跨层相关性、请求级激活矩阵、时序神经预测器以及缓存/卸载系统。
4. **未解决问题**：单一统计或单一时序规律难以同时适配专家数、原生 K 和路由稳定性均不同的 MoE 架构；高 Recall 也不必然转化为较少传输，因为错误预取可能污染缓存。
5. **研究问题**：能否构建一个跨异构 MoE 模型共享的专家预测器，并将预测分数转化为受预算约束、可拒绝且缓存感知的预取决策？

## 2.2 论文核心挑战

### 挑战一：路由规律具有模型异质性

- Qwen3：128专家、Top-8，历史可预测性较强。
- DeepSeek-R1：256专家、Top-8，路由更加分散，预测与校准最困难。
- Llama4 Maverick：128专家、Top-1，扩大预算会迅速降低 Precision。

### 挑战二：排序正确不等于概率可信

Top-K 排名可用于 Recall，但动态预算需要概率质量具有可解释性。因此预测后必须独立校准，而不能把未经校准的 logit 直接解释成概率。

### 挑战三：预测准确不等于系统收益

固定 Top-K 无条件预取可能导致：重复驻留、错误搬运、提前淘汰和缓存污染。因此需要把“候选排序”和“是否实际搬运”分离。

## 2.3 Related Work 预定结构

> 引用必须后续通过原论文逐条核验，本节暂不虚构 BibTeX 条目。

1. **MoE inference and expert offloading**：解释有限显存下的专家驻留、按需加载和预取问题。
2. **Statistics-based expert prediction**：Heatmap、Popularity、Previous-route 等轻量方法；优点是低开销，限制是上下文适应性不足。
3. **Request-level activation prediction**：以 MoE-Infinity EAM/EAMC 为代表，利用请求级激活模式；限制是对局部跨 token 动态建模不足。
4. **Learning-based route forecasting**：GRU/TCN/Transformer 等历史编码器；需要公平控制输入、历史长度与参数量。
5. **Confidence-aware selective prediction and caching**：概率校准、拒绝预测、价值感知缓存；用于解释 RouteCast 后半部分的理论来源。

---

# 3. 论文贡献定义

建议最终贡献写成三项，而不是罗列所有模块：

1. **跨架构通用预测**：提出一个共享的跨 token 专家预测器，动态融合统计先验、专家转移、请求级上下文和多尺度时序证据，统一处理不同专家数与原生 K 的 MoE 架构。
2. **校准的预算与拒绝机制**：将专家分数校准为可用于累计质量决策的排序分布，在原生预算硬约束下选择动态 K，并允许低置信度时不预取。
3. **选择性缓存感知执行**：将候选预测与缓存驻留/淘汰决策解耦，在轨迹回放中显著减少固定 Top-K 的无效专家传输，并明确报告模型与容量依赖的失败边界。

---

# 4. 方法：RouteCast

## 4.1 任务定义

给定请求在 token `1:t` 的已观测专家路由历史，预测 token `t+1` 在同一 MoE 层 `l` 上的真实专家集合：

\[
\hat{\mathcal{Y}}_{t+1,l}
=f_\theta\!\left(\mathcal{R}_{1:t},l,\mathbf{c}_t\right).
\]

真实集合大小为 `K_native`。预测层面报告固定 Top-K 指标；决策层面允许动态候选数和实际预取数为0。

## 4.2 多源预测分支

对每个专家构造六类证据：

\[
\mathbf{s}_{t,l,e}=
[s^{pop},s^{1hop},s^{2hop},s^{prefill},s^{temp},s^{multi}]_{t,l,e}.
\]

- `popularity`：训练请求中的层级专家流行度。
- `one-step transition`：当前激活专家到下一 token 专家的条件转移。
- `two-hop transition`：隔一个 token 的跳跃转移。
- `prefill/request prior`：当前请求 Prefill 阶段的专家偏好。
- `temporal GRU`：逐专家编码最近 `H=16` 个 token 的二值激活历史。
- `multiscale history`：1/2/4/8/H窗口均值与最近激活位置。

## 4.3 上下文门控融合

每个分支首先形成最大概率、Top-1/Top-2 margin 与归一化熵摘要。门控网络根据六分支摘要和结构上下文输出权重：

\[
\mathbf{g}_{t,l}=\operatorname{softmax}
\left(f_g([\phi(\mathbf{s}_{t,l}),\mathbf{c}_{t,l}])\right).
\]

结构上下文包含归一化层位置、专家规模、原生稀疏率、请求熵和 token 位置。融合采用概率混合而非直接相加不同尺度的 logits：

\[
s_{t,l,e}=\log\sum_b
g^{(b)}_{t,l}\exp(\log p^{(b)}_{t,l,e})
+0.1\,r_{t,l,e},
\]

其中 `r` 为轻量 refinement correction。

## 4.4 训练目标

训练目标由四部分组成：

\[
\mathcal{L}=0.20\mathcal{L}_{BCE}
+0.25\mathcal{L}_{list}
+0.35\mathcal{L}_{rank}
+0.20\mathcal{L}_{budget}.
\]

- BCE：多标签专家激活监督。
- Listwise：真实激活专家上的归一化列表损失。
- Pairwise ranking：拉开正专家与难负专家。
- Multi-budget loss：在多个候选 K 下推动真实专家进入预测前缀。

采用 group-DRO 风格模型组权重缓解三模型样本量与难度差异。**待复核**：limit=1000 报告出现 `group_weights=NaN`，需在最终复现实验中修复数值稳定性。

## 4.5 温度校准

冻结预测器后，对每个模型在验证集学习温度 `T_M`：

\[
\tilde{s}_{t,l,e}=s_{t,l,e}/T_M.
\]

固定 Top-K 排名不随正温度改变，因此 Recall@K 保持不变。动态累计质量使用：

\[
p_{t,l,e}=\frac{\exp(\tilde{s}_{t,l,e})}
{\sum_j\exp(\tilde{s}_{t,l,j})}.
\]

## 4.6 Budget-preserving adaptive mass

给定 mass `m`，动态候选数为：

\[
K_{t,l}(m)=\min\left\{k:
\sum_{i=1}^{k}p_{t,l,(i)}\ge m\right\},
\]

并施加逐样本硬约束：

\[
1\le K_{t,l}\le K_{native}.
\]

验证集选择：

\[
m^*=\arg\min_m\overline{K}(m),
\quad
\text{s.t. }R_{val}(m)\ge \rho R_{val}(K_{native}),
\]

其中 `rho=0.98`。若不可满足，则 `m=1` 显式退化到完整原生预算。

## 4.7 置信度拒绝

候选专家只有在概率超过验证集选择的阈值 `tau` 时才进入预取集合：

\[
\mathcal{A}_{t,l}=\{e\in\operatorname{TopK}(p,K_{t,l}):p_{t,l,e}\ge\tau\}.
\]

阈值通过单位效用选择：

\[
U(\tau)=N_{useful}(\tau)-\lambda_w N_{wasted}(\tau),
\]

当前 `lambda_w=1`。阈值集合包含 `tau>1`，因此当全部预测预取均为负效用时可以完全拒绝预取。

## 4.8 缓存感知预取与淘汰

先移除已驻留专家：

\[
\mathcal{N}_{t,l}=\mathcal{A}_{t,l}\setminus\mathcal{C}_t.
\]

缓存未满时插入候选；缓存已满时，仅当候选价值超过最低价值驻留项与安全边际时淘汰：

\[
p_{t,l,e}>V_t(v)+\delta.
\]

缓存价值随时间衰减，并在新预测到来时刷新。当前 `decay=0.90`，`delta=0.01`。纯 LRU 对照只按最近访问时间淘汰，不使用预测分数。

---

# 5. 实验协议

## 5.1 数据与模型

| 模型 | 专家数 | 原生 K | 路由特点 | limit=100 测试 token-layer 样本 |
|---|---:|---:|---|---:|
| Qwen3-235B-A22B | 128 | 8 | Top-8，历史规律较强 | 203,510 |
| DeepSeek-R1 | 256 | 8 | 专家空间更大、最难预测 | 124,505 |
| Llama4 Maverick | 128 | 1 | Top-1，错误预取风险高 | 36,864 |

数据按 request 固定划分 train/validation/test，防止同一请求的 token 泄漏至不同集合。历史长度 `H=16`。开发与消融使用 limit=100；经 finite 检查和稳定 Group-DRO 处理的 limit=1000 全量结果作为主预测证据。

## 5.2 评价指标

### 预测层

- Recall@K
- Precision@K
- MRR
- NDCG@K
- 平均命中专家数
- BCE、Brier、ECE（校准）

### 决策与缓存层

- 平均候选 K
- Cache hit rate
- On-demand loads
- Prefetch loads
- Useful/wasted prefetches
- Prefetch precision
- Total transfers

### 明确区分

- `Precision@K`：专家排序的集合精度。
- `Prefetch precision`：实际执行的预取中最终被使用的比例。
- `test_adaptive Recall`：动态候选集合的覆盖率，不等于拒绝后实际预取覆盖率；后续应补充 `issued-prefetch recall/coverage`。

## 5.3 Baselines

现有：Paper Heatmap、Request-aware Heatmap、GRU-only、MoE-Infinity EAM、LRU、固定原生 Top-K。

待补强：同输入 TCN 或轻量 Transformer；Oracle upper bound；必要时复现更近期可获得源码的学习型专家预测方法。

---

# 6. 结果证据

## 6.1 主预测结果（limit=1000，seed=2026）

| 模型 | Paper Heatmap | Request-aware Heatmap | GRU-only | RouteCast V3 |
|---|---:|---:|---:|---:|
| Qwen3, Recall@8 | 0.435187 | 0.433020 | 0.526822 | **0.555837** |
| DeepSeek-R1, Recall@8 | 0.213133 | 0.212111 | 0.270504 | **0.317228** |
| Llama4, Recall@1 | 0.329761 | 0.331059 | 0.293755 | **0.360242** |

相对当前最强神经 baseline GRU-only 的 Recall 绝对提升为 Qwen `+0.029015`、DeepSeek `+0.046723`、Llama `+0.066487`。以 request 为重采样单位的 10,000 次 paired bootstrap 95% CI 分别为 `[+0.027485,+0.030514]`、`[+0.042063,+0.051749]`和 `[+0.063104,+0.069711]`，三者的提升概率均为 1.0。

## 6.2 随机种子复现

| 模型 | Seeds | Mean Recall | Sample std | Paper Heatmap | Mean delta | 全种子超过 |
|---|---|---:|---:|---:|---:|---|
| Qwen3 | 2024/2025/2026 | **0.558412** | 0.002246 | 0.461987 | +0.096425 | Yes |
| DeepSeek-R1 | 2024/2025/2026 | **0.290884** | 0.005325 | 0.187861 | +0.103023 | Yes |
| Llama4 | 2024/2025/2026 | **0.328125** | 0.002363 | 0.313287 | +0.014838 | Yes |

注意：当前多种子汇总主要与 Paper Heatmap 比较；seed=2026 的 limit=1000 主结果已完成与 GRU-only 的 request-level paired bootstrap。后续仍需对最终算法补齐 limit=1000 多随机种子。

## 6.3 模块消融

| 模型 | Full | No history | Delta | No two-hop | Delta | No prefill | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 | **0.555820** | 0.541027 | −0.014793 | 0.546326 | −0.009494 | 0.551244 | −0.004577 |
| DeepSeek-R1 | 0.287609 | 0.265053 | −0.022556 | **0.294057** | +0.006449 | **0.294554** | +0.006946 |
| Llama4 | 0.326118 | 0.320475 | −0.005642 | **0.327148** | +0.001031 | 0.325792 | −0.000326 |

解释边界：history 对三模型均有正贡献；two-hop 和 prefill 并非普遍有效，在 DeepSeek/Llama 上移除后可能改善。该结果支持“模型异质性”，但也削弱“每个分支都必要”的主张。论文必须如实报告，并考虑在最终模型中使用模型/上下文门控自动抑制无效分支，而不是声称全模块普遍增益。

## 6.4 数据规模结果（limit=1000，暂定）

| 模型 | RouteCast native Recall | Paper Heatmap | Delta |
|---|---:|---:|---:|
| Qwen3 | 0.534394 | 0.435187 | +0.099207 |
| DeepSeek-R1 | 0.276585 | 0.213133 | +0.063452 |
| Llama4 | 0.337137 | 0.329761 | +0.007376 |

**审计标记**：训练报告中 `group_weights` 为 `NaN`。在修复数值稳定性并复现前，本表只能作为趋势证据，不能进入最终主表或摘要。

## 6.5 校准结果

| 模型 | T | Test BCE before→after | Brier before→after | ECE before→after | Recall变化 |
|---|---:|---|---|---|---|
| Qwen3 | 0.8177 | 0.459615→0.458133 | 0.154535→0.153519 | 0.289428→0.280106 | 不变 |
| DeepSeek-R1 | 0.3191 | 0.549593→0.459981 | 0.181861→0.144124 | 0.382736→0.276319 | 不变 |
| Llama4 | 1.0538 | 0.196067→0.195912 | 0.066288→0.066167 | 0.123607→0.124990 | 不变 |

解释：DeepSeek 原始 logits 明显偏平，低温度将其放大；但校准后 ECE 仍偏高。Llama 测试 ECE略恶化，因此不能声称所有模型的所有校准指标都改善。

## 6.6 最终预算与拒绝参数

| 模型 | Selected mass | Abstain threshold | 平均 K | 动态候选 Recall | 固定 Recall |
|---|---:|---:|---:|---:|---:|
| Qwen3 | 0.80 | 0.05 | 7.866 | 0.548385 | 0.555822 |
| DeepSeek-R1 | 1.00 | 0.15 | 8.000 | 0.287609 | 0.287609 |
| Llama4 | 0.01 | 0.50 | 1.000 | 0.326118 | 0.326118 |

机制解释：Qwen 可略微缩减候选预算；DeepSeek 自动退化至完整 Top-8；Llama 无法缩减 K，但通过高阈值拒绝绝大多数低置信度预取。

## 6.7 最终缓存回放：完整策略 vs 纯 LRU

| 模型 | 容量 | LRU hit | RouteCast hit | Hit delta | LRU transfers | RouteCast transfers | Transfer delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 | 1× | 0.375097 | **0.504341** | +0.129244 | 1,017,392 | **842,279** | −17.21% |
| Qwen3 | 2× | 0.657425 | **0.697748** | +0.040323 | 557,739 | **498,291** | −10.66% |
| Qwen3 | 4× | 0.844794 | **0.859834** | +0.015040 | 252,687 | **231,524** | −8.37% |
| DeepSeek | 1× | 0.174686 | **0.221270** | +0.046584 | 822,046 | **782,459** | −4.82% |
| DeepSeek | 2× | **0.330946** | 0.306772 | −0.024174 | **666,405** | 694,581 | +4.23% |
| DeepSeek | 4× | **0.450116** | 0.424739 | −0.025377 | **547,706** | 575,808 | +5.13% |
| Llama4 | 1× | 0.284207 | **0.293484** | +0.009277 | **26,387** | 26,803 | +1.58% |
| Llama4 | 2× | 0.359565 | **0.409804** | +0.050239 | 23,609 | **22,404** | −5.10% |
| Llama4 | 4× | 0.480740 | **0.525608** | +0.044868 | 19,142 | **17,979** | −6.08% |

主要结论：Qwen 全容量、DeepSeek 1×、Llama 2×/4×形成 Pareto dominance；DeepSeek 2×/4×被 LRU 支配，Llama 1×为小幅命中收益换取小幅传输增加。

## 6.8 完整策略 vs 固定原生 Top-K

| 模型 | 1×传输降低 | 2×传输降低 | 4×传输降低 |
|---|---:|---:|---:|
| Qwen3 | 32.6% | 13.9% | 8.6% |
| DeepSeek-R1 | 41.3% | 34.6% | 27.2% |
| Llama4 | 39.9% | 31.2% | 26.3% |

拒绝机制还将1×缓存的预取次数从固定方案大幅压缩：Qwen `519,643→35,306`，DeepSeek `597,447→6,813`，Llama `19,697→758`。

---

# 7. 结果分配：主文、图注与补充材料

| 结果 | 功能分类 | 放置位置 | 理由 |
|---|---|---|---|
| 三模型强 baseline 主表 | core discovery | 主文 Table 1 | 建立核心预测优势 |
| 最终缓存 Pareto 对比 | core discovery + qualification | 主文 Fig. 4/Table 2 | 建立执行价值并展示失败边界 |
| 校准摘要 | necessary support | 主文一小表或 Fig. 3a | mass/拒绝机制的前提 |
| 动态 K—Recall 曲线 | necessary support | 主文 Fig. 3b | 解释预算选择 |
| 消融 | necessary support + qualification | 主文 Table 3 | history有效，two-hop/prefill有异质性 |
| 随机种子 | robustness | SI，主文一句 mean±std | 支撑稳定性 |
| 完整 reliability bins | robustness | SI | 避免主文过载 |
| 99点 mass 曲线 | robustness/design sensitivity | SI | 主文仅展示关键 Pareto 曲线 |
| limit=1000 | scale evidence | 修复后主文；当前仅内部记录 | 当前有 NaN 审计问题 |
| 每文件日志、缓存构建细节 | provenance detail | Methods/SI/代码仓库 | 可复现但不推进主结论 |
| DeepSeek 2×/4×失败 | qualification | 主文必须出现 | 改变“普遍优于LRU”的解释 |

最短主文证据链：

1. RouteCast 在三种异构 MoE 上超过强统计与神经 baseline。
2. 消融表明跨 token history 是稳定贡献，其他上下文分支呈模型依赖性。
3. 校准与预算机制在不超过原生 K 的前提下形成模型特定决策。
4. 选择性缓存调度相对固定 Top-K 显著减少传输，并在若干模型—容量组合下支配 LRU。
5. DeepSeek 大缓存条件构成明确失败边界，说明应采用验证集策略选择或保守回退。

---

# 8. 图表规划

## Figure 1 — RouteCast 方法框架（主文）

四个连续模块：

1. 多源跨 token 路由特征；
2. 动态门控预测器；
3. Temperature + adaptive mass + abstention；
4. Cache-aware admission/eviction。

必须明确区分 `predicted candidate set` 与 `issued prefetch set`。图中不得出现未经验证的 PCIe latency 数字。

## Figure 2 — 三模型预测主结果（主文）

- 分组点图或条形图；三模型分面。
- 方法：Paper Heatmap、Request-aware、GRU-only、MoE-Infinity EAM、RouteCast。
- 原生 K 各自标注；不要把 Llama Recall@1 与 Qwen Recall@8误写成同一任务难度。
- 最好加入三种 seed 的散点与 mean±std。

## Figure 3 — Calibration and adaptive budgeting（主文）

- Panel a：ECE/Brier before–after，突出 DeepSeek改善与 Llama边界。
- Panel b：mass/平均 K/Recall Pareto 曲线，三模型分面。
- Panel c：验证集选择的 `(mass,tau)` 及最终 issued prefetch coverage/precision（该 coverage 指标待补）。

## Figure 4 — Cache replay Pareto frontier（主文）

- x轴：Total transfers（越低越好）；y轴：Cache hit rate（越高越好）。
- 模型分面，容量用点型或颜色编码。
- 方法：LRU、fixed-native、adaptive-mass、full selective cache-aware。
- 标出 Pareto-dominated 点，尤其 DeepSeek 2×/4×。

## Figure 5 — Ablation and routing heterogeneity（主文或SI）

- 热图展示去除 history/two-hop/prefill 的 delta。
- 强调 history 跨模型稳定，其他分支模型依赖。

## Supplementary figures

- Reliability diagrams（三模型）。
- 完整 K-budget curves。
- 随机种子分布。
- limit=100 vs 1000（修复后）。
- 历史长度与开销曲线。
- 层级/请求长度分组。

## Tables

- Table 1：主 baseline 对比。
- Table 2：完整缓存结果与传输下降。
- Table 3：消融。
- Table S1：数据划分与模型结构。
- Table S2：参数量、推理时间、内存和调度开销。
- Table S3：完整 calibration metrics。
- Table S4：随机种子与 bootstrap CI。

---

# 9. 论文结构与段落地图

## 9.1 暂定标题

推荐：**RouteCast: Calibrated Cross-token Expert Forecasting with Selective Cache-aware Prefetching for Heterogeneous Mixture-of-Experts Models**

备选：

1. **RouteCast: Universal Expert-route Forecasting under Adaptive Prefetch Budgets**
2. **Calibrated Expert-route Forecasting for Selective MoE Prefetching**
3. **From Route Prediction to Selective Prefetching in Heterogeneous MoE Models**

## 9.2 Abstract 写作合同

1. 问题：稀疏 MoE 的专家路由难以提前获知，错误预取可能抵消预测收益。
2. 缺口：统计方法、单一时序模型与请求级 EAM难以跨异构路由结构统一工作，也通常没有把预测置信度转成可拒绝的缓存决策。
3. 方法：RouteCast 四层结构。
4. 核心证据：三模型上超过最强已有内部 baseline；选择性调度相对固定 Top-K 降低传输。
5. 边界：轨迹驱动缓存回放，不主张真实端到端延迟。

### Provisional English abstract（待强 baseline/统计检验后定稿）

Sparse mixture-of-experts models activate only a small subset of experts for each token, but the selected experts are not known until routing is performed, making speculative expert placement vulnerable to inaccurate and wasteful prefetches. We present RouteCast, a cross-token expert forecasting framework that combines statistical priors, expert transitions, request context and multiscale route histories through context-dependent gating. RouteCast calibrates its expert scores, adapts the candidate budget without exceeding each model's native routing width, and abstains from low-confidence prefetches before applying cache-aware admission and eviction. Across traces from Qwen3, DeepSeek-R1 and Llama4 Maverick, RouteCast improves native-budget recall over heatmap, request-aware, GRU and request-level activation-matrix baselines. In trace-driven cache replay, selective scheduling reduces total expert transfers relative to fixed native-Top-K prefetching across all evaluated models and capacities, while simultaneously improving cache hit rate over LRU in Qwen3, memory-constrained DeepSeek-R1, and moderate-to-large-cache Llama4 settings. These results establish calibrated selective route forecasting as a practical algorithmic interface between expert prediction and cache management, while leaving end-to-end deployment latency for future validation.

## 9.3 Introduction（4段）

1. MoE扩展与专家驻留问题。
2. 路由预测的机会与“错误预取”风险。
3. 现有统计、请求级、时序方法的能力边界；提出跨架构与预测—决策断层。
4. RouteCast研究问题、三项贡献和证据范围。

## 9.4 Method

1. Problem formulation。
2. Multi-source route representation。
3. Context-gated universal predictor。
4. Training objectives and group balancing。
5. Calibration and adaptive budgeting。
6. Selective cache-aware scheduling。
7. Complexity and implementation。

## 9.5 Experiments / Results

1. Setup, datasets, splits, metrics and baselines。
2. Cross-model forecasting effectiveness。
3. Robustness and ablation。
4. Calibration and adaptive budget behavior。
5. Selective cache replay and Pareto analysis。
6. Failure modes and overhead。

## 9.6 Discussion

1. 预测与执行解耦为何必要。
2. 三模型差异说明了什么。
3. DeepSeek 大缓存失败的可能原因：局部性强于预测价值替换、校准残差、跨请求异质性。
4. 轨迹回放与真实部署之间的边界。
5. 可推广方向：验证集策略选择、真实系统集成、更强通用预测器。

## 9.7 Conclusion

贡献一句 + 三模型预测证据 + 选择性传输证据 + trace-driven边界。

---

# 10. Claim–Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| RouteCast 跨三模型优于现有内部 baseline | limit=100 baseline主表 | Supported |
| history 是跨模型稳定贡献 | 三模型 no-history 均下降 | Supported |
| two-hop/prefill 普遍有效 | DeepSeek/Llama反例 | Rejected；不得使用 |
| Temperature 不改变 Recall | 三模型 before/after完全相同 | Supported |
| Calibration 全指标在三模型均改善 | Llama ECE略恶化 | Rejected；需有边界 |
| 动态预算不超过原生 K | Budget-mode硬约束及结果 | Supported |
| 选择性调度减少固定 Top-K 传输 | 三模型全容量均降低 | Supported |
| 完整方法全面优于 LRU | DeepSeek 2×/4×反例 | Rejected |
| 完整方法在特定模型—容量组合支配 LRU | 最终正确 LRU回放 | Supported |
| RouteCast 实现端到端加速 | 无真实部署 | Needs evidence / currently unsupported |
| limit=1000证明规模稳定 | 指标存在，但group weights NaN | Provisional; audit required |

---

# 11. A会标准补充清单（按顺序）

## P0：必须完成

- [ ] 修复并复现 limit=1000 的 group-DRO NaN。
- [ ] 补一个公平的 TCN 或轻量 Transformer baseline。
- [ ] 计算 MRR、NDCG@K，并统一所有预测指标定义。
- [ ] 增加 request-level paired bootstrap 95% CI。
- [ ] 报告完整模型与各 baseline 的参数量、推理时间、峰值内存。
- [ ] 增加拒绝后的 issued-prefetch coverage/recall，而不仅是候选 Recall。
- [ ] 用验证集选择 `LRU/RouteCast` 策略，或将 DeepSeek 2×/4×明确作为失败边界。

## P1：强烈建议

- [ ] 最终完整算法多随机种子，而不仅是预测器多种子。
- [ ] 请求长度、Prefill长度、层位置分组。
- [ ] history长度与准确率—开销曲线。
- [ ] mass、tau、decay、eviction margin 敏感性。
- [ ] Oracle预测/Oracle缓存上界。
- [ ] 复现开销与缓存模拟吞吐。

## P2：资源允许时

- [ ] 小型可部署 MoE真实系统验证。
- [ ] 真实专家大小与非均匀传输成本。
- [ ] CPU–GPU异步预取时间线和端到端 latency。

---

# 12. 复现与来源索引

## 核心模型

- `F:\MOEresearch\code\models\routecast_cross_token_limit100_v3.pt`
- `F:\MOEresearch\code\models\routecast_cross_token_limit100_v3.json`
- `F:\MOEresearch\code\models\routecast_cross_token_limit1000_v3.pt`（待NaN审计）
- `F:\MOEresearch\code\models\routecast_cross_token_limit1000_v3.json`（待NaN审计）

## 核心结果

- Baselines：`F:\MOEresearch\record\baselines\strong_cross_token_limit100\strong_baselines_limit100.json`
- MoE-Infinity EAM：`F:\MOEresearch\record\baselines\moe_infinity_eam_limit100.json`
- Ablation：`F:\MOEresearch\record\ablations\cross_token_limit100\cross_token_ablation_limit100.json`
- Seeds：`F:\MOEresearch\record\seed_repro\cross_token_limit100\cross_token_seed_summary.json`
- Calibration：`F:\MOEresearch\record\calibration\routecast_v3_limit100_calibration.json`
- Final adaptive cache：`F:\MOEresearch\record\adaptive_cache\routecast_v3_limit100_budget_abstain.json`

## 当前最终代码

- Predictor training：`F:\MOEresearch\code\train_universal_routecast_v3.py`
- Calibration：`F:\MOEresearch\code\calibrate_routecast_v3.py`
- Mass sweep：`F:\MOEresearch\code\sweep_dynamic_topk_v3.py`
- Final adaptive cache：`F:\MOEresearch\code\evaluate_adaptive_cache_v3.py`
- MoE-Infinity EAM：`F:\MOEresearch\code\evaluate_moe_infinity_eam.py`

---

# 13. 下一次更新规则

每次新增实验必须同时更新：

1. 第6节对应结果表；
2. 第7节证据分配；
3. 第10节 Claim–Evidence Map；
4. 第11节待办状态；
5. 第12节来源路径。

不得用终端截图作为唯一证据；所有最终结果必须保存为 JSON/CSV，并记录脚本、模型、数据划分和随机种子。

最终协议：`F:\MOEresearch\record\paper\protocol\final_protocol.json` 与 `final_protocol.md`。协议变更必须升级版本并重跑所有受影响方法。
