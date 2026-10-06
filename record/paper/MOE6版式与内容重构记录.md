# MOE6 版式与内容重构记录

## 诊断

- 第 2--4 页连续出现方法公式，读者在尚未形成整体设计图景前就进入符号细节。
- 第 6 页以及第 8--9 页图片过于集中，图前缺少明确的研究问题，图后解释又被后续浮动体切断。
- 三张预算曲线分别作为独立单栏图，形成连续图片队列，并将小节标题推迟到图后。
- 多个 `\FloatBarrier` 强制清空浮动体，放大了“公式区”和“图片区”的视觉割裂。
- MRR、NDCG、缓存计数和四项损失的完整展开式对复现有帮助，但不是理解主算法所必需的主线内容。

## 新的正文架构

1. Introduction
2. Related Work
3. RouteCast Design
   - Problem and Design Principles
   - Multi-Source Route Forecasting
   - Cross-Model Learning
   - Calibrated Candidate Selection
   - Selective Cache Admission
4. Experimental Design
   - Trace Data and Request-Disjoint Protocol
   - Comparators and Metrics
   - Development and Cache Replay
5. Results
   - RouteCast Improves Cross-Model Forecasting
   - Budget Expansion Reveals Model-Specific Cost
   - Component Utility Depends on the Routing Regime
   - Calibration Enables Adaptive Candidate Selection
   - Selective Scheduling Changes Cache-Replay Outcomes
6. Discussion
7. Conclusion
8. Additional Evidence

## 公式取舍

主稿的独立陈列公式由 24 组减少到 6 组。保留：

- 下一 token 专家预测任务；
- 条件概率融合；
- 多目标训练损失；
- 温度校准后的动态候选宽度；
- validation mass 选择；
- 缓存替换准入条件。

删除或改写为正文说明：

- Recall/Precision、MRR/NDCG 的教科书式定义；
- 六个分支的逐项计数展开；
- 四个损失项的完整求和表达式；
- Group-DRO 权重更新展开式；
- BCE temperature search 的完整展开；
- cache hit、transfer 和 prefetch precision 的计数公式；
- confidence threshold 与 cache value 的中间公式。

这些删减没有改变算法定义或实验结果，只减少了阻断阅读的派生细节。

## 图表重新布局

- 主基线图移到对应 Results 小节，而不再提前出现在实验设置末尾。
- 三张预算曲线合并为一个横向三面板 `figure*`，共享一个结论型图注。
- 消融图移动到消融结果段落之后。
- Calibration 和 cache Pareto 图均遵循“问题与结果在前，图在后”的顺序。
- 删除 Results 中间的大部分 `\FloatBarrier`，仅在 Discussion 前保留一道边界。
- 增加固定 float-page 顶部对齐与间距设置，避免纯图片页在垂直方向被拉散。

## 主文本压缩审计

- 原稿约 3,110 个英文词，重构后约 2,856 个英文词。
- 独立公式从 24 组降到 6 组。
- figure 环境从 10 个降到 8 个，三张预算图合并但原始证据仍全部保留。
- 主结果、bootstrap、消融、校准和缓存失败边界均保留在正文。
- 三种子结果仍作为稳定性证据，明确标注为 limit=100 development screening。
- 真实部署延迟和吞吐仍明确列为未验证边界。

## 文件

- 原稿备份：`routecast_draft_before_layout_restructure.tex`
- 当前主稿：`routecast_draft_v1.tex`
- 便于上传的新版本：`routecast_draft_v2_restructured.tex`

本地环境未安装 LaTeX 编译器。源文件已完成环境配对、标签唯一性和引用完整性检查；最终分页仍需在 Overleaf 编译后依据新 PDF 进行第二轮视觉校正。
