# MOE7 版面验收与二次平衡

## MOE7 相对 MOE6 的改善

- 总页数由 10 页降至 8 页。
- 主方法独立公式由 24 组降至 6 组，连续公式墙已经消失。
- 方法、实验设置和结果之间的章节过渡更清楚。
- 三张预算曲线已经形成统一证据组。

## MOE7 仍存在的问题

- 第 5 页主基线图与 bootstrap 图上下堆叠，右下形成大面积空白。
- 第 6 页预算图与消融图上下排列，仍接近纯图页。
- 第 7 页校准图和缓存图均为跨栏图，正文无法进入页面。
- 缓存结果的最后一段被浮动到第 5 页图下，视觉上容易被误解为 Fig. 3 的解释。
- MOE7 中主基线图仍是旧版本，没有显示新增的 TCN-only 柱；Overleaf 需要同步本地最新版 `Fig2a_main_grouped_comparison`。

## 二次修改

- 将主基线柱状图与 request-level bootstrap 森林图合并为一个跨栏证据块。
- 将预算曲线与模块消融合并为一个跨栏证据块。
- 将校准图重新绘制为单栏纵向两面板。
- 将缓存 Pareto 图重新绘制为单栏纵向三面板。
- 主文 figure 环境由 8 个降为 6 个，其中跨栏图由 5 个降为 3 个。
- 新单栏图通过 5 pt PDF 字号下限审计；校准图最小 6 pt，缓存图最小 5.5 pt。

## 新文件

- LaTeX：`routecast_draft_v3_balanced.tex`
- 校准单栏图：`figures/compact/Fig5_column_calibration_budget.*`
- 缓存单栏图：`figures/compact/Fig6_column_cache_pareto.*`

下一次 Overleaf 编译必须同时上传新版 LaTeX 和上述新图，并覆盖 `figures/split/Fig2a_main_grouped_comparison.*`，否则主图仍会缺少 TCN-only。
