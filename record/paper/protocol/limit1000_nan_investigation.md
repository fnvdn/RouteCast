# RouteCast limit=1000 NaN 调查与修复

日期：2026-08-24  
状态：稳定版全量复训进行中

## 1. 观察到的异常

旧文件 `F:\MOEresearch\code\models\routecast_cross_token_limit1000_v3.json` 中，验证集与测试集预测指标均为有限值，但最终三个 Group-DRO 权重均为 `NaN`。旧训练未保存逐 batch 日志，因此不能从现存证据精确反推出首次异常的 batch；能够确定的是，污染发生在最佳 checkpoint（epoch 1）保存之后、最终 Group-DRO 权重写入之前。

## 2. 数据排查

对 `cross_token_limit1000_h16_packed` 的 3000 个缓存文件、22,376,230 个样本进行了完整扫描：

- `static`、`context`、`history` 与 `labels` 未发现非有限值；
- 审计状态为 PASS；
- 详细机器记录：`limit1000_cache_numeric_audit.json`。

因此，现有证据不支持“缓存本身包含 NaN”的解释。

## 3. 根因与修复

旧训练在 FP16 autocast 下直接对复合 ranking/listwise 损失反向传播，没有 GradScaler，也没有 finite 检查。大规模训练中，一次 FP16 溢出即可污染模型或 group loss，随后 Group-DRO softmax 权重全部成为 NaN。

稳定版采取以下措施：

1. 所有损失及其归约强制使用 FP32；
2. CUDA 前向优先使用 BF16；不支持 BF16 时使用 FP16 + GradScaler；
3. 对输入、scores、raw loss、weighted loss、gradient norm 逐 batch 检查；
4. 梯度裁剪启用 `error_if_nonfinite=True`；
5. Group-DRO 更新改为 float64、减最大值的稳定 softmax；
6. 每轮打印 group weight sum，任何非有限权重立即终止并报告 epoch/file/batch/model。

## 4. 回归测试

limit=100 与 limit=1000 均完成三模型覆盖的短反向传播测试，每个模型 8 个 batch。全部 scores、loss 和梯度范数有限，两个规模的损失量级一致。记录：

- `limit100_training_stability.json`
- `limit1000_training_stability.json`

## 5. 闭环结果

稳定版 limit=1000 已完成三轮全量复训，未再出现非有限数值：

- 最终 Group-DRO 权重为 Qwen3 `0.284021`、DeepSeek-R1 `0.463299`、Llama-4 `0.252679`，和为 1；
- 最佳 checkpoint 为 epoch 3，验证集 macro native-K Recall 为 `0.408511`；
- native-K 测试 Recall 为 Qwen3 `0.555837`、DeepSeek-R1 `0.317228`、Llama-4 `0.360242`；
- checkpoint、JSON 和全量日志均已保存，JSON 全字段 finite 检查通过；
- 新结果正式取代旧 NaN 报告，旧报告仅保留为失效证据，不进入论文表格。

有效模型：`F:\MOEresearch\code\models\routecast_cross_token_limit1000_v3_stable.pt`。
