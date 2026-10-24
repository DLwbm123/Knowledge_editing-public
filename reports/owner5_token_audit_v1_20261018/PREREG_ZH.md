# 构造负例 NLL 最小补证

授权范围：同一构造负例，仅原生 Base 与按原 seed 重建并 canonicalize 的零专家 Base。沿用原 FP16 输入准备、FP32 decoder 与 teacher routing。精确核对旧零初始化状态，不恢复训练终点。

固定目标三个 token：708 / 28705 / 2。两次 decoder 前向，零训练、零生成、零 Judge、零扩样本。前向前检查四种候选 no/yes/No/Yes 在原 prompt 后均为单 token。每次保存三个预测位置的完整 logits、labels、mask、三个交叉熵及候选 log probability；loss 平均与模型 loss 误差须小于 2e-6。

报告 lowercase、capitalized 与大小写合并的 log probability margin；正值偏向否定。报告格式项损失占比，不解释为梯度贡献。比较两状态并与历史平均 NLL 5.913671016693115 对照；不因偏差调整实现、重试或扩大实验。

两个已删除训练终点及历史逐候选损失无法从现存聚合记录恢复。本次不创建模型 checkpoint，只保留小型 logits 与可复算记录。完成后分析并发布匿名指标与代码，不自动开下一轮。
