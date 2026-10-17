# HR 首槽前工程失败与交付

本轮状态为 **ENGINEERING_STOP_BEFORE_FORMAL_SLOT**。HR 未完成正式实验，没有可评分的 HR 最终模型，也没有科学通过/失败结论。不能将此错误解释为 HR 方法无效。

## 已执行与故障原因

运行源码 `18d0262f58cb91c750fdbab1e93d49a9ffe8fb55` 已先推送。存储/显存、Base3 和真实 VLM 准入均通过；真实准入包含一次临时 native 权重写入并恢复，参数数量不变。随后 `learn_HR_1` 在首个正式槽前退出：learner 的末尾局部字典变量 `diagnostic` 遮蔽了同名模块上下文管理器，前面的 `with diagnostic(model)` 因 Python 局部作用域规则触发 `UnboundLocalError`。

此前 CPU 检查覆盖目标、梯度、候选与回滚等组件，未发现 learner 整体作用域错误；真实准入也未覆盖该正式学习路径。这是本次执行器实现和验证遗漏，不是数据或科学门失败。

## 保留的成本和缺失

- 正式槽尝试 0/20，正式更新 0；账本 committed_updates=1 仅为准入临时写入。
- GPU 租约累计 73.24508452415466 秒，3 个租约均关闭；控制器和学习 worker 已退出。
- forward 204（其中 vision 23），backward 21（准入 1、学习前探针 20），generation 3（Base canary），candidate 0，Judge 0。
- 没有 active checkpoint、final native、HR 87 消费者结果或 CE/证据间隔/保护/T2G 的最终评估。未调用的 Judge 不计作失败调用；旧 522 消费者保持原样。
- 原激活时钟、24 小时 deadline 和 2 GPUh 等限额保留，未清零或重新安装。数值见 ENGINEERING_STOP_RECEIPT.json。

## 最小修复与验证

将末尾字典改名为 `margin_diagnostic`，保持计算目标、阈值、预算和所有科学门不变；增加 symtable 回归断言，检查 learner 的 diagnostic 必须引用模块上下文管理器。完整生产路径 CPU 检查再次 PASS，CUDA/真实模型/Judge 调用均为 0。修复只提交源码，没有覆盖本次部署，没有再次 GPU 准入或重跑。因此 CPU PASS 不代表修复后真实 GPU 全流程已验证。

## 停止与交付边界

冻结生命周期规定共享机械失败停止并保留成本，重放额度为 0；本轮又没有可恢复 checkpoint，因此未自动重试、重启、重置预算或开启下一轮。有限评分/发布 tail 已写入 TAIL_REVIEW_REQUIRED 后结束，四个评分 worker 因失败屏障停止。保留远程失败记录、账本和所需现场，未伪造 REPORT_COMPLETE、FINAL_CONSUMED 或科学完成状态。

公开交付包含此故障报告、匿名汇总回执、作用域修复及 CPU 检查。模型权重、原始数据、私有日志和运行现场不发布。交付完成仅表示故障材料与修复已发布，不表示 HR 实验完成。
