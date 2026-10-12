# 训练加载前导入故障与零工作接管

2026-10-12 07:35 UTC 实况检查确认：原 controller EXIT=1，三个训练分区错误均为 `ModuleNotFoundError: No module named llava`，全部训练 forward/backward/update/generation 计数为零，未产生 M13 候选、恢复点、曲线、校准或输出。评分队列零 payload/consumer，四个有限评分器因 OWN_STAGE_FAILED 安全退出，无新增 Judge。Fisher 已实际完成 61 题/1514 预测位置/1952 方向且准入 PASS；不得重建或重训对照。

根因：train() 在 common.load() 设置官方 LLaVA 路径之前导入 llava_runtime.seed_everything；llava_runtime 的模块常量因此提前捕获默认 external/LLaVA-Med。Fisher 按另一顺序加载，未受影响。修复在 M13 入口导入共享代码前设置官方路径；同一入口覆盖 Fisher/训练/评估/路由。旧实验源码和结果不改动。

一次性 m13_recovery.check() 硬核对精确错误、每项零计数、原父/三个子 PID+start_ticks 已结束、四个租约已结束、无训练产物、无评估/冻结、无 Judge 请求及官方模块可解析。拒绝非该错误或任何部分训练；非通用训练恢复入口。接管先归档原 START/EXIT/失败/日志，再仅启动原未执行的三分区，接同一 finish()。八例/种子/160步、全部数学/损失/系数/路由/门槛不变。

三次零模型工作失败租约合计 3.048560 秒仍保留账本；加上本轮 Fisher 240.606990 秒，本次检查新增 243.655550 秒。完整运行预期正常 8 + 失败 3 = 11 租约，报告显式核对异常租约。评分器以同一四份配置、epoch、隔离与队列接管，只等待生成；不重发请求。每小时监测继续，实际模型加载和进度须后续实况确认。接管失败时保留现场，不盲目重启。
