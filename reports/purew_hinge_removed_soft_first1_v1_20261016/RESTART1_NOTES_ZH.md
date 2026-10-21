# HR 一次恢复授权与检查

用户明确允许修复后重启一次，覆盖本次首槽前工程失败的零重放限制，仅适用于 restart1；不修改科学协议。沿用原始激活 1792128293.5797498、deadline 1792214693.5797498、2 GPUh、20 正式槽、21 次含准入写入、300 backward、15000 forward、92 generation 和 87 新 Judge 等累计限额。首轮 73.24508452415466 GPU秒、21 backward、204 forward、3 generation 和一次准入写入继续计费。

正式学习路径 CPU 检查实际调用生产 worker.learner，使用小型张量模型，走完 20 槽和正间隔/零间隔两个终态分支；执行真实 probe、attempt、梯度诊断、checkpoint 校验和 final native 保存。模型加载、输入和准入环境为 CPU fixture，不声称真实 VLM 全流程通过。恢复控制器测试确认跳过已有成功准入、仅启动学习/评估/重载三阶段，并拒绝时钟变化。既有损失/梯度/回滚/预算/评分门检查继续执行。

在原运行目录独占创建 restart1 claim；检查旧进程退出、无正式槽/最终权重、账本与时钟、原准入和空的新 Judge 队列。旧失败日志、旧 worker 源码与 GPU_SOURCE_VERSION、启动标识及账本快照保留。旧 failures 标记移入 private/restart1，避免其继续阻断已明确授权的恢复；发生新故障仍立即停止，不自动再重启。原生产工具只更新作用域修复和独立恢复入口，科学协议不变。

四评分 worker 和有限发布 tail 复用原队列，使用独立本地 restart1 状态；不重试旧 Judge，不自动下一轮。最终报告须包含累计成本和此次恢复来源。源码先推送，再使用中性入口启动。

首次恢复控制器在导入阶段误解析同名 controller 模块，尚未启动学习阶段或建立新 GPU 租约即退出。已修正入口模块搜索顺序，并新增与部署相同的 source-first PYTHONPATH 独立进程导入检查。仅在原账本逐字段未变化、无 START_learn_HR_1_restart1、原控制器退出且错误精确匹配时，允许一次完成该 restart1 启动；使用独占 CONTROLLER_IMPORT_REPAIR 记录，保留首个入口日志和来源，不创建第二个实验恢复，不重复消费者。
