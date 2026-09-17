# 实现与数据缺口

已实现真实 FACT/EXTRA continuation、独立 W0 克隆、相同 native-fit-U 序列、H/G 同额外槽位、逐项 loss/tokens/梯度/样本/更新日志、全绑定 teacher、恢复检查、两条真实来源 smoke 入口。复用原模型加载、初始化、CP/freeR4、KL、hook、router 与优化器。

CPU 已通过 13 项相关检查。真实 GPU 结果只以运行目录的 COMPLETE/SMOKE_RESULT/EXIT 回执为准；PID 和模型加载不算 smoke 通过。

当前 Q_H 历史候选 2，新环境已生成并评分的 Q_H 为 0，Q_HG_eval 为 0。两条复用同一个 H/G/U 图像，患者/研究关系 unknown，无新增临床签字。不能进行来源隔离 DEV16，也不能进行完整主比较。

完整正式 single/sequential 核心矩阵尚未在 Stage18 执行。新的 formal 调度器必须绑定合格共同队列、已评分 Base、独立 eval 面板及 Judge 预算后才可启用；smoke 入口硬限定两条，不可当正式入口绕过合同。两条旧样本不另跑所谓正式训练，不用旧146 NO_H补齐。

保留所有本次生成 W0、writer、optimizer、router，未删除旧权重。新来源的标注和合同建设见 DATA_SUPPORT_BUILD_PLAN.md。
