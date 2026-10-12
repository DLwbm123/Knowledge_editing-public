# 最小实现与入场验证

复用PR69/PR67正式FP32 runtime、原源数据/路由、M1规范与两标量RMS、实际算子步幅匹配、算子Fisher和资源/准入账本；仅新增guard_math.py、guard_probe.py和guard_report.py，无新依赖，不修改已结束实验。

guard_math将右规范候选尾张量固定，把原Fisher sketch精确收缩成1952×512根核心响应行矩阵；通过PSD谱与拉格朗日乘子固定二分求欧氏最近的预算可行根。这是限定尾空间内的最小实际算子纠偏，所有源提案仍更新7168参数；不是完整7168维全局投影。数学自检覆盖KKT、Fisher值与行矩阵一致、根核心到实际算子等距、源RMS状态；实际八owner 7168参数的根等距检查、旧M1曲线/校准和来源隔离PASS，0VLM/0Judge。

guard_probe的同状态M13规则使用未clip源梯度加保护梯度后joint clip；纯M1和源RMS预条件用source-only clip及相同历史，避免误把先clip再相加当原M13。节点检查在diagnostic_scope内恢复RNG/hook，之后恢复真实轨迹状态；不推进正式RMS时钟。节点teacher-forced保护变化保留在私有明细，公开仅匿名聚合。

前缀存有完整GUARD80/RMS/RNG/源顺序/曲线/预算/锁。阶段A通过后严格81–160续接是已登记但尚未实现的后续消费者，不能在未校验状态延续时直接重启controller；本轮半小时监测将推进它。阶段A失败时不启动完整评估。CPU通过不等于模型加载、完整机制门槛、泛化或保护成功。
