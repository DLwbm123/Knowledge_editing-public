# 执行与复核边界

冻结计算代码提交为 `0f8f942`；后续报告生成器只读取匿名数值，不参与 GPU 计算。分支叠加在已停止的 online 2×2 分支上。

运行入口为 `experiments/medtrace_minimal_mechanism_v1/minimal_probe.py`，数学自检为 `minimal_math.selfcheck()`。复用前序运行的 source/official_llava/cpu_gate/derived_inputs 和工具目录，不改变它们；新建独立 RUN_ROOT、RESOURCE_LEDGER、RUN_MANIFEST 和 STAGE_CAP。旧运行的 LAUNCH_ENV 仅用于环境与祖先路径，RUN_ROOT 改为新运行，DIAGNOSTIC_PARENT 指向旧 C 候选。工具按原目录组织，新增 minimal_probe.py/minimal_math.py；父进程与 GPU worker 通过中性入口启动。

依赖旧工具中的 scope.py、protection.py 及原运行的传递依赖；关键共享实现对应本仓库：

- source/task 定义与原 source loss：`experiments/medtrace_scope_cd160_v1/scope.py`。
- TT88：`experiments/medtrace_tt_only_budget_h_v1_20261005/structures.py`。
- runtime 加载及 ledger：`experiments/medtrace_tt_u_continuation_v1/common.py`。
- FP16 冻结输入与 FP32 执行：`experiments/medtrace_fp32_constrained_v1/constrained.py`。
- 旧 RAW 写入：`experiments/medtrace_tt_adam_direction_v1/direction.py`。
- 算子差分内积：`experiments/medtrace_tt_mechanism_v1/mechanism_math.py`。

必须提供授权范围内的固定 FROZEN_TASKS/FROZEN_ANCHORS、8 份保留的 C BASE/FINAL、原模型和图像；公开仓库不包含这些私有材料。本轮代码不能用替代题目或权重冒充相同实验。

公开结果可独立复核：对包含 COMPLETE.json 和 rows/ 的报告目录运行 `python3 experiments/medtrace_minimal_mechanism_v1/minimal_report.py <报告目录>`，只用标准库重建 SUMMARY.json 与 NUMERICAL_REPORT_ZH.md。报告同时保留 16 个 owner/state 单元，含重复噪声与更新前等价检查。无原始文本、图像、完整 logits、答案 token ID 或权重。

预计 864 次语言前向、144 反向、48 临时更新之外，另有 56 次固定视觉预处理；GPU 计时包含加载/预处理/等待。1 小时墙钟 watchdog 与 ledger cap 同时存在，错误不自动重试。
