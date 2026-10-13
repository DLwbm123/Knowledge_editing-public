# 当前决策

主矩阵与holdout全部计算/评分结束。AHS提高压力及holdout保持，但REG T2G仍下降6.38个百分点，不满足预定泛化约束，不能宣布净改进；P5不准入，不追加seed、M3或大bank。holdout上AHS也未稳定优于AH，不据此重新挑选方法。

R1/Rneg缺合法CAL_SCOPE正例，未拟合，不声称无效。AK现有结果不支持作为改进保留；AH/AU仍有泛化代价，不直接替换E_orig/A0。完整结果见FINAL_RESULTS_ZH.md和HOLDOUT_RESULTS_ZH.md；未测prefill、缺校准曲线、holdout规模不足均明确保留为证据限制。

checkpoint依赖清理按CHECKPOINT_LIFECYCLE.json回执执行；最终范围与完成状态见DELIVERY_COVERAGE.json。未经新计划与授权，不继续扩展科学搜索。

最终收口：holdout已完成，110个无依赖文件已回收，192个计划要求的最终adapter保留；没有待运行或待评分任务。以已公开的缺项和数据限制结束本次有限计划，关闭小时监测。
