# 方向保持：唯一八例完整消费者

可靠机制完成后执行用户已授权的八例160步及1158评测。复用本轮341前缀和一个Fisher，只新增939源提案；八例、seed、七源.25/.25/.5、两RMS、秩/7168参数、原C标定0.1C、90%完整纯提案真实收益、原步幅/trust/四比例与三修正不变。评价题不进训练，不进行146。

每个新步先按原FISHER_MIN算法求同before基线，包括原MULTI及根fallback。纯源收益为正时，已有基线仅在真实算子方向投影>=.9−1e-7时可保留；四原比例依次用本轮方向约束/Fisher固定目标纠偏，新候选必须通过原真实规则和方向，且比已合格基线Fisher至少低1e-4预算才替代。若基线不满足方向，首个真实合格新候选可采用；若两者均不可行，则显式记SKIPPED_ALIGNMENT_INFEASIBLE并保持before，不回退无方向保障的更新。纯收益非正仍沿用原FISHER_MIN处理，单独记录；方向恢复假设仅作用于正纯收益提案。两个RMS只随源梯度推进，诊断恢复RNG。

八冻结首状态从已记录ALIGNED首见证与满足方向的FISHER_MIN首见证中按同固定Fisher替代差额选取；不选owner、不重做首提案。先验state/RNG/两RMS/源序/seed/预算/锁和保存加载，再于每worker首步前真实复核before/pure/selected损失、GPU RNG、原finite规则；CPU还对精确候选算子方向验收。原机制CPU失败及单对照数值修复记录完整保留，后续权威为BASELINE_REPAIR_EXIT1=0和完整FULL_EXIT1。

本链含修复的总上界200000NNback、46000算子back、65000局部求解、1500000非生成F、1280源提案（341已执行+939新增），最多1158生成及1158新Judge（累计<=18499）。上述为该唯一完整消费者的有限追加工程上界，不增加科学步数；原48h截止不延长，所有失败/加载/活跃租约成本继承，设备0/1/2/3/4/7、实时45GiB/UUID、最多三模型、owned2GiB/free8GiB不变。

全部八终态冻结后forced939+natural219评分，历史C/M1/M3/M13/E/MULTI/FISHER_MIN按完整身份只读复用，0重打旧有效/永久缺失。新隔离四scorer与队列judge_aligned_astra_medium、epoch MEDTRACE_ALIGNED_FISHER_20261014_V1；一次尝试。必须全局1158覆盖、四worker完成、0缺失，逐编辑FIT/T2G保留M1、损伤<=M3及两项natural门槛全通过才称兼顾，平均值不能掩盖退化。

发生工程错误保留状态，不自动训练续跑/迁移；本轮最终消费者后仅释放本轮Fisher/probe/active/diagnostic/resume/ALIGNED候选，历史C/D不动。完整结果/成本/中文报告代理推送PR78并验证remoteSHA/匿名HTTP200。全DEV无独立确认，T2L两探针/Base正确支持0保持率NA。
