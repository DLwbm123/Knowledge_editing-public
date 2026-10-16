# 最小目标形状对照：HM/HU

冻结首源、原seed、原五视图、20槽、原始FP16 W、FP32 master、FP64统计、Base16/removed4 Fisher、步长.01、裁剪5、proposal上限.05、48 dual迭代、8 line search、8候选、原Armijo规则。PM/PU/SH00使用已交付结果，不重跑或补Judge。

定义a*=max(p0(y),.95)，o*=log(a*)-log(1-a*)。HM为mean ReLU(o*-o)，HU再加原(1-a*)条件KL及权重1的prefix KL。使用稳定log masses；不使用概率floor。边界ReLU子梯度0。保留Bernoulli KL仅作诊断。该对照包含目标形状及其固有梯度尺度变化，不声称等梯度范数或官方KLOD复现。

主保护政策对照HU-HM；目标形状对照HM-PM、HU-PU；单源描述性交互(HU-HM)-(PU-PM)。同样报告T0/T1G、完整CE、10个严格正有限证据间隔、固定36保护零损伤、4题T2G及跨臂逐题转移。通过CE不替代其他AND门。无调参、扩种子、历史病例回放或评价集训练；无自动146。

新日志target_gap_view_min_weighted为五视图内min的加权均值；target_gap_global_min为真正全局min。alpha_pass_fraction仅指a>=.95，新增target_pass_fraction指a>=a*。旧日志只能纠正解释，缺失全局min保持NA。单源总体区间统一NA_SINGLE_SOURCE，保留固定面板共享缺分差值界，不做伪总体区间。

新增174消费者与261历史消费者合并为435，最多174次新Judge，按完整binding复用既有合法评分/永久缺分。所有实际轨迹终态、精确重载与租约关闭后才评分。

实现沿用已验证PM/PU链的独立快照，避免改动冻结历史执行器。科学结论仅限同一首源和固定预算。
