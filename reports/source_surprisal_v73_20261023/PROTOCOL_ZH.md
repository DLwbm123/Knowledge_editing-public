# V73：固定Base惊讶度加权源答案目标

V72预条件方向真实改变更新却丢失泛化；停止该候选，不搜阻尼。新独立假设：完整答案均匀CE可能把预算用于Base已会预测的套话，而未充分强调需学习的源信息。仅利用合法训练源答案，在未编辑Base上计算每个目标token的teacher-forced NLL，按同一答案内均值归一形成冻结非负权重。全部目标含EOS，原ignore_index与causal shift不变；零NLL允许零权重并披露数量，无平滑常数/裁剪/温度搜索。权重是预测困难度，不声称等价于临床信息量或事实词识别。

每个10源事实的原问题及两条既定训练改写各计算一次，共30额外Base前向、0反传，训练前冻结全部权重。每步仍为原问题与交替改写的两个加权CE等权平均，loss=sum(weight*NLL)/sum(weight)。原V69全10未训练初始化复用，全部7168核心可更新，恢复原欧氏方向/clip1/双组scalar RMS/canonicalize；层21、rank4、ALL_TOKENS、lr/beta/eps、160步和生成配置不变。无新增训练资料、评价答案、保护损失、路由或网格。

CPU检查causal mask、全1权重loss及gradient一致、权重detached/均值1、整体缩放不变、无效值拒绝；生产cache fixture PASS。真实Base每个绑定要求均匀token NLL均值与原模型loss在rtol1e-6 atol1e-7一致，此为新增目标重构检查；原全30绑定zero/nonzero/cache loss-gradient精确/teacherKV rtol1e-5 atol5e-4/RNG/save-load准入不变，任一失败STOP不放宽。

主对照V64 A/B、直接机制对照V69；全219消费者，K_SOURCE_SURPRISAL新臂目录沿用B_VIEW_CE。V62–V72三元组/完整scope四元组精确复用，剩余完整回答seed730023混盲冻结后映射。保留B35正确T2G、相对A净>=2且2个正编辑、T0/T1G/改写正确身份保持、locality总数不降且Base正确无新损失、源断言A子集、无新增cap；歧义阻断晋级。全为曝光DEV、独立确认0，不能声明临床安全。

1600更新、3400back预计3380、230生成预计221、40万语言前向；首次2小时墙钟/GPU进程上限与总11:59:35取早，独立守护提前300秒停，重试不重置。远端1.5GB、本地.25GB，窗口累计预留22GB、剩余8GB；删除不返还。原V69初始化成本保留，本次30Base权重提取前向单列。完成全部消费者与评分后明确清单退休10正式状态；小初始化、权重向量、原始输出/标签私有保留。唯一30分钟监测继续。
