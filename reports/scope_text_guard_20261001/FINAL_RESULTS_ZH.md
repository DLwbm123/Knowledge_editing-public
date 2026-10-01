# Scope Text Guard 最终结果

Finite preregistered REG complete。专家权重全程冻结，训练/backward/optimizer 为0。

|DEV Candidate|DEV联合判定|bank24 Pressure已知增益|精确差界|已知route rescue出现次数|
|---|---|---|---|---|
|TXT|PASS|+11|[14.000, 24.000] pp|15|

TXT bank24 T1G新增错误精确界为[0, 0]（门槛0），single新增错误界为[0, 0]。完整各门槛见joint JSON。

REG expert12仅使用U_bg时H_count=47，H_source=28，H_switch=26，排名{'H_count': 3, 'H_source': 2, 'H_switch': 4}。hazard预先冻结后才读取暴露damage；非空capture可以识别latent capture风险，不能证明它是可部署的损伤分类器。

完整机械验证见 ROUTER_MECHANICAL_TESTS.json；科学门槛及shared-key missing界见 JOINT_ROUTER_DECISION.json；按完整分母的主结果见 DEV_ROUTER_RESULTS_ZH.md。没有加权总分；路由救援不替代T0/T1G/T2G/Pressure/old35/T2L门槛。原冻结T1L资格为空，明确不可评估，不当作accuracy通过。

CAL_SCOPE_UNAVAILABLE：继承支持及CHECK为native图像改写，未提供>=4独立verified同图和>=2隔离verified跨图scope-positive。未复制formal面板或自动改写充作新cross-image独立样本；本轮参数不变。若联合门槛未通过，本轮冻结规则与现有支持不足以支持部署结论，不证明所有latent或文本规则均无解，优先获取合法scope校准标注。

本轮已收口，REG=COMPLETE；GPU驻留0.9538小时（含latent构造、机械检查及推理，不是训练时间）。新Judge 316项，累计7472，无重试，积压0。复用生成统计{'historical_R0_reuse': 4321, 'effective_route_reuse': 671, 'Base_OFF_reuse': 91, 'new_generation': 0}。本轮无adapter/optimizer或权重清理，历史资产不动。

DEV与old47为暴露面板；REG即便执行也为EXPOSED_REGRESSION，不是independent confirmation。不重新训练expert，不搜索阈值；用户已另行授权逐轮审阅并设计后续阶段，每一阶段另建计划、预算与冻结锁。公开仅source/config/tests/脱敏聚合与限制；私有输入、raw responses、逐题评分、weights及凭证不公开。

REG 按同一冻结门槛复核为 PASS：bank24 Pressure 已知正确从43/60到54/60，增益+11，shared-key missing精确差界[18.333,18.333] pp（共有missing变量相消）；全部T0/T1G/T2G/合格T2L与old35各prefix新增错误为0。完整资格分母与界见 REG_JOINT_REVIEW.json；这仍是暴露回归，不能声称独立或临床确认。

技术恢复已完成：old47 order=0历史绑定优先采用明确REG cohort，989 DEV及1058 REG旧绑定全通过；原失败证据、已完成结果和全部消耗保留。健康worker自然退出后释放剩余任务，未干预其他进程，未重复Judge payload。见RUNTIME_RECOVERY_AUDIT.json。
