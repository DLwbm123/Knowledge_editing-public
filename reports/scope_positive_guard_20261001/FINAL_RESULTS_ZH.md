# Scope-positive Guard 最终结果

No DEV router candidate clear joint PASS。专家权重全程冻结，训练/backward/optimizer 为0。

|Candidate|联合判定|bank24 Pressure已知增益|精确差界|已知route rescue出现次数|
|---|---|---|---|---|
|PLOO|FAIL|+11|[14.000, 24.000] pp|15|

PLOO bank24 T1G新增错误精确界为[3, 3]（门槛0），single新增错误界为[7, 7]。完整各门槛见joint JSON。

REG expert12仅使用U_bg时H_count=47，H_source=28，H_switch=26，排名{'H_count': 3, 'H_source': 2, 'H_switch': 4}。hazard预先冻结后才读取暴露damage；非空capture可以识别latent capture风险，不能证明它是可部署的损伤分类器。

完整机械验证见 ROUTER_MECHANICAL_TESTS.json；科学门槛及shared-key missing界见 JOINT_ROUTER_DECISION.json；按完整分母的主结果见 DEV_ROUTER_RESULTS_ZH.md。没有加权总分；路由救援不替代T0/T1G/T2G/Pressure/old35/T2L门槛。原冻结T1L资格为空，明确不可评估，不当作accuracy通过。

CAL_SCOPE_UNAVAILABLE：继承支持及CHECK为native图像改写，未提供>=4独立verified同图和>=2隔离verified跨图scope-positive。未复制formal面板或自动改写充作新cross-image独立样本；本轮参数不变。本轮联合门槛失败，这一固定保护规则未能支持部署结论；不据此推断所有latent方法均不可能刻画scope，优先获取合法scope校准标注。

本轮已收口，REG=NOT_ADMITTED；GPU驻留0.3546小时（含latent构造、机械检查及推理，不是训练时间）。新Judge 0项，累计7156，无重试，积压0。复用生成统计{'historical_R0_reuse': 2403, 'effective_route_reuse': 501, 'Base_OFF_reuse': 63, 'new_generation': 0}。本轮无adapter/optimizer或权重清理，历史资产不动。

DEV与old47为暴露面板；REG即便执行也为EXPOSED_REGRESSION，不是independent confirmation。不重新训练expert，不搜索阈值；用户已另行授权逐轮审阅并设计后续阶段，每一阶段另建计划、预算与冻结锁。公开仅source/config/tests/脱敏聚合与限制；私有输入、raw responses、逐题评分、weights及凭证不公开。

经逐消费者审阅，989个PLOO结果与NEG0的route及Judge payload全部相同（0个改变），156个guard命中均没有新增接受。保住240个合法fit支持点不能推断formal正例迁移。T1G single新增7错、bank24新增3错，均无missing不确定性；其余非空joint门槛通过。native/S_fit闭球在正式数据上没有提供额外正例保护。hazard/CAL资产为PR10继承只读审计，非本轮新增独立数据。

下一阶段只做预先固定的暴露几何/来源诊断及私有标注待审清单，不以该诊断拟合新阈值，不将已有formal正例改成新校准数据。新Judge0，正式生成消费者2967项全部基于完整绑定复用；实际模型生成仅在机械检查中验证。
