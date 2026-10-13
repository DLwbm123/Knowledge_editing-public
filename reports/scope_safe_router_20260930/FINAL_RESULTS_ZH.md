# Scope-safe Router 最终结果

No DEV router candidate clear joint PASS。专家权重全程冻结，训练/backward/optimizer 为0。

|Candidate|联合判定|bank24 T1G正确/93|bank24新增T1G错误（门槛0）|single新增T1G错误|bank24 Pressure已知增益|精确差界|
|---|---|---|---|---|---|---|
|RCAP|FAIL|61/93|26|30|+11|[16.0, 24.0] pp|
|NEG0|FAIL|84/93|3|7|+11|[14.0, 24.0] pp|
|SAFE|FAIL|61/93|26|30|+12|[16.0, 26.0] pp|

三个候选的明确失败项均为T1G：误拒合法正例，single和各bank prefix都超过零新增错误门槛。Pressure负例救援有效，但不能抵消正例损伤，因此不准入REG。bank24参考R0的T1G为87/93，Pressure为29/50（missing=5）；新增错误按输入逐一比较，不能用两方法正确总数之差代替。T0、T2G、非空正式T2L和各prefix的old35相对损伤门槛均通过；空资格分母保留不可评估。

REG expert12仅使用U_bg时H_count=47，H_source=28，H_switch=26，排名{'H_count': 3, 'H_source': 2, 'H_switch': 4}。hazard预先冻结后才读取暴露damage；非空capture可以识别latent capture风险，不能证明它是可部署的损伤分类器。

完整机械验证见 ROUTER_MECHANICAL_TESTS.json；科学门槛及shared-key missing界见 JOINT_ROUTER_DECISION.json；按完整分母的主结果见 DEV_ROUTER_RESULTS_ZH.md。没有加权总分；路由救援不替代T0/T1G/T2G/Pressure/old35/T2L门槛。原冻结T1L资格为空，明确不可评估，不当作accuracy通过。

CAL_SCOPE_UNAVAILABLE：继承支持及CHECK为native图像改写，未提供>=4独立verified同图和>=2隔离verified跨图scope-positive。未复制formal面板或自动改写充作新cross-image独立样本；本轮参数不变。本轮联合门槛未通过，现有native/S_fit+U_bg及这三种冻结规则不足以支持本轮部署结论；这不证明所有latent方法均不可能刻画scope，优先获取合法scope校准标注。

本轮已收口，REG=NOT_ADMITTED；GPU驻留0.8210小时（含latent构造、机械检查及推理，不是训练时间）。新Judge 0项，累计7156，无重试，积压0。复用生成统计{'historical_R0_reuse': 2924, 'effective_route_reuse': 946, 'Base_OFF_reuse': 86, 'new_generation': 0}。本轮无adapter/optimizer或权重清理，历史资产不动。

DEV与old47为暴露面板；REG即便执行也为EXPOSED_REGRESSION，不是independent confirmation。不重新训练expert，不搜索阈值，不自动进入下一阶段。公开仅source/config/tests/脱敏聚合与限制；私有输入、raw responses、逐题评分、weights及凭证不公开。

已公开交付：[PR10](https://github.com/DLwbm123/Knowledge_editing-public/pull/10)；分支 `review/scope-safe-router-20260930`。源代码及13项必需报告已发布，远端提交及匿名报告访问已核验。
