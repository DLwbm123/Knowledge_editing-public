# Pre-insertion hazard 审计

U_bg可在不看formal answer时发现该insertion的capture风险；这不是已经验证的correctness危险分类器。

REG expert12: H_count=47，H_source=28，H_switch=26；24次insertion中的三个排名分别为{'H_count': 3, 'H_source': 2, 'H_switch': 4}。同分共享排名，不以formal损伤排序。

主bank使用历史H；本审计只依赖相同R0 keys/radii，故与PR9 A0路由诊断权重无关。hazard及RCAP/prototype先冻结，随后才读暴露damage，未调任何规则。
