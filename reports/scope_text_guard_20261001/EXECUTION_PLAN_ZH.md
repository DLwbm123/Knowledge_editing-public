# Scope Text Guard 固定实验计划

基于PR12公开冻结c5a1e0ce9854ac9bdcd0f0330a0864985f1a739e，用户授权自主逐轮设计执行，GPU只6/7，禁止5。

PR11同图latent保护球没有正式路由效应；PR12诊断7个失效正例均跨图、超出球9.75–23.37倍。等待真实verified CAL期间，只检验一个无需新标签的固定词面控制TXT，不继续调球半径，不声称独立确认或新增clinical verification。

定义：每edit合法native+4 S_fit问句，经Unicode NFKC、casefold、空白折叠形成严格词面集合；保留标点，不做同义词/模糊匹配。排除任何与合法U_bg问句归一化后相同的词面，先冻结全部48编辑集合。实际输入仍由原R0最近winner和原radius决定，NEG0 tau0通过或实际问句属于该winner的无冲突集合才保留，否则OFF；不回退其他expert，不扩大R0，不使用formal问题/答案/得分构造集合。U_bg词面不会因guard新增激活。词面一致本身不是已验证医学scope关系，风险由完整joint gate暴露评估。

先P0 role审计、固定集合、合法支持与真实H/OFF token/RNG/hook/权重机械测试，后完整DEV：R0/NEG0/TXT各single三批、bank4/8/12/24、old47四prefix，共15任务。完成所有DEV再判定，只有唯一TXT明确联合PASS才REG R0+TXT共10任务。固定输入/权重/生成/评分协议，复用完整绑定旧H/Base输出，不重复已完成评分或41永久missing。

门槛同PR10/11：T0/T1G零新增错误，T2G最多1，bank24 Pressure已知+3且共享missing精确下界>=0，old35各prefix最多1，非空正式T2L最多1，至少一处已知改变路由救援；空资格不可评估，不用加权总分。FAIL/INCONCLUSIVE/NO_ROUTING_EFFECT时REG不启动、不在本轮改词面规则。

最多6墙钟小时、4累计GPU小时（含机械/失败/恢复）、500新Judge，继承累计7156不重置；连续3新transport故障停请求，每payload最多一次尝试。存储soft10GiB/hard20GiB、余量>=8GiB；零训练/adapter/checkpoint。有限controller每轮检查预算，包括评分等待阶段。大文件只/data/bmw，保护旧资产和无关进程。完成后公开source/config/tests/脱敏aggregate与限制、draft PR、remote SHA和匿名验证。真实CAL仍待带来源验证的同图/跨图标注，不将formal样本或自动改写充作独立新数据。
