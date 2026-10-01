# TXT词面迁移诊断结果

完整2047暴露消费者完成来源与路由公式审计；CPU-only，无训练、生成、Judge或新增CAL。

DEV词面单独保留：20次出现，7个不同输入，7个输入/winner组合；来源{'native_only': 20}；同图0、跨图20；缓存Base反事实{'benefit': 20}。

REG词面单独保留：22次出现，11个不同输入，11个输入/winner组合；来源{'native_only': 22}；同图0、跨图22；缓存Base反事实{'benefit': 22}。

完整分母、missing和歧义计数见PROVENANCE_AGGREGATES.json。多prefix出现不是独立样本量；Base反事实不是新执行的REG NEG0；source group隔离不是患者独立。native/S_fit严格词面迁移不能代替真实verified医学scope标注，暴露面板PASS不能声称独立确认或临床可部署。下一步需要合法、带来源验证且角色隔离的同图/跨图scope标注；没有这类材料时保持小时监测，不空转GPU或追加无依据搜索。私有QA、逐题诊断及权重不发布。

共有34个missing出现完整保留。48专家的240支持对应190个不同归一问句，其中15问句由多专家共享；这说明词面相同不自动给出医学scope唯一性。42次词面单独保护（DEV20、REG22）均来自native原问句跨图复用，当前formal面板没有fit_only词面的可观测激活贡献；不能推出4个S_fit支持在其他输入上无效。

初始化失败在读取formal输入之前，已保存私有证据并修正只读来源绑定；预算/首次计时保留。下一阶段等待真正verified且角色隔离的scope材料，8个预先确定pilot每edit至少4同图、2跨图正例并附来源验证；16个旧CHECK提案仍UNVERIFIED。该等待不影响小时监测，见NEXT_STAGE_READINESS.json。
