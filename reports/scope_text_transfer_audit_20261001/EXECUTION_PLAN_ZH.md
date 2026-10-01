# TXT词面迁移来源固定诊断计划

公开父提交 b6e5aec9474ecdf44803bb2d54b7397489ee9e40（PR13）。用户已授权逐轮审阅及下一阶段执行。本计划在读取本阶段诊断输出前冻结。

问题：TXT通过已有暴露DEV/REG，需分清词面保护的来源、跨图迁移及跨专家词面歧义。只审计已执行的R0/TXT完整2047消费者（DEV989、REG1058，含全部prefix与old47）；不搜索新路由、半径、阈值或规则。

固定定义：复用原NFKC/casefold/空白归一且保留标点，由48编辑native+4 S_fit减去全部U_bg问题重建词面集合，核对原冻结240合法支持。以原R0实际winner归因，不以gold编辑选择expert。词面来源分native_only、fit_only、native_and_fit、unmatched、R0_OFF。同图/跨图按实际winner native图像比较，source group相同单独保留且不等于患者独立。全48专家及当前bank的词面支持者数仅为歧义计数，不当scope真值。

用已执行TXT的q、text_guard及R0路由核验固定公式，不重新算latent或模型；q<0且词面匹配仍激活的LEXICAL_ONLY_RETAIN分层归因。Judge已存在的R0/TXT/Base键只读；未知一律missing，不请求新Judge、不重判41永久missing。R0->TXT已知正确/错误转移与词面单独保留的缓存Base反事实分别报告；REG没有执行NEG0，缓存Base反事实不能写成新增NEG0实验。完整分母、missing、不同输入和不同输入/winner数一起保留，prefix出现次数不当独立样本数。正式T1L/T2L资格沿用原Base masks，未合格locality只作暴露描述，不并入正式门槛。

单次CPU任务，GPU小时预算0、新Judge0、训练/生成0、时钟上限1小时、CPU任务上限60秒、存储硬上限128MiB、启动前free>=8GiB。禁止GPU5；本任务不占用任何GPU。无checkpoint或删除动作、旧资产只读。不产生新CAL、cross-image临床标注或独立确认。任务成功条件仅完整2047消费者一一绑定、路由公式一致、脱敏聚合完成；断言或预算失败保留证据，不盲重跑。

结束后公开源码、配置、自检、脱敏聚合和限制，新review分支/draft PR附着，代理与远端SHA/匿名验证同前。审阅后优先取得真实角色隔离verified scope标注；若下步依赖外部标注，保持小时监测ACTIVE但不空转GPU或循环模糊/半径/seed/layer/rank搜索。
