# M3Bench待审提案独立准备计划

在读取本轮提案输出前冻结。前轮身份角色诊断2070条（SLAKE119、VQA-RAD1951）只有Base/pool/metadata登记；这不证明未执行或患者独立。本轮仅准备待审scope关系，不准入CAL、不执行GPU/Judge，不改前两轮结论或旧4同图+2源隔离跨图要求。

固定来源为前两轮只读SOURCE_QA_ROLE_AUDIT、FACT_CARDS与修复后的IDENTITY_ROLE_LEDGER。候选必须不含FROZEN_FIT_SOURCE、FORMAL_OR_RESERVED_SOURCE、CAL_CHECK_SUPPORT_RESERVED、PROTECTED_SPLIT，图像存在；VQA-RAD additionally evaluation=evaluated；SLAKE仅原train英文。允许Base/pool/metadata登记存在，但新角色严格为REVIEW_ONLY_EXPOSURE_UNVERIFIED，不称独立新验证。所有scope=UNKNOWN、reviewer为空。

8个pilot固定4/5/6/11/12/15/16/21。检索完全复用前轮固定bucket/canon/stopwords，不做模糊匹配、embedding/阈值/seed或输出搜索。每edit同问同答、同问异答、同属性内容词交集每桶最多4，按image_id/qid稳定排序；保留完整桶分母，不凑数、不用关系占位符自动标真值。选中图片相对于anchor图像ID应不同；身份病例独立仍UNKNOWN。图像不复制，不调用生成模型。

产物为私有BLINDED_REVIEW.csv/JSON，包含目标源QA、候选源QA、注释证据、源角色及reviewer/evidence/IN-OUT-UNKNOWN/仲裁字段；不含预测/分数/路由。与前轮20共享原图材料分开；不把条件性同图材料当满足来源独立CAL。只有合法真实人类/来源审核证据到达后才能形成下一独立预算计划。

单次CPU，首次墙钟1小时、任务60秒、输入64MiB、存储16MiB；GPU/Judge/训练/生成0。机械自检必须拒收每种保护角色，不把同问异答自动当OUT，不把占位模板自动当IN。原stage只读、无checkpoint/删除。完成后新draft PR代理公开源码/冻结计划/匿名汇总/限制，保留私有QA和完整账本；更新ACTIVE_STAGE及小时维护为WAITING_REAL_SCOPE_REVIEW，不能无限追加CPU/GPU检索。
