# TXT margin失败机制诊断独立冻结计划

前轮预先固定三点margin验证已完成，REG +0.1在bank12/bank24 T1G新增错误各1，DEV通过。本轮在读取新机制输出前冻结；目的核对是否为同一输入重复prefix和具体拒绝机制，不再选阈值、不追加参数。

只读原TXT已完成25任务、冻结三点缓存反事实和transfer audit逐条来源诊断。完整989DEV+1058REG为底层分母；仅汇总相对tau0发生路由变化的两个固定点，保留所有变化、缺失和不同任务。每点按完整mode/prefix/edit/task/input_id强身份键匹配，必须精确重现前轮路由变化数与tau0key。不调用模型或Judge。

固定机制核对：+0.1的新增OFF必须是原R0内且无严格词面保护、0<=q<0.1；-0.1的新增ON必须是原R0内且无严格词面保护、-0.1<=q<0。按cached score分类known rescue/damage/both/missing，缺失不判断方向；按unique input、input/winner对和各prefix分别报告真实计数。正例任务已知damage按T0/T1G/T2G分开；来源origin、same_image与same_source均只读原transfer审计，不把source group或不同hash当患者独立。

不利用输出重新定义医学scope、生成CAL或调tau。若发现同一个输入多prefix重复，明确相关性；若机制是无词面保护的低margin正例受损，后续优先真实scope签核/角色证据。既有源QA的待审提案只能人工/合法来源验证，未知标签保持UNKNOWN。只有获得新合法scope证据或明确独立的未测科学机制时才冻结后续实际推理/训练阶段；不得把持续授权当作无限搜索许可。

首次墙钟1小时，单次CPU60秒，存储32MiB，GPU/Judge/训练/生成0，累计Judge7472及41永久missing不重置。中性入口、GPU不可见，/data/bmw存储；不改历史运行/旧PR。失败保留证据和原计时，不盲重启。完成后代理发布source/plan/selfcheck/匿名聚合/限制到新draft PR并attach、验证，更新ACTIVE_STAGE和小时监测继续推进真实scope材料和有依据后续，不自动暂停。
