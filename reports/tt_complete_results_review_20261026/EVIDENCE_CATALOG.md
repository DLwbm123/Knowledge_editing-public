# 完整证据目录：TT实验、医学资格与历史前驱

本目录覆盖报告生成前已有Git跟踪的275个资料目录、2752个文件（含4个根目录文件）。248个目录有叙述入口，27个没有。目录数不是实验数；多份报告可能复用同一运行、输入或评分。

返回[主报告](TT_ALL_RESULTS_REPORT_ZH.md)。下列“原报告结论节选”是已有公开报告的有限摘录，保持当时语境，不自动当作当前状态、完整复核或新的实验判定。各轮具体指标、分母、缺失与成本请读链接全文。历史budget/monitor文字不构成当前执行指令。

## E001 · answer_writer_v65_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V65：回答位置写入未通过保护与配对泛化门槛](../answer_writer_v65_20261023/REPORT_ZH.md)。

原报告结论节选：状态 `ANSWER_POSITION_WRITER_NOT_PROMOTED`。C的T2G为35/40，与V64 B总数相同，但配对上丢失2道原正确、增加2道。相对原题A为丢失2、增加4，净增2；净数掩盖了损失。T2L从A/B的1/5降为0/5，损伤了唯一Base原正确T2L；出现1个新触顶消费者，虽然触顶总数仍为1。

原聚合文件状态字段：`ANSWER_POSITION_WRITER_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../answer_writer_v65_20261023/REPORT_ZH.md) · [RESULTS.json](../answer_writer_v65_20261023/RESULTS.json)

## E002 · autonomous_12h_summary_20261025

阶段综合/研究审阅，非独立实验计数。

入口：[过去12小时实验汇总：目标完整回答恢复，保护与确认门仍未通过](../autonomous_12h_summary_20261025/REPORT_ZH.md)。

原报告结论节选：统计窗口按用户离开时的完整12小时授权：**2026-10-24 23:37:29至2026-10-25 11:37:29，北京时间**。V107在窗口开始前已启动，此处报告其窗口内完成的终态结果；V108–V127在本窗口内执行。V111条件未触发，没有运行或打开新8组。最新状态和清理于窗口结束后再次核验。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../autonomous_12h_summary_20261025/REPORT_ZH.md) · [RESULTS.json](../autonomous_12h_summary_20261025/RESULTS.json)

## E003 · autonomy_12h_20261023

阶段综合/研究审阅，非独立实验计数。

入口：[12小时自主研究窗口结果汇总（截至2026-10-23 09:51北京时间）](../autonomy_12h_20261023/REPORT_ZH.md)。

原报告结论节选：**尚无版本通过冻结联合门槛。最好的主线泛化结果是V69的37/40，相对原TT单问法33/40增加4题、相对双问法35/40增加2题；但损失Base唯一原正确locality题，并新增2条参考未支持的源事实断言，因此不能晋级，也不能报告为整体方法成功。** V75为36/40，仍有同类保留问题；最新V78为34/40，没有解决问题。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../autonomy_12h_20261023/REPORT_ZH.md) · [RESULTS.json](../autonomy_12h_20261023/RESULTS.json)

## E004 · autonomy_20261022_0051

阶段综合/研究审阅，非独立实验计数。

入口：[过去12小时实验报告：V32–V49](../autonomy_20261022_0051/REPORT_LAST12H_ZH.md)。

原报告结论节选：整理时间：2026-10-22 09:52（北京时间）。回溯区间约为2026-10-21 21:52至2026-10-22 09:52。按区间内完成并交付的实验纳入V32–V49；V32在区间开始前已启动，其整轮成本不能冒充严格切片的12小时成本。本报告是截至整理时的交付快照；原自主窗口约00:51–12:51，尚未到期，未延长或重置时钟。

全部结果/审阅/分析及聚合入口：[REPORT_LAST12H_ZH.md](../autonomy_20261022_0051/REPORT_LAST12H_ZH.md)

## E005 · autonomy_20261024_001706

阶段综合/研究审阅，非独立实验计数。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[autonomy_20261024_001706](../autonomy_20261024_001706/)。

## E006 · bank_compare_v80_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V80：10编辑真实 bank 结果](../bank_compare_v80_20261023/BANK_REPORT_ZH.md)。

原报告结论节选：10编辑自然路由下，TT T2G为37/40，BalancEdit与同R0检索均为40/40；TT未建立性能优势。TT专家参数存储显著更小，但本轮端到端延迟没有优势证据。两种方法仍破坏Base唯一原正确局部性答案，原V69联合门槛FAIL保持。所有数据均为DEV。

原聚合文件状态字段：`ALL_BANK_CONSUMERS_SCORED`；`PASS`；`UNIFIED_FULL_TEXT_SCORED`。

全部结果/审阅/分析及聚合入口：[BANK_REPORT_ZH.md](../bank_compare_v80_20261023/BANK_REPORT_ZH.md) · [UNIFIED_REPORT_ZH.md](../bank_compare_v80_20261023/UNIFIED_REPORT_ZH.md) · [BANK_RESULTS.json](../bank_compare_v80_20261023/BANK_RESULTS.json) · [COMPLETION_AUDIT.json](../bank_compare_v80_20261023/COMPLETION_AUDIT.json) · [UNIFIED_RESULTS.json](../bank_compare_v80_20261023/UNIFIED_RESULTS.json)

## E007 · base_answer_replay_v130_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V130结果：挑战修复1条，已正确保护事实新增损伤1条，冻结门未通过](../base_answer_replay_v130_20261025/REPORT_ZH.md)。

原报告结论节选：V130两臂全部完成。A精确复现V126全部238个正式输入绑定、完整输出文本和tokens；B只替换预先冻结的80条保护监督序列，另外4条保留原短参考。B的挑战全文支持由48/56升至49/56，但known保护CORE由81/84降至80/84、FULL由80/84降至79/84。因此不晋级，也不把此固定配方扩到146条。

原聚合文件状态字段：`STAGE0_BINDING_AND_TOKEN_AUDIT_PASS_GPU_NOT_STARTED`；`COMPLETE_DEVELOPMENT_GATE_FAILED`；`CLOSE_FIXED_RECIPE_NO_LOCAL_VARIANTS_NO146_LAUNCH`。

全部结果/审阅/分析及聚合入口：[PHASE0_REPORT_ZH.md](../base_answer_replay_v130_20261025/PHASE0_REPORT_ZH.md) · [REPORT_ZH.md](../base_answer_replay_v130_20261025/REPORT_ZH.md) · [PHASE0_RESULTS.json](../base_answer_replay_v130_20261025/PHASE0_RESULTS.json) · [RESULTS.json](../base_answer_replay_v130_20261025/RESULTS.json)

## E008 · base_screen_v100_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V100：两个候选 Base 均未形成可迁移的纠错与保护任务集](../base_screen_v100_20261024/REPORT_ZH.md)。

原报告结论节选：固定对照已完成：旧 LLaVA-Med 90 个输出只读复用，Qwen2.5-VL-7B-Instruct 和 Lingshu-7B 各新增 90 个输出。**两个候选的较强纠错与保护组合均为 0，未达到预注册的至少 8 个及类别覆盖门槛，因此本轮不迁移、也不训练 TT/CR-TTI。** 不能据此说所有更强 Base 都无效，或 TT 方法失败。

原聚合文件状态字段：`NO_MIGRATION_ADMISSION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../base_screen_v100_20261024/REPORT_ZH.md) · [RESULTS.json](../base_screen_v100_20261024/RESULTS.json)

## E009 · base_two_protection_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[两个保护候选的当前 Base 单次核验](../base_two_protection_20261018/REPORT_ZH.md)。

原报告结论节选：两条预先选定的来源负例，在当前 M13P 未编辑 Base 配置下各生成一次，均明确回答否，与来源标注一致：2/2 正确、0 缺失、0 重试、0 训练。两条结果均正常 EOS 结束，未触及 1024-token 上限。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../base_two_protection_20261018/REPORT_ZH.md) · [RESULTS.json](../base_two_protection_20261018/RESULTS.json)

## E010 · canonedit_v92_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[CanonEdit V92：完整自由回答与严格审核结果](../canonedit_v92_20261023/REPORT_ZH.md)。

原报告结论节选：本轮没有真实增益。775个自由回答全部完成，严格T2G为CE25/32，RO0/32，CANON0/32，CANON_TT0/32。CANON−RO净0，CANON_TT−CE为0胜25负，CANON−CE同为0胜25负。Canonical筛选抑制了部分无约束RO退化，但没有形成正确的新事实写入，不满足任何主要继续门槛。

原聚合文件状态字段：`SCORED_ALL_775`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../canonedit_v92_20261023/REPORT_ZH.md) · [RESULTS.json](../canonedit_v92_20261023/RESULTS.json)

## E011 · canonical_solver_v94_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V94终态：局部求解修正没有产生有效非零写入](../canonical_solver_v94_20261023/REPORT_ZH.md)。

原报告结论节选：全部930正式消费者已评分，其中620从V93完整绑定继承、310为本轮新生成；另生成40条来源资格。314个唯一全文均与V93冻结审核完整绑定一致，无新增未审核文本，先冻结继承标签再解映射。8编辑全部保留，单研究助手按来源合同审核，非独立临床评审。

原聚合文件状态字段：`SCORED_ALL_930`。

全部结果/审阅/分析及聚合入口：[PRO_ANALYSIS_REVIEW_ZH.md](../canonical_solver_v94_20261023/PRO_ANALYSIS_REVIEW_ZH.md) · [REPORT_ZH.md](../canonical_solver_v94_20261023/REPORT_ZH.md) · [RESULTS.json](../canonical_solver_v94_20261023/RESULTS.json)

## E012 · capability_source_v102_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V102：来源准入失败，未启动模型](../capability_source_v102_20261024/REPORT_ZH.md)。

原报告结论节选：按前置提交 `de05243e` 的规则固定了 32 名新患者，排除 60 名历史候选；D/C 患者互斥，所有候选都来自官方 TRAIN。随后逐例阅读全部来源 finding 句子，发现两个结节阳性候选不能支持所需参考答案：一个结构化异常标签与否定句意相反，另一个含明确不确定表达。前者说明异常标签本身不能代替语义核查；后者说明原不确定词筛选存在遗漏。

原聚合文件状态字段：`SOURCE_ADMISSION_FAIL`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../capability_source_v102_20261024/REPORT_ZH.md) · [RESULTS.json](../capability_source_v102_20261024/RESULTS.json)

## E013 · capability_source_v103_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V103：有界来源修复完成，32 名患者队列已固定](../capability_source_v103_20261024/REPORT_ZH.md)。

原报告结论节选：在前置协议与源码提交 `4fd562d3` 后，排除此前全部 92 名候选患者，按既定规则生成唯一 48 人来源审阅池。没有加载模型、查看模型答案或追加池外候选。逐例阅读所有来源 finding 后，45/48 通过文字语义准入；3 例因全文存在矛盾，或结节样影的肺内定位不能由文字明确支持而否决。后一类属于证据不足，不宣称原始标注已被临床证伪。

原聚合文件状态字段：`SOURCE_TEXT_ADMISSION_PASS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../capability_source_v103_20261024/REPORT_ZH.md) · [RESULTS.json](../capability_source_v103_20261024/RESULTS.json)

## E014 · capability_trial_v104_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V104：原生生成完成，两模型均未通过能力准入](../capability_trial_v104_20261024/REPORT_ZH.md)。

原报告结论节选：**2026-10-24 21:06（北京时间）结束：两模型 D 面板均未达到冻结门槛，未选模型，C 面板保持封存。当前不具备按本方案启动 TT 纠错与保护实验的条件。** 这不是 TT 方法失败，也不证明模型在其他医学任务上无能力；结论仅针对本次冻结的两类别、两问法和确定性抽取的 D16 患者。

原聚合文件状态字段：`D_CAPABILITY_FAIL_C_SEALED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../capability_trial_v104_20261024/REPORT_ZH.md) · [RESULTS.json](../capability_trial_v104_20261024/RESULTS.json)

## E015 · complement_kl_v76_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V76：目标外条件分布约束未改善泛化或正确题保留](../complement_kl_v76_20261023/REPORT_ZH.md)。

原报告结论节选：`COMPLEMENT_KL_NOT_PROMOTED`。T0 10/10、T1G40/40、T2G22/40、改写39/40、T1L33/84、T2L0/5。相对B35丢13增0，相对V69 37丢15增0；丢失Base唯一正确T2L。源断言34为A子集、触顶0，也不能抵消这些失败，联合FAIL。

原聚合文件状态字段：`COMPLEMENT_KL_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../complement_kl_v76_20261023/REPORT_ZH.md) · [RESULTS.json](../complement_kl_v76_20261023/RESULTS.json)

## E016 · contrast_center_v105_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V105：中心化改变训练几何，但没有改善留出行为](../contrast_center_v105_20261024/REPORT_ZH.md)。

原报告结论节选：本轮按用户“基于负结果推进TT、不换Base、不筛数据”的要求完成了预登记最小机制验证。CENTERED_TT与匹配JOINT_CE均训练2/2正确、旧留出1/3正确，主要行为门失败。关闭这一固定配方，不追加中心系数、步数或seed搜索。V104结论及C封存不变。

原聚合文件状态字段：`DEVELOPMENT_GATE_FAILED`；`CLOSE_FIXED_CENTERING_RECIPE; PRIORITIZE_EXISTING_M13P_EVIDENCE; NO_NEW_GPU_JOB`。

全部结果/审阅/分析及聚合入口：[PRO_REVIEW_ZH.md](../contrast_center_v105_20261024/PRO_REVIEW_ZH.md) · [REPORT_ZH.md](../contrast_center_v105_20261024/REPORT_ZH.md) · [RESULTS.json](../contrast_center_v105_20261024/RESULTS.json)

## E017 · core_metric_v72_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V72：局部Gram方向没有改善联合目标](../core_metric_v72_20261023/REPORT_ZH.md)。

原报告结论节选：`CORE_METRIC_NOT_PROMOTED`。全10编辑、1600更新、219输出完成，30真实准入全部通过。T2G34/40，相对B35丢2增1，相对同初始化V69的37丢3增0；T0 10/10、T1G40/40、改写40/40、T1L33/84保持。T2L仍1/5，但丢失Base唯一正确题、另增1题，不能用总数相同称保护通过。

原聚合文件状态字段：`CORE_METRIC_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../core_metric_v72_20261023/REPORT_ZH.md) · [RESULTS.json](../core_metric_v72_20261023/RESULTS.json)

## E018 · current_experiments_20260907

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[GPT Pro review: MedTRACE execution-preservation long run](../current_experiments_20260907/medtrace_execution_preserving/GPT_PRO_REVIEW.md)。

原报告结论节选：Primary comparison: additional step 800, SAFETY_FIRST, viewed development panel.

全部结果/审阅/分析及聚合入口：[medtrace_execution_preserving/GPT_PRO_REVIEW.md](../current_experiments_20260907/medtrace_execution_preserving/GPT_PRO_REVIEW.md) · [medtrace_execution_preserving/GPU_THROUGHPUT_AND_BUDGET_REPORT.md](../current_experiments_20260907/medtrace_execution_preserving/GPU_THROUGHPUT_AND_BUDGET_REPORT.md)

## E019 · decomp_24h_20260927

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE 24h 阶段报告](../decomp_24h_20260927/final/24H_FINAL_REPORT_ZH.md)。

原报告结论节选：状态：阶段已收口。已完成 387/387 个已排入队列的任务组。

全部结果/审阅/分析及聚合入口：[final/24H_FINAL_REPORT_ZH.md](../decomp_24h_20260927/final/24H_FINAL_REPORT_ZH.md) · [final/CORE_MATRIX_RESULTS.json](../decomp_24h_20260927/final/CORE_MATRIX_RESULTS.json) · [fix_pr5/FIX_TEST_RESULTS.json](../decomp_24h_20260927/fix_pr5/FIX_TEST_RESULTS.json)

## E020 · decomp_appendix_20260929

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE E1+E2 DEV24 配对验证报告](../decomp_appendix_20260929/dev24/EXTENSION_DEV24_REPORT_ZH.md)。

原报告结论节选：DEV24 的 78/78 任务组已完成训练、single 与真实 bank12/24，评分已收敛。144 条 continuation 全部为 80 步，72 对分别共享重建 Direct/CP/Tucker W0。父 W0 已回收，本轮没有将父最终权重当作 W0。原始追加时钟及账本不重置。

全部结果/审阅/分析及聚合入口：[dev24/EXTENSION_DEV24_REPORT_ZH.md](../decomp_appendix_20260929/dev24/EXTENSION_DEV24_REPORT_ZH.md) · [e1/EXTENSION_E1_REPORT_ZH.md](../decomp_appendix_20260929/e1/EXTENSION_E1_REPORT_ZH.md) · [e2/EXTENSION_E2_REPORT_ZH.md](../decomp_appendix_20260929/e2/EXTENSION_E2_REPORT_ZH.md) · [dev24/CORE_MATRIX_RESULTS.json](../decomp_appendix_20260929/dev24/CORE_MATRIX_RESULTS.json) · [e1/CORE_MATRIX_RESULTS.json](../decomp_appendix_20260929/e1/CORE_MATRIX_RESULTS.json) · [e2/CORE_MATRIX_RESULTS.json](../decomp_appendix_20260929/e2/CORE_MATRIX_RESULTS.json)

## E021 · dpc_v87_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V87决策敏感内化完整实验结果](../dpc_v87_20261023/REPORT_ZH.md)。

原报告结论节选：DPC相对普通响应压缩只有1题净收益，未超过直接CE对照，不达到预登记继续门槛。

原聚合文件状态字段：`SCORED_ALL_620`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../dpc_v87_20261023/REPORT_ZH.md) · [RESULTS.json](../dpc_v87_20261023/RESULTS.json)

## E022 · edit_guarded_protection_20260930

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[实现审阅与恢复入口修复](../edit_guarded_protection_20260930/IMPLEMENTATION_REVIEW_ZH.md)。

原报告结论节选：本轮训练、CHECK 选择、DEV gate、条件 REG 与收口流程已实现并审阅。保持 PR7/PR8 只读；不改变训练梯度规则、数据、共享 rho 选择口径、Judge prompt/key 或科学停止条件。

原聚合文件状态字段：`CLOSED_NOT_ADMITTED`。

全部结果/审阅/分析及聚合入口：[IMPLEMENTATION_REVIEW_ZH.md](../edit_guarded_protection_20260930/IMPLEMENTATION_REVIEW_ZH.md) · [FINAL_EXECUTION_AUDIT.json](../edit_guarded_protection_20260930/FINAL_EXECUTION_AUDIT.json)

## E023 · evidence_scope_v50_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V50：作用范围与证据条件化编辑（已完成：本轮无赢家）](../evidence_scope_v50_20261022/REPORT_ZH.md)。

原报告结论节选：阶段1全部完成，868个臂×问题×上下文绑定检查通过。862个绑定精确复用既有答案及tokens评分；剩余6个绑定对应3个隐藏方法的新输出，已完整评分并在解码方法映射前冻结。评分为助手参考评价，不是独立医学专家复核。

原聚合文件状态字段：`COMPLETE_NO_WINNER`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../evidence_scope_v50_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../evidence_scope_v50_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../evidence_scope_v50_20261022/RESULTS.json)

## E024 · external_comparison146_v131_20261025

混合阶段：M13P/Alpha与MEMIT single完成；MEMIT顺序未完。

入口：[M13P 原146完整评测：保持优势与泛化代价](../external_comparison146_v131_20261025/M13P_COMPLETE_REPORT_ZH.md)。

原报告结论节选：M13P的7,494个single、插入及前缀评测消费者已全部生成并获得接受评分，无缺失。结束的是M13P缺项补齐阶段；MEMIT统计与后续编辑仍在本轮原时钟内进行，不把此报告称为全部外部比较结束。

原聚合文件状态字段：`AUDIT_COMPLETE_COST_COUNTER_FIXED_NO_IDENTIFIED_OUTPUT_LOGIC_ERROR`；`M13P_ALL7494_ACCEPTED_ALPHA_ALL5722_ACCEPTED_MEMIT_PENDING`；`SINGLE146_STAGE_COMPLETE_SEQUENTIAL_NOT_COMPLETE`。

全部结果/审阅/分析及聚合入口：[CUMULATIVE_IMPLEMENTATION_AUDIT_REPORT_ZH.md](../external_comparison146_v131_20261025/CUMULATIVE_IMPLEMENTATION_AUDIT_REPORT_ZH.md) · [FINAL_REPORT_PREPARATION_ZH.md](../external_comparison146_v131_20261025/FINAL_REPORT_PREPARATION_ZH.md) · [M13P_COMPLETE_REPORT_ZH.md](../external_comparison146_v131_20261025/M13P_COMPLETE_REPORT_ZH.md) · [MEMIT_SINGLE146_STAGE_REPORT_ZH.md](../external_comparison146_v131_20261025/MEMIT_SINGLE146_STAGE_REPORT_ZH.md) · [CUMULATIVE_IMPLEMENTATION_AUDIT_RESULTS.json](../external_comparison146_v131_20261025/CUMULATIVE_IMPLEMENTATION_AUDIT_RESULTS.json) · [M13P_COMPLETE_RESULTS.json](../external_comparison146_v131_20261025/M13P_COMPLETE_RESULTS.json) · [MEMIT_SINGLE146_STAGE_RESULTS.json](../external_comparison146_v131_20261025/MEMIT_SINGLE146_STAGE_RESULTS.json)

## E025 · finite_protection_witness_v129_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V129终态：两种提案均有局部见证，后继机制门失败](../finite_protection_witness_v129_20261025/REPORT_ZH.md)。

原报告结论节选：全部3组原V126 160步轨迹复建、全部48个未提交有限提案已完成，worker与guard均EXIT0。当前V126期望提案和纯保护投影提案在3/3组均有符合固定数值条件的联合下降见证。因此“投影全通过且当前至少一个失败组全无见证”的后继条件为假；本轮不能作为启动有限步约束新训练方法的依据，不改变阈值或追加比例。

原聚合文件状态字段：`COMPLETE_LOCAL_WITNESSES_FOUND_CURRENT_AND_PROJECTED_FOLLOWUP_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../finite_protection_witness_v129_20261025/REPORT_ZH.md) · [RESULTS.json](../finite_protection_witness_v129_20261025/RESULTS.json)

## E026 · fixed_input_v77_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V77：固定输入、训练输出两核仍欠拟合](../fixed_input_v77_20261023/REPORT_ZH.md)。

原报告结论节选：`FIXED_INPUT_NOT_PROMOTED`。T0 6/10、T1G22/40、T2G15/40、改写23/40、T1L7/84、T2L2/5。相对B的T2G丢21增1，相对V69丢22增0。Base唯一正确题保留，T2L比V69增1，但不能抵消源任务和泛化大幅下降。源断言33相对A34仍新增身份1，新增触顶2；联合FAIL。

原聚合文件状态字段：`FIXED_INPUT_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../fixed_input_v77_20261023/REPORT_ZH.md) · [RESULTS.json](../fixed_input_v77_20261023/RESULTS.json)

## E027 · frozen_tail_v71_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V71：固定尾核在当前训练预算下严重欠拟合](../frozen_tail_v71_20261023/REPORT_ZH.md)。

原报告结论节选：`FIXED_TAIL_NOT_PROMOTED`。全部10编辑完整160步、219输出完成。T0仅1/10、T2G5/40；相对V69丢32个正确T2G、无新增正确题。固定尾核没有保留已有泛化收益。

原聚合文件状态字段：`FIXED_TAIL_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../frozen_tail_v71_20261023/REPORT_ZH.md) · [RESULTS.json](../frozen_tail_v71_20261023/RESULTS.json)

## E028 · full146_direction_freeze_20261025

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[full146_direction_freeze_20261025](../full146_direction_freeze_20261025/)。

## E029 · full_dev_pair_scope_v115_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V115：增加已曝光关系标签后仍有2/98个整组留一错误](../full_dev_pair_scope_v115_20261025/REPORT_ZH.md)。

原报告结论节选：按预登记将全14个已曝光DEV的70来源正例、28不同谓词负例用于14折整来源组留一，保持范数+abs坐标输入、λ=1、样本权重和符号阈值0。98点错2条，分别T06、T11；原8组共有58点仍1错，新增6组的40点另1错。来源组标签增加没有消除两个器官关系边界错误。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../full_dev_pair_scope_v115_20261025/REPORT_ZH.md) · [RESULTS.json](../full_dev_pair_scope_v115_20261025/RESULTS.json)

## E030 · full_gradient_init_v70_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V70：联合尾核初始化未保留泛化收益](../full_gradient_init_v70_20261023/REPORT_ZH.md)。

原报告结论节选：`FULL_GRADIENT_INITIALIZATION_NOT_PROMOTED`。全10编辑与219输出完成。T2G32/40，相对B丢3增0，相对V69丢5增0；T1L31/84，相对A/B/V69均丢2增0。T2L2/5且保留唯一Base原正确题，但不能抵消泛化和生成稳定性损失。

原聚合文件状态字段：`FULL_GRADIENT_INITIALIZATION_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../full_gradient_init_v70_20261023/REPORT_ZH.md) · [RESULTS.json](../full_gradient_init_v70_20261023/RESULTS.json)

## E031 · gradient_guard_v67_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V67：冲突投影未改变最终输出](../gradient_guard_v67_20261023/REPORT_ZH.md)。

原报告结论节选：`GRADIENT_PROJECTION_NOT_PROMOTED`。1600步中42步触发投影，全部位于编辑9；其余9个编辑无负点积。全部219条正式输出token与V64 B精确相同，因此T2G仍35/40、T0=10/10、T1G=40/40、T1L=33/84、T2L=1/5、改写40/40。不能声称没有发生干预，也不能声称投影改善了输出。

原聚合文件状态字段：`GRADIENT_PROJECTION_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../gradient_guard_v67_20261023/REPORT_ZH.md) · [RESULTS.json](../gradient_guard_v67_20261023/RESULTS.json)

## E032 · h_completion_v2_20261003

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[H-completion-v2 首轮数据构建报告](../h_completion_v2_20261003/H_COMPLETION_REPORT_ZH.md)。

原报告结论节选：原146个编辑全部保留。首轮获取、逐题审计和私有候选包已完成，**146/146合法H_fit目标尚未完成**。没有将新候选自动签为H，没有新增临床签核；PR20全部结果与证据只读保留。

全部结果/审阅/分析及聚合入口：[H_COMPLETION_REPORT_ZH.md](../h_completion_v2_20261003/H_COMPLETION_REPORT_ZH.md)

## E033 · independent_routes_v87_20261023

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[independent_routes_v87_20261023](../independent_routes_v87_20261023/)。

## E034 · input_signal_v101_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V101：未发现显示映射错误，缓存分数仅有局部排序信号](../input_signal_v101_20261024/REPORT_ZH.md)。

原报告结论节选：按用户要求，先完成并交付 V100，再完整阅读 Pro 分析，随后冻结并执行本次有限核验。**没有发现需要修正的 `/257` 灰阶映射；四类中三个正负配对的 Yes/No 首 token 分数方向正确，但一类反向，全部有图分数仍偏向 No。** 因此不修改旧图像、不重新运行旧实验，也不拟合阈值追分。

原聚合文件状态字段：`COMPLETE_NO_INPUT_CONVERSION_FIX_AND_MIXED_DIRECTION_SIGNAL`。

全部结果/审阅/分析及聚合入口：[PRO_REVIEW_ZH.md](../input_signal_v101_20261024/PRO_REVIEW_ZH.md) · [REPORT_ZH.md](../input_signal_v101_20261024/REPORT_ZH.md) · [RESULTS.json](../input_signal_v101_20261024/RESULTS.json)

## E035 · known_fact_replay_v120_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V120：路径约束保住新保护身份，但目标与训练拟合门失败](../known_fact_replay_v120_20261025/REPORT_ZH.md)。

原报告结论节选：全部28条轨迹按固定160步完成、4480次更新，全部终态保存加载通过；先冻结所有训练，再完成476次方法评价。新增Base56、零函数28及方法476共560次生成，559次EOS、1次触及1024上限，原始长回答和失败均保留。guard与worker正常退出，没有重选状态、重试或改变配方。

原聚合文件状态字段：`COMPLETE_BOTH_DEVELOPMENT_GATES_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../known_fact_replay_v120_20261025/REPORT_ZH.md) · [RESULTS.json](../known_fact_replay_v120_20261025/RESULTS.json)

## E036 · label_smooth_v78_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V78：固定标签平滑没有通过联合门槛](../label_smooth_v78_20261023/REPORT_ZH.md)。

原报告结论节选：`LABEL_SMOOTH_NOT_PROMOTED`。T0 10/10、T1G40/40、T2G34/40、改写40/40、T1L33/84、T2L1/5。相对B的T2G丢2增1，相对V69丢3增0。T2L总数虽为1，仍丢失Base唯一原正确题并新增另一题；源断言35相对A34新增1，触顶1且是新身份。联合FAIL，不搜索平滑系数或追加平滑版本。

原聚合文件状态字段：`LABEL_SMOOTH_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../label_smooth_v78_20261023/REPORT_ZH.md) · [RESULTS.json](../label_smooth_v78_20261023/RESULTS.json)

## E037 · local_boundary_v59_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V59：本地数据上的固定正负支持对照](../local_boundary_v59_20261022/REPORT_ZH.md)。

原报告结论节选：复用已在本地的S00图像/图注，训练C正例与A负例，B负例及D/E正例留出。SOURCE_ONLY重复正例CE，JOINT_CE平均正负CE；同160步、seed、7168参数和优化器，无hinge。全部12新资格题只评价，不训练；临床Base正确分母仅1（肾组织器官识别，原结构缩写保留），另有直接标注脚趾题单列。

原聚合文件状态字段：`DEVELOPMENT_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../local_boundary_v59_20261022/REPORT_ZH.md) · [RESULTS.json](../local_boundary_v59_20261022/RESULTS.json)

## E038 · local_boundary_v60_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V60：错误分析与本地新支持资格检查](../local_boundary_v60_20261022/REPORT_ZH.md)。

原报告结论节选：**结论：本轮数据门槛未通过，按用户要求收尾；未启动 Base 推理或固定训练对照。** 已保留 V59 的泛化失败，不增加 hinge，不调参，旧 B/D/E 及其同来源面板均未进入新训练名单。

原聚合文件状态字段：`CLOSED_DATA_INSUFFICIENT`。

全部结果/审阅/分析及聚合入口：[ERROR_ANALYSIS_ZH.md](../local_boundary_v60_20261022/ERROR_ANALYSIS_ZH.md) · [REPORT_ZH.md](../local_boundary_v60_20261022/REPORT_ZH.md) · [RESULTS.json](../local_boundary_v60_20261022/RESULTS.json)

## E039 · local_pool_v58_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V58 本地已有数据资格检查](../local_pool_v58_20261022/REPORT_ZH.md)。

原报告结论节选：只读取本地6987条元数据、图像、已有CSV图注及使用记录；0外部数据下载。先按固定元数据顺序冻结64个未在本地历史清单中匹配到的来源，再检查所有64幅图像；10个临床关系、2个标注解剖关系进入Base资格检查。初次元数据扫描误将排除索引当作使用记录，得到空名单；模型和像素检查前修复并保留错误记录。

原聚合文件状态字段：`MINIMAL_CLINICAL_RETENTION_AVAILABLE_WITH_ANNOTATION_LIMITATION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../local_pool_v58_20261022/REPORT_ZH.md) · [RESULTS.json](../local_pool_v58_20261022/RESULTS.json)

## E040 · m13p_confirmation_inventory_v106_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[M13P独立任务包核查：已完成固定事实核对和Base验证，尚无可启动确认包](../m13p_confirmation_inventory_v106_20261024/REPORT_ZH.md)。

原报告结论节选：本次没有停在文档讨论：核对了现有任务包、161份历史候选/消费清单和DataP现存任务表；从已有T4L表事前冻结8个图像组，阅读80条原始英文QA及8张图，7组目标/保护事实均有原始标注支持，随后实际生成并全文判读14条Base回答。固定任务中原存在性目标错误为0，因此未启动M13B/M13P训练。这是任务资格结果，不是M13P方法负结果。

原聚合文件状态字段：`NO_ADMITTED_M13P_CONFIRMATION_TASK_IN_FIXED_AUDIT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m13p_confirmation_inventory_v106_20261024/REPORT_ZH.md) · [RESULTS.json](../m13p_confirmation_inventory_v106_20261024/RESULTS.json)

## E041 · m13p_confirmation_prep_v107_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V107：P未获得独立小组确认支持，目标纠错相同而保护更弱](../m13p_confirmation_prep_v107_20261024/REPORT_ZH.md)。

原报告结论节选：预登记的14任务、56条完整160步轨迹和462次生成全部完成，无截断、无改题补位。方法身份隐藏后完整阅读448评价绑定去重得到的183条回答，冻结来源/核心/全文标签，再解盲。主分析6个CONFIRM来源组：B/P/S联合成功均5/6，P−B差值0、双侧完全枚举p=1，未通过“正差值且p≤0.05”标准。DEV8组联合成功分别8/8、6/8、7/8。没有证据支持P优于B，也没有P优于等范数缩小S的证据。

全部结果/审阅/分析及聚合入口：[PREPARATION_REPORT_ZH.md](../m13p_confirmation_prep_v107_20261024/PREPARATION_REPORT_ZH.md) · [REPORT_ZH.md](../m13p_confirmation_prep_v107_20261024/REPORT_ZH.md) · [RESULTS.json](../m13p_confirmation_prep_v107_20261024/RESULTS.json)

## E042 · m3bench_adaptive_v37_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V37完成：相对V36有增益，完整损伤门槛仍未通过](../m3bench_adaptive_v37_20261022/REPORT_ZH.md)。

原报告结论节选：两上下文共同正确由COVER的11/16增至ADAPT的12/16。原始条件12→13/16，SHORT11→12/16；相对COVER各修复1、旧正确损伤0。相对更早JOINT原始10→13无损伤，但SHORT8→12包含5修复、1旧正确损伤，因此预注册的双历史控制无损伤门槛false，不升级为通过全部条件的方法。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_adaptive_v37_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_adaptive_v37_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_adaptive_v37_20261022/RESULTS.json)

## E043 · m3bench_algorithm_v22_20261020

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[精简版Base审阅与接续回执](../m3bench_algorithm_v22_20261020/COMPACT_BASE_REVIEW_ZH.md)。

原报告结论节选：2078条当前精度OFF答案和查询键全部完成、完整覆盖。旧五进程按预登记控制性交接退出：各EXIT1及堆栈保留，不是训练/科学失败；五进程均0训练更新/0反向，新启动前已核实完整OFF与键及原精确身份退出。原阶段成本124716语言forward、2078生成、4156视觉调用、2078查询键，GPU进程秒以原五EXIT求和入总账，不重置。

原聚合文件状态字段：`COMPLETE_SCORED_DEVELOPMENT`；`COMPLETE_DEVELOPMENT_NOT_CLINICAL_CONFIRMATION`。

全部结果/审阅/分析及聚合入口：[COMPACT_BASE_REVIEW_ZH.md](../m3bench_algorithm_v22_20261020/COMPACT_BASE_REVIEW_ZH.md) · [MECHANISM_ANALYSIS_ZH.md](../m3bench_algorithm_v22_20261020/MECHANISM_ANALYSIS_ZH.md) · [SEVEN_ARM_REPORT_ZH.md](../m3bench_algorithm_v22_20261020/SEVEN_ARM_REPORT_ZH.md) · [SOURCE_ROUTE_REPORT_ZH.md](../m3bench_algorithm_v22_20261020/SOURCE_ROUTE_REPORT_ZH.md) · [SEVEN_ARM_RESULTS.json](../m3bench_algorithm_v22_20261020/SEVEN_ARM_RESULTS.json) · [SOURCE_ROUTE_RESULTS.json](../m3bench_algorithm_v22_20261020/SOURCE_ROUTE_RESULTS.json)

## E044 · m3bench_attribute_balance_v26_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V26：医学属性等权源监督未改善异问完整性](../m3bench_attribute_balance_v26_20261021/REPORT_ZH.md)。

原报告结论节选：**结论：M没有通过预登记改进条件。** 相同7168参数、B写入位置与160步下，医学属性单元等权使T2G由34/40降到32/40，0修复、2丢失；T1L逐消费者四分类不变，24个错误源答案复述完全保留。保留普通CE的B/S为稳定参照，不采用M为下一主配方，不围绕本次阴性扫描属性权重或增加rank。

原聚合文件状态字段：`COMPLETE_SCORED_DEVELOPMENT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_attribute_balance_v26_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_attribute_balance_v26_20261021/RESULTS.json)

## E045 · m3bench_brief_v35_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V35完成：简短回答指令未通过改善门槛](../m3bench_brief_v35_20261021/REPORT_ZH.md)。

原报告结论节选：全部62份原条件输出精确复现之后，双卡才执行62份统一简短指令输出。新增输出去重为61份，隐藏方法逐条参考审核、冻结C/I/U/A及理由后解码。原评分仅通过精确绑定复用，未改分。本轮不采用SHORT指令，不另搜一个提示词以追求正结果。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_brief_v35_20261021/REPORT_ZH.md) · [COMPLETION_AUDIT.json](../m3bench_brief_v35_20261021/COMPLETION_AUDIT.json) · [RESULTS.json](../m3bench_brief_v35_20261021/RESULTS.json)

## E046 · m3bench_content_v42_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V42完成：移除源EOS损失未改善](../m3bench_content_v42_20261022/REPORT_ZH.md)。

原报告结论节选：CONTENT共同正确11/16，低于ADAPT12。原始13/16相同、0修复0损伤；SHORT12→11，0修复1旧正确损伤。冻结gate false，不采用，也不扫描EOS权重。源两条件8/8，临床1/3（旧正确支持仅1），技术2/4均保持。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_content_v42_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_content_v42_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_content_v42_20261022/RESULTS.json)

## E047 · m3bench_context_v36_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V36完成：上下文共同正确增加，但未通过无损伤门槛](../m3bench_context_v36_20261022/REPORT_ZH.md)。

原报告结论节选：四源等权训练令16条留出中原始与SHORT同时正确数从7增至11。原始条件正确10→12，修复2、旧正确损伤0；SHORT正确8→11，修复5、旧正确损伤2。由于SHORT新增2条旧正确损伤，冻结的有限改善门槛为false，不升级默认配置，也不按此结果解锁跨数据集实验。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_context_v36_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_context_v36_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_context_v36_20261022/RESULTS.json)

## E048 · m3bench_contrast_v39_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V39完成：源答案错配对比未带来稳定改善](../m3bench_contrast_v39_20261022/REPORT_ZH.md)。

原报告结论节选：共同正确CONTRAST11/16，低于ADAPT12。原始13/16、SHORT12/16总数看似与ADAPT相同，但各自均1修复、1旧正确损伤，两条件交集反而减少。冻结门槛false，不采用，也不扫描对比系数。源两条件8/8，临床1/3支持仅1不变；原始技术2→1/4，SHORT仍2/4但1修复1损伤。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_contrast_v39_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_contrast_v39_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_contrast_v39_20261022/RESULTS.json)

## E049 · m3bench_cycle_v43_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V43完成：固定逐源更新未改善](../m3bench_cycle_v43_20261022/REPORT_ZH.md)。

原报告结论节选：CYCLE共同正确10/16，低于ADAPT12，也低于COVER11。对ADAPT，原始13→12、SHORT12→11，各0修复1旧正确损伤。对COVER原始1修复1损伤，SHORT无修复损伤。冻结gate false，不采用，不继续扫描源顺序或步长。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_cycle_v43_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_cycle_v43_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_cycle_v43_20261022/RESULTS.json)

## E050 · m3bench_discrimination_v34_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V34完成：候选偏好与自由回答存在落差](../m3bench_discrimination_v34_20261021/REPORT_ZH.md)。

原报告结论节选：固定JOINT的4例开发诊断已完成。源题全部正确，留出表达改善但仍有损伤；有限候选排名不能代替自由回答正确率，也不能证明事实已完整写入或生成路径是唯一原因。本轮不升级默认方法。

原聚合文件状态字段：`PASS`；`COMPLETE_FIXED_REFERENCE_DIAGNOSIS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_discrimination_v34_20261021/REPORT_ZH.md) · [COMPLETION_AUDIT.json](../m3bench_discrimination_v34_20261021/COMPLETION_AUDIT.json) · [RESULTS.json](../m3bench_discrimination_v34_20261021/RESULTS.json)

## E051 · m3bench_distcontrol_v47_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V47完成：ADAPT相对COVER的平均TV略高](../m3bench_distcontrol_v47_20261022/REPORT_ZH.md)。

原报告结论节选：三方同条件等病例TV：原始Base0.072356、COVER0.060078、ADAPT0.063453；SHORT分别0.066944、0.060951、0.062400。ADAPT−COVER主差值原始+0.003375、SHORT+0.001449。JS差值同为正（+0.000884/+0.000826）；argmax变化差原始+0.013059、SHORT−0.001105。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_distcontrol_v47_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_distcontrol_v47_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_distcontrol_v47_20261022/RESULTS.json)

## E052 · m3bench_distribution_v46_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V46完成：固定前缀后敏感性指标方向不一致](../m3bench_distribution_v46_20261022/REPORT_ZH.md)。

原报告结论节选：固定相同参考token前缀、相同图像置换后，等病例TV原始Base0.072356→ADAPT0.063453，差−0.008903；SHORT0.066944→0.062400，差−0.004544。31题中TV下降分别22/19题。主指标显示平均TV较低，但幅度有限。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_distribution_v46_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_distribution_v46_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_distribution_v46_20261022/RESULTS.json)

## E053 · m3bench_firstpos_v48_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V48完成：首位置与参考续写呈相反方向](../m3bench_firstpos_v48_20261022/REPORT_ZH.md)。

原报告结论节选：首个输出预测位置（无答案token前缀）的等病例TV：原始Base0.019796、COVER0.144837、ADAPT0.152716；SHORT分别0.008841、0.134037、0.135901。编辑后首位置TV上升，JS也上升；这与全参考位置平均TV略降的方向不同。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_firstpos_v48_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_firstpos_v48_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_firstpos_v48_20261022/RESULTS.json)

## E054 · m3bench_fixed_output_v33_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V33：固定输出去重与人工错误审计](../m3bench_fixed_output_v33_20261021/REPORT_ZH.md)。

原报告结论节选：V33完成，未重训、未生成新回答、未调用自动Judge、未改动任何历史C/I/U/A标签。错误分布不支持把所有失败归结为关系绑定；下一轮应固定JOINT，区分关系辨别、同关系事实取值及自由生成。

原聚合文件状态字段：`COMPLETE_MANUAL_FIXED_OUTPUT_AUDIT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_fixed_output_v33_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_fixed_output_v33_20261021/RESULTS.json)

## E055 · m3bench_influence_v49_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V49：探索性病例影响与措辞分析](../m3bench_influence_v49_20261022/REPORT_ZH.md)。

原报告结论节选：本分析在看到V48全体结果后登记，只读已有数据，不新增模型调用、生成或评分。不能充当独立验证。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_influence_v49_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_influence_v49_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_influence_v49_20261022/RESULTS.json)

## E056 · m3bench_new_cases_v32_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V32：新病例未复现表达覆盖增益，JOINT继续默认](../m3bench_new_cases_v32_20261021/REPORT_ZH.md)。

原报告结论节选：新六病例上，JOINT_EXPR没有新增正确回答，全体留出题比JOINT少一条正确答案。SOURCE与Base-C保留门通过，但HELDOUT零丢失和正增益门均不通过；不升级默认，不追加比例或相位扫描。

原聚合文件状态字段：`COMPLETE_FIXED_REFERENCE_REVIEW`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_new_cases_v32_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_new_cases_v32_20261021/RESULTS.json)

## E057 · m3bench_norm_v41_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V41完成：源梯度归一化未改善](../m3bench_norm_v41_20261022/REPORT_ZH.md)。

原报告结论节选：共同正确11/16，低于ADAPT12。原始13/16与ADAPT相同、零修复零损伤；SHORT12→11/16，0修复、1旧正确损伤。冻结门槛false，不采用此方向归一化。源两条件8/8、临床1/3不变（旧正确支持仅1）；技术两条件各2/4，SHORT另有1个不确定回答：轴面正确，但新增的病例部位未由盲评参考确认，不把它计为修复。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_norm_v41_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_norm_v41_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_norm_v41_20261022/RESULTS.json)

## E058 · m3bench_prefix_v40_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V40完成：较短源拟合前缀未改善留出结果](../m3bench_prefix_v40_20261022/REPORT_ZH.md)。

原报告结论节选：PREFIX与ADAPT共同正确同为12/16；原始13/16、SHORT12/16均与ADAPT一致，两个上下文各自零修复、零旧正确损伤。源两条件8/8，临床1/3（旧正确支持仅1），技术2/4。相对JOINT_SHORT仍存在1个留出旧正确损伤，完整冻结门槛false，不升级默认，也不扫描停止阈值。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_prefix_v40_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_prefix_v40_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_prefix_v40_20261022/RESULTS.json)

## E059 · m3bench_projection_v38_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V38完成：冲突投影未改善结果](../m3bench_projection_v38_20261022/REPORT_ZH.md)。

原报告结论节选：PROJECT两上下文共同正确11/16，低于ADAPT的12/16。原始条件均13/16；SHORT由12降至11/16，0修复、1条旧正确损伤。冻结改善门槛false，不采用投影，也不扫描投影顺序。源两条件8/8、临床1/3、技术2/4，与ADAPT相同，不能抵消留出损伤。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_projection_v38_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_projection_v38_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_projection_v38_20261022/RESULTS.json)

## E060 · m3bench_relation_expression_v29_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V29：增加固定训练表达覆盖取得有限留出收益](../m3bench_relation_expression_v29_20261021/REPORT_ZH.md)。

原报告结论节选：相同7,168参数、160步、每步两个关系损失下，JOINT_EXPR相对JOINT在24条预先固定的留出问题上从19条正确提高到23条正确，新增正确4条、丢失0条。结果支持有限表达覆盖的作用，不支持采用V28无增量收益的RELATION投影。

原聚合文件状态字段：`COMPLETE_FIXED_REFERENCE_REVIEW`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_relation_expression_v29_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_relation_expression_v29_20261021/RESULTS.json)

## E061 · m3bench_relation_order_v31_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V31：交换交替顺序未修复旧改写损失](../m3bench_relation_order_v31_20261021/REPORT_ZH.md)。

原报告结论节选：JOINT继续默认，FORWARD与REVERSE均不升级。固定80/80曝光下交换奇偶顺序，仅改变60份方法输出中的2份，未新增任何正确回答，也未丢失FORWARD的正确回答。V30相对JOINT的两条旧改写丢失都仍存在，否定“仅交换此相位足以修复已见回归”。

原聚合文件状态字段：`COMPLETE_FIXED_REFERENCE_REVIEW`；`JOINT_REMAINS_DEFAULT; phase swap does not repair two old-paraphrase losses`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_relation_order_v31_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_relation_order_v31_20261021/RESULTS.json)

## E062 · m3bench_relation_panel_v28_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V28：关系配对修复成立，冲突投影未带来增量收益](../m3bench_relation_panel_v28_20261021/REPORT_ZH.md)。

原报告结论节选：六个固定来源病例的两臂已完成；36个Base回答与72个方法回答齐全，两个阶段EXIT0且精确进程身份退出。三对合格源题上，JOINT与RELATION均修复3/3个原非正确事实，保留3/3个原正确事实，两条同时正确3/3。RELATION相对JOINT没有新增正确，也没有减少损伤；36个输入四分类逐题完全相同。因此本轮支持低参数联合关系监督能够完成这些源题配对，但不支持采用冲突投影作为额外有效组件。

原聚合文件状态字段：`COMPLETE_FIXED_REFERENCE_REVIEW`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_relation_panel_v28_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_relation_panel_v28_20261021/RESULTS.json)

## E063 · m3bench_relation_regression_v30_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V30：V29收益精确复现，但旧改写退化，JOINT_EXPR不升级默认基线](../m3bench_relation_regression_v30_20261021/REPORT_ZH.md)。

原报告结论节选：冻结推进门未通过。保留JOINT为当前开发默认基线，JOINT_EXPR仅作为有表达分布权衡的候选。原问题保持全部正确；V29复现全部通过；旧改写丢失两条JOINT原正确答案，均发生在预注册三源资格病例。不得用其他题的新增正确或三组总分抵销损失。

原聚合文件状态字段：`COMPLETE_FIXED_REFERENCE_REVIEW`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_relation_regression_v30_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_relation_regression_v30_20261021/RESULTS.json)

## E064 · m3bench_relation_selectivity_v27_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V27：关系级监督可用，但固定面板缺少修复—保留机会](../m3bench_relation_selectivity_v27_20261021/REPORT_ZH.md)。

原报告结论节选：已完成有限来源核查、CPU、真实模型机械检查和六条Base源问句评分。**STOP_NO_REPAIR_PRESERVE_OPPORTUNITY**：三个预定病例的待保留关系均为Base明确错误，没有一例同时具备“更新关系非正确、保留关系正确”。按运行前门槛结束，0正式训练、0方法输出。不能将此结论称为关系梯度方法的负结果，也不能以同时学会两条错误事实冒充保留旧知识。

原聚合文件状态字段：`STOP_NO_REPAIR_PRESERVE_OPPORTUNITY`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_relation_selectivity_v27_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_relation_selectivity_v27_20261021/RESULTS.json)

## E065 · m3bench_replicate_v44_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V44完成：第二初始化仍见一题配对增益，但绝对表现下降](../m3bench_replicate_v44_20261022/REPORT_ZH.md)。

原报告结论节选：冻结配对复验门槛通过：新种子COVER2→ADAPT2共同正确9→10/16；原始10→11（1修复0损伤），SHORT11→11（0修复0损伤）。每上下文SOURCE/HELDOUT/CLINICAL均无新增COVER2旧正确损伤。原种子COVER→ADAPT为11→12/16；两个种子均为一题增益，但绝对分数不同，不能跨seed归因于方法，也不筛选更好的seed。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_replicate_v44_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_replicate_v44_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_replicate_v44_20261022/RESULTS.json)

## E066 · m3bench_scope_audit_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定答案选择上限与边界支持审计](../m3bench_scope_audit_20261021/REPORT_ZH.md)。

原报告结论节选：本轮完成了用户要求的零模型决策审计。对齐 B（V24）、C160/C_STOP（V25）和 OFF 的全部 216 个编辑×查询消费者、180 个唯一输入，直接沿用冻结四分类；0训练、0生成、0新评分。逐消费者清单私有保存，未恢复任何权重、旧进程、V23 或其余136个编辑。

原聚合文件状态字段：`COMPLETE_ZERO_MODEL_FROZEN_LABEL_AUDIT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_scope_audit_20261021/REPORT_ZH.md) · [RESULTS.json](../m3bench_scope_audit_20261021/RESULTS.json)

## E067 · m3bench_visual_v45_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V45完成：换图后ADAPT更常维持完全相同输出](../m3bench_visual_v45_20261022/REPORT_ZH.md)。

原报告结论节选：固定异病例图像替换后，原始条件Base有9/31回答token完全不变，ADAPT为16/31；SHORT分别9/31与18/31。原始配对中9题Base变而ADAPT不变，2题相反；SHORT为11与2。留出16题原始不变数3→8，SHORT5→9。源8题原始4→6、SHORT4→7。全部组别、角色和病例见RESULTS。

原聚合文件状态字段：`PASS`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_visual_v45_20261022/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../m3bench_visual_v45_20261022/EXECUTION_AUDIT.json) · [RESULTS.json](../m3bench_visual_v45_20261022/RESULTS.json)

## E068 · m3bench_writer_alignment_v24_20261020

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[三臂写入位置与token范围：最终开发结果](../m3bench_writer_alignment_v24_20261020/REPORT_ZH.md)。

原报告结论节选：三臂各216回答、共同OFF180，648+180全部覆盖；五分片EXIT0且精确身份退出。30个训练臂各160步，4800更新完整。480个隐藏唯一回答全部有固定参考四分类及reason：401个精确复用V22冻结判断，79个由助手隐藏组别审阅（8C/44I/23U/4A）；没有自动Judge，不是临床独立审定。评分完成前未读取arm映射进行新判定。

原聚合文件状态字段：`COMPLETE_SCORED_DEVELOPMENT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_writer_alignment_v24_20261020/REPORT_ZH.md) · [RESULTS.json](../m3bench_writer_alignment_v24_20261020/RESULTS.json)

## E069 · m3bench_writer_optimization_v25_20261020

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V25：延长同一C轨迹修复了什么](../m3bench_writer_optimization_v25_20261020/REPORT_ZH.md)。

原报告结论节选：结论：追加训练在同一7168参数TT内修复了明显的训练预算不足，T2G由18/40升至29/40（+27.5个百分点，修复11、丢失0）；源拟合与生成稳定性均改善。但源答案传播没有减轻，T1L精确复述65→75；当前终态仍落后历史B，不能作为整体writer改进或容量上限证据。

原聚合文件状态字段：`COMPLETE_SCORED_DEVELOPMENT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../m3bench_writer_optimization_v25_20261020/REPORT_ZH.md) · [RESULTS.json](../m3bench_writer_optimization_v25_20261020/RESULTS.json)

## E070 · maximal_scope_v110_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V110：最大来源范围恢复改写，仍放行一个不同事实问题](../maximal_scope_v110_20261025/REPORT_ZH.md)。

原报告结论节选：闭式来源约束规则在d0909813冻结后执行：r=min(18个校准负例距离)−10^-6=0.7942808468720327，包含全部40个来源留一正例，并排除全部18个校准负例。求r只读取原校准回执，先固定r再读缓存查询距离；没有扫描半径或读取回答选择阈值。原V109中点结果未改。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../maximal_scope_v110_20261025/REPORT_ZH.md) · [RESULTS.json](../maximal_scope_v110_20261025/RESULTS.json)

## E071 · mechanism_v51_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V51：换基诊断、独立选层与医学边界支持准入](../mechanism_v51_20261022/REPORT_ZH.md)。

原报告结论节选：新授权窗口为2026-10-22 12:16:06至10月23日00:16:06（北京时间），新增存储上限30,000,000,000字节，初批预留远程2GB、本地500MB。唯一owner5仍为每30分钟，旧窗口成本与V50阴性结论保留，不重启旧终态。

原聚合文件状态字段：`BOUNDED_REPAIR_REVIEW_COMPLETE`；`COMPLETE`；`NO_PARETO_IMPROVEMENT`。

全部结果/审阅/分析及聚合入口：[BOUNDARY_REPAIR_REPORT_ZH.md](../mechanism_v51_20261022/BOUNDARY_REPAIR_REPORT_ZH.md) · [REPORT_ZH.md](../mechanism_v51_20261022/REPORT_ZH.md) · [BOUNDARY_REPAIR_RESULTS.json](../mechanism_v51_20261022/BOUNDARY_REPAIR_RESULTS.json) · [RESULTS.json](../mechanism_v51_20261022/RESULTS.json)

## E072 · medical_admission_v56_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V56：现有数据池资格检查](../medical_admission_v56_20261022/REPORT_ZH.md)。

原报告结论节选：冻结64来源池；复核已审查16项，9个原文与题图一致的临床关系列为只评估保护候选，未审查48项不冒称已拒绝。原图A/C作为后续训练支持候选，B/D/E留出，B是未训练负面板；同来源、切面与注释混杂保留，非独立病例确认。

原聚合文件状态字段：`CLINICAL_RETENTION_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_admission_v56_20261022/REPORT_ZH.md) · [RESULTS.json](../medical_admission_v56_20261022/RESULTS.json)

## E073 · medical_condition8_v81_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V81：教师监督准入失败，未进入四臂训练](../medical_condition8_v81_20261023/REPORT_ZH.md)。

原报告结论节选：2026-10-23。原8编辑与原40个native/FIT输入全部保留，按冻结方案完成120个一次性混合教师采样。源内问答的隐藏arm审核已冻结；之后才解开映射。**40个输入均未同时获得S1/S2/S3合格目标，全部共同回退原参考，非等价可用目标为0。按预先规定的准入门槛，训练前科学终止。**

原聚合文件状态字段：`SCIENTIFIC_STOP_BEFORE_TRAINING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_condition8_v81_20261023/REPORT_ZH.md) · [RESULTS.json](../medical_condition8_v81_20261023/RESULTS.json)

## E074 · medical_floor_v54_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V54：显式正例增量下限](../medical_floor_v54_20261022/REPORT_ZH.md)。

原报告结论节选：已完成，两路EXIT0、96绑定PASS；86项精确冻结评分复用，10个新答案盲评后冻结。未通过开发提升门槛。唯一变化是给 V53 增加 `relu(0.1-delta_positive)`，权重1、下限沿用既定pair margin0.1。V53曾出现正例增量 -0.5165、负例 -1.9059，而pair/negative hinges均为0；本轮修补这一训练目标逻辑缺口，不据此声称它解释所有开发失败。

原聚合文件状态字段：`COMPLETE`；`NO_REGISTERED_DEVELOPMENT_PROMOTION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_floor_v54_20261022/REPORT_ZH.md) · [RESULTS.json](../medical_floor_v54_20261022/RESULTS.json)

## E075 · medical_gradient_v55_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V55：有限训练支持梯度诊断与 Pro 分析核对](../medical_gradient_v55_20261022/REPORT_ZH.md)。

原报告结论节选：两路正常完成：48次模型前向、64次反向、0训练更新、0评估生成。只读取原训练支持，比较零writer与生成所用V54最终权重。没有新损失/参数搜索。完整曲线审查见 CURVE_AUDIT.json，局部梯度及最终分数见 RESULTS.json。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_gradient_v55_20261022/REPORT_ZH.md) · [RESULTS.json](../medical_gradient_v55_20261022/RESULTS.json)

## E076 · medical_pool_review_v57_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V57：剩余48来源完整资格审查](../medical_pool_review_v57_20261022/REPORT_ZH.md)。

原报告结论节选：V56固定9题的失败保持不变。本次完成原64池剩余48项的统一审查，不提前按Base效果终止、不隐式替换V56名单。44项原文可读，4项一次获取失败；48份现有裁剪均已实际查看。逐项理由私有保留；题图错配、参考冲突、非临床、仅模态/布局问题和依赖不可见病例信息均单列。

原聚合文件状态字段：`EXISTING_POOL_CLINICAL_RETENTION_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_pool_review_v57_20261022/REPORT_ZH.md) · [RESULTS.json](../medical_pool_review_v57_20261022/RESULTS.json)

## E077 · medical_ratio_v53_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V53：主线竞争答案相对增量](../medical_ratio_v53_20261022/REPORT_ZH.md)。

原报告结论节选：已完成，两路 EXIT0、96 输入输出绑定 PASS；89 项沿用精确匹配的冻结评分，7 个新答案盲评冻结后汇总。未通过开发提升门槛。V52 绝对目标概率约束在训练代理上达标，但自由答案及保护表现退化。本轮仅改变增量分数，检验它与实际答案选择的一致性；不扩大数据、不搜索权重/层/seed，不重训或优化 CE。

原聚合文件状态字段：`COMPLETE`；`NO_REGISTERED_DEVELOPMENT_PROMOTION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_ratio_v53_20261022/REPORT_ZH.md) · [RESULTS.json](../medical_ratio_v53_20261022/RESULTS.json)

## E078 · medical_support_v52_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V52：有限医学开发包与主线增量约束](../medical_support_v52_20261022/REPORT_ZH.md)。

原报告结论节选：实验已完成，两个 worker 均 EXIT0；96 绑定核验通过，90 个唯一输出盲评后冻结。增量约束未通过开发提升门槛：0/2 来源取得双提示正负配对修复，并出现来源及保护损失。

原聚合文件状态字段：`COMPLETE`；`NO_REGISTERED_DEVELOPMENT_PROMOTION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medical_support_v52_20261022/REPORT_ZH.md) · [RESULTS.json](../medical_support_v52_20261022/RESULTS.json)

## E079 · medmkeb_answer_v12_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V12 固定自由回答检查：源编辑能力未通过](../medmkeb_answer_v12_20261019/REPORT_ZH.md)。

原报告结论节选：两个V11终态完整生成64/64回答，64EOS、0cap，两EXIT0。源回答0/2、own跨图0/2、改写0/2正确；原27eval每任务均11正确/12错误/4不确定，11个Base正确问题均保持，新增损伤0/22（11问题×2编辑配对检查，不是22独立病例）。源能力准入失败。

原聚合文件状态字段：`COMPLETE_ANSWER_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_answer_v12_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_answer_v12_20261019/RESULTS.json)

## E080 · medmkeb_conditional_v21_20261020

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V21 有限数据准入验证结果](../medmkeb_conditional_v21_20261020/REPORT_ZH.md)。

原报告结论节选：本轮未通过模型前的数据门槛，二维写入×路由比较未运行；这不是方法有效或无效的结果。用户取消时间截止已落实，本轮停止原因是冻结的数据准入标准，而非时间用尽。旧V20负结果不改写。

原聚合文件状态字段：`FAIL_INSUFFICIENT_QUALIFIED_SCOPE_COVERAGE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_conditional_v21_20261020/REPORT_ZH.md) · [RESULTS.json](../medmkeb_conditional_v21_20261020/RESULTS.json)

## E081 · medmkeb_direction_audit_v7_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V7 初始方向可行性审计：完成报告](../medmkeb_direction_audit_v7_20261019/REPORT_ZH.md)。

原报告结论节选：两任务48个固定名义候选全部检查完成、两个EXIT0。工程准入及预登记候选资格均通过；这只证明初始一步存在同时降低源CE并满足训练支持容差的非零方向，不是语义保护有效性、长期编辑或独立泛化证据。旧V6负结果不撤销。

原聚合文件状态字段：`COMPLETE_ENGINEERING_AUDIT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_direction_audit_v7_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_direction_audit_v7_20261019/RESULTS.json)

## E082 · medmkeb_direction_v8_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V8 31步轨迹结果](../medmkeb_direction_v8_20261019/REPORT_ZH.md)。

原报告结论节选：有限链已结束：两任务正常EXIT0，全部PROJECT与相同步幅NORM_SOURCE消费者、初始复用准入及save/load检查完成。任务1完成31步并通过单任务轨迹资格；任务2接受20步，第21步全部固定候选违例而停止，剩余11步为NA。预登记的“两任务均完整通过”条件失败，不能直接进入后续完整训练或多方法效用确认。未改变容差、候选比例、迭代上限、样本或收益标准。

原聚合文件状态字段：`TERMINAL_FULL_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[POSTHOC_CPU_REPORT_ZH.md](../medmkeb_direction_v8_20261019/POSTHOC_CPU_REPORT_ZH.md) · [REPORT_ZH.md](../medmkeb_direction_v8_20261019/REPORT_ZH.md) · [POSTHOC_CPU_RESULTS.json](../medmkeb_direction_v8_20261019/POSTHOC_CPU_RESULTS.json) · [RESULTS.json](../medmkeb_direction_v8_20261019/RESULTS.json)

## E083 · medmkeb_expanded_v18_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V18 七组冻结比较：完整结果](../medmkeb_expanded_v18_20261019/REPORT_ZH.md)。

原报告结论节选：全部350方法回答完整，350 EOS/0 cap，两worker EXIT0，隐藏方法固定参考四分类覆盖全部158唯一回答：19份精确继承Base冻结判定，139份新增审阅，0新外部Judge。原56 Base成本保留。两个合格开发编辑全部纳入；不是患者独立或大规模临床确认。

全部结果/审阅/分析及聚合入口：[BASE_REPORT_ZH.md](../medmkeb_expanded_v18_20261019/BASE_REPORT_ZH.md) · [REPORT_ZH.md](../medmkeb_expanded_v18_20261019/REPORT_ZH.md) · [BASE_RESULTS.json](../medmkeb_expanded_v18_20261019/BASE_RESULTS.json) · [RESULTS.json](../medmkeb_expanded_v18_20261019/RESULTS.json)

## E084 · medmkeb_fair_v17_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V17 修订与图像模态准入结果](../medmkeb_fair_v17_20261019/REPORT_ZH.md)。

原报告结论节选：已取消患者ID额外硬门槛，保留完整图像连通组、历史排除和PMC来源角色隔离。患者重叠未知仅限制结论，不再阻止启动。七条件已冻结：SOURCE、CE、KL、CE-PROJ、RESERVE、原版BalancEdit、共同支持BalancEdit+CE；正式阶段覆盖全部合格编辑，最多8个，7200次更新上限。

原聚合文件状态字段：`DATA_GATE_NOT_MET_BEFORE_BASE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_fair_v17_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_fair_v17_20261019/RESULTS.json)

## E085 · medmkeb_full160_v14_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V14 原160步同路径对照结果](../medmkeb_full160_v14_20261019/REPORT_ZH.md)。

原报告结论节选：四个终态全部完成160步，128/128原greedy回答完整，128 EOS、0截断，两EXIT0且精确进程身份均已退出。原31步时SOURCE/RESERVE源、跨图、改写均0/2；本轮完整配方下两条件源和跨图均2/2，改写均1/2。实际源编辑能力资格已通过，但没有识别到语义保护增益。

原聚合文件状态字段：`COMPLETE_FULL160_PAIRED_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_full160_v14_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_full160_v14_20261019/RESULTS.json)

## E086 · medmkeb_guard_v6_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V6 逐对与前瞻语义保护：完成报告](../medmkeb_guard_v6_20261019/REPORT_ZH.md)。

原报告结论节选：本轮完成1600更新、10终态、320回答及完整隐藏组别审阅，两个worker EXIT0。没有达到任何预登记增量保护标准。逐对保护及前瞻回退均未学会两个源编辑；不能把源学习失败和输出不变解释为保护成功。

原聚合文件状态字段：`ALL_TRAINING_COMPLETE`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_guard_v6_20261019/REPORT_ZH.md) · [MECHANISM_RESULTS.json](../medmkeb_guard_v6_20261019/MECHANISM_RESULTS.json) · [RESULTS.json](../medmkeb_guard_v6_20261019/RESULTS.json)

## E087 · medmkeb_independent_admission_v15_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V15 元数据准入：严格独立编辑候选不足](../medmkeb_independent_admission_v15_20261019/REPORT_ZH.md)。

原报告结论节选：固定既有MedMKEB两个原始分区读取完成：train4490记录、eval2497记录。0图像像素读取、0模型/训练/生成/Judge/Fisher调用，0GPU小时，不曾启动Base或公平方法比较。当前结果是来源准入缺口，不是CT图像审查结论、Base准确率或语义方法负结果。

原聚合文件状态字段：`METADATA_COHORT_FROZEN_PATIENT_INDEPENDENCE_UNRESOLVED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_independent_admission_v15_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_independent_admission_v15_20261019/RESULTS.json)

## E088 · medmkeb_locality6_v2_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[MedMKEB训练侧locality补充与正式方法对比准入 V2](../medmkeb_locality6_v2_20261018/REPORT_ZH.md)。

原报告结论节选：本轮在物理GPU0/1完成29次Base生成。训练侧1/8明确正确、评测侧2/21明确正确，未满足预先登记的4个不同来源原正确检查要求，DATA_GATE_NOT_MET。没有修改门槛、补选、改问或训练六组/BalancEdit。此次准入失败不证明方法无效；没有新增方法性能结果。

原聚合文件状态字段：`DATA_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_locality6_v2_20261018/REPORT_ZH.md) · [RESULTS.json](../medmkeb_locality6_v2_20261018/RESULTS.json)

## E089 · medmkeb_locality6_v3_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[扩大CT同模态准入 V3：未达到冻结门槛](../medmkeb_locality6_v3_20261018/REPORT_ZH.md)。

原报告结论节选：本轮已实际在物理GPU0/1完成22次新Base生成，仍未满足至少4个不同PMC来源明确正确的评估检查门槛。没有启动六组、两版BalancEdit、新Fisher或当前任务参考训练，不是方法性能对比结果。

原聚合文件状态字段：`DATA_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_locality6_v3_20261018/REPORT_ZH.md) · [RESULTS.json](../medmkeb_locality6_v3_20261018/RESULTS.json)

## E090 · medmkeb_locality6_v4_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V4：实际图像数据准入通过，语义代理准入未通过](../medmkeb_locality6_v4_20261018/REPORT_ZH.md)。

原报告结论节选：本轮不再按题目是否出现CT筛选，已在物理GPU0/1完成实际图像模态/题图源审查、47次Base准入、训练侧新Fisher和语义构造一致性检查。数据门槛通过，但既定S0>0机制门槛未通过，因此0正式更新，没有执行六组与两版BalancEdit性能对比。不能把数据PASS或实现PASS叫作方法收益。

原聚合文件状态字段：`PROXY_ADMISSION_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_locality6_v4_20261018/REPORT_ZH.md) · [RESULTS.json](../medmkeb_locality6_v4_20261018/RESULTS.json)

## E091 · medmkeb_numeric_v9_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V9 候选构造与计值取证结果](../medmkeb_numeric_v9_20261019/REPORT_ZH.md)。

原报告结论节选：两任务均完整完成取证及32格交叉，EXIT0，19次重建/任务的冻结位移和源梯度前输入核对通过。结果排除了简单的梯度计值模式/缓存错误，也证明单独去除规范化不足以修复E02。不得将预先怀疑当作已证实根因。

原聚合文件状态字段：`COMPLETE_FORENSIC_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_numeric_v9_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_numeric_v9_20261019/RESULTS.json)

## E092 · medmkeb_precision_v10_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V10 精度来源审计结果](../medmkeb_precision_v10_20261019/REPORT_ZH.md)。

原报告结论节选：两任务完整结束、EXIT0，全部原FP32状态及8候选NLL在1e-6内复现，捕获同logits CE32与模型loss差0。只有TT残差收缩与输入归一化采用FP64的对照，不是全Transformer FP64。四类精度读出中，E02全部8候选仍超出原1e-4边界。

原聚合文件状态字段：`COMPLETE_PRECISION_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_precision_v10_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_precision_v10_20261019/RESULTS.json)

## E093 · medmkeb_provenance_v16_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V16 来源核查与完整连通组重划分](../medmkeb_provenance_v16_20261019/REPORT_ZH.md)。

原报告结论节选：本轮按用户新授权在全图像连接组上重划分，解决了V15原官方train/eval共享组直接禁止产生的0编辑候选问题。新冻结8个EDIT_EVAL组、32个RETENTION_TRAIN来源与32个RETENTION_EVAL来源，80张实际审查代表图、111条所选来源成员记录，native/cross连接组和PMC来源在新角色间冲突0。所有旧轮次（包括V15零模型预候选）涉及的历史图像组/论文来源仍排除；

原聚合文件状态字段：`COMPONENT_RESPLIT_FROZEN_STRICT_SOURCE_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_provenance_v16_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_provenance_v16_20261019/RESULTS.json)

## E094 · medmkeb_proxy_audit_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定20训练样本的保护代理审计](../medmkeb_proxy_audit_v1_20261018/REPORT_ZH.md)。

原报告结论节选：120次教师强制forward已完成，0自由生成/0训练更新/0NN反向/0Judge；另一次单独登记6forward工程核对已完成。V4旧停止与负结果保持。本轮没有根据哪种分数更正来替换训练主指标。

原聚合文件状态字段：`AUDIT_COMPLETE_INITIAL_DRIFT_PRESERVED_PREFILL_REPAIR_PASS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_proxy_audit_v1_20261018/REPORT_ZH.md) · [RESULTS.json](../medmkeb_proxy_audit_v1_20261018/RESULTS.json)

## E095 · medmkeb_relative_v5_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V5 相对语义保护：完成报告](../medmkeb_relative_v5_20261019/REPORT_ZH.md)。

原报告结论节选：八条件、两个编辑任务、16个终态及512条方法回答全部完成，两worker EXIT0。结论：本轮仍没有满足预注册的增量语义保护标准；六组内部对照均没有新增损伤，因此保护收益不可识别。此结果不能证明保护无效，也不能作为方法胜过BalancEdit的完整基准证据。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_relative_v5_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_relative_v5_20261019/RESULTS.json)

## E096 · medmkeb_reserve_v11_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V11 31步内部余量修复结果](../medmkeb_reserve_v11_20261019/REPORT_ZH.md)。

原报告结论节选：两任务都完整31/31，EXIT0，全部实际正确CE/margin保护约束<=原1e-4，save/load与汇总消费者通过。E02跨过V8第21步停止点，联合开发轨迹资格通过。内部线性目标固定5e-5，没有放宽外部标准、改候选比例或搜索余量参数。

原聚合文件状态字段：`COMPLETE_RESERVE_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_reserve_v11_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_reserve_v11_20261019/RESULTS.json)

## E097 · medmkeb_route_audit_v19_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V19 冻结专家路由归因：完整结果](../medmkeb_route_audit_v19_20261019/REPORT_ZH.md)。

原报告结论节选：两专家库、28唯一输入、五TT臂140自然消费者齐备；仅5份新回答(5 EOS/0cap)，120份所选专家原forced回答及15份Base回答复用。100份已有BE单专家门控消费者另核验，不假称BE双专家库。两final EXIT0、两native-key与保存anchor差0。无训练/反向/新Base/Judge/Fisher。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_route_audit_v19_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_route_audit_v19_20261019/RESULTS.json)

## E098 · medmkeb_scope_matrix_v20_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V20：SOURCE能力矩阵与三路由结果](../medmkeb_scope_matrix_v20_20261019/REPORT_ZH.md)。

原报告结论节选：本轮全部完成。三个原本答错的源输入经固定160步SOURCE写入后均答对；完整输入×专家矩阵已建立。原R0、同支持联合原型基线与图像/文本一致性排序分别答对留出7/17、8/17、8/17。**新排序没有超过同支持简单基线，三者都损伤3/7条Base原本正确的留出回答。** 本轮结果支持将写入能力、适用范围与实际选择分开检查，尚不支持新排序具有额外语义保护收益。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_scope_matrix_v20_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_scope_matrix_v20_20261019/RESULTS.json)

## E099 · medmkeb_semantic6_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[MedMKEB历史隔离与Base准入：保护检查资格不足](../medmkeb_semantic6_v1_20261018/REPORT_ZH.md)。

原报告结论节选：本轮已复用既有DataP MedMKEB，在物理GPU0/1完整执行24个冻结CT图像组、48个原题原图/跨图输入的Base生成。助手按上游原答案四分类审阅得到1正确、39错误、8不确定、0矛盾；48条均EOS，0截断。这里是源标签一致性，不是独立临床准确率。出现18组两图都明确不符源答案的候选，但0组两图都正确；预注册要求至少2个原正确CT组分别提供保留构造和检查，故DATA_GATE_NOT_MET，六组没有启动，训练更新为0。

原聚合文件状态字段：`DATA_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_semantic6_v1_20261018/REPORT_ZH.md) · [RESULTS.json](../medmkeb_semantic6_v1_20261018/RESULTS.json)

## E100 · medmkeb_source31_v13_20261019

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V13 同31步SOURCE-only配对结果](../medmkeb_source31_v13_20261019/REPORT_ZH.md)。

原报告结论节选：SOURCE-only两任务各完整31源更新、64/64回答、64EOS0cap，两EXIT0。源、own跨图、改写均0/2，新增损伤0/22；与固定V12 RESERVE31同为源0/2/迁移0/2/损伤0/22。两方法27eval每任务均11正确/12错误/4不确定。源能力门槛均失败，控制无损伤，语义保护收益不可识别。

原聚合文件状态字段：`COMPLETE_SOURCE31_PAIRED_CONSUMERS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medmkeb_source31_v13_20261019/REPORT_ZH.md) · [RESULTS.json](../medmkeb_source31_v13_20261019/RESULTS.json)

## E101 · medtrace_12h_20261021_1002

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[MedTRACE前12小时实验总结](../medtrace_12h_20261021_1002/REPORT_ZH.md)。

原报告结论节选：统计窗口：**北京时间2026-10-20 22:02:06—2026-10-21 10:02:06**，以本次汇总任务开始时刻为截止。范围仅本研究会话；本次为已有结果整理，0新增模型调用、0重评分、0新实验。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_12h_20261021_1002/REPORT_ZH.md)

## E102 · medtrace_24gpuh_20260921

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[medtrace_24gpuh_20260921](../medtrace_24gpuh_20260921/)。

## E103 · medtrace_24gpuh_20260921_continuation

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[medtrace_24gpuh_20260921_continuation](../medtrace_24gpuh_20260921_continuation/)。

## E104 · medtrace_aligned_fisher_v1_20261014

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[源方向保持的有限机制](../medtrace_aligned_fisher_v1_20261014/MECHANISM_REPORT_ZH.md)。

原报告结论节选：32配对中21组恢复方向、第一fraction改善7组，真实通过29，机制门槛True。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[MECHANISM_REPORT_ZH.md](../medtrace_aligned_fisher_v1_20261014/MECHANISM_REPORT_ZH.md) · [MECHANISM_RESULTS.json](../medtrace_aligned_fisher_v1_20261014/MECHANISM_RESULTS.json)

## E105 · medtrace_binary_interface_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[知识保持修补：两轮验证完成，均不替换主方法](../medtrace_binary_interface_v1_20261008/REPORT_ZH.md)。

原报告结论节选：本轮没有解决普遍知识保持问题。保留现有 MARGIN_002 主方案；两种轻量修补的收益和代价全部保留，不能把历史保持子集的高分当作整体改进。

原聚合文件状态字段：`COMPLETE_NOT_PROMOTED`；`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_binary_interface_v1_20261008/REPORT_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_binary_interface_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_binary_interface_v1_20261008/RESULTS.json)

## E106 · medtrace_bounded_response_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[收益约束的低扰动方向：完成复核](../medtrace_bounded_response_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：冻结结论 **NO_LOCAL_SUPPORT**。方向不再退化成零写入，但保护与逐专家编辑收益门槛都未过。

原聚合文件状态字段：`PASS`；`COMPLETE`；`NO_LOCAL_SUPPORT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_bounded_response_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_bounded_response_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_bounded_response_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_bounded_response_v1_20261009/RESULTS.json)

## E107 · medtrace_cm_eval146_v1_20261016

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[C/M13原146完整开发评估：未通过兼容门槛](../medtrace_cm_eval146_v1_20261016/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：2026-10-16 22:08北京时间核验，生成与评分均已实际完成。5002/5002唯一payload有效，17081/17081消费者有效，零PENDING/RESERVED/MISSING；四路DONE、全局ALL_WORKERS_COMPLETE、报告COMPLETE和第二恢复控制器EXIT0齐全。全部评估GPU和四路本机评分器退出。此前2249有效评分status/batch/correct逐项保持不变。

原聚合文件状态字段：`SCORING_RESOLVED`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[CM_REPORT_ZH.md](../medtrace_cm_eval146_v1_20261016/CM_REPORT_ZH.md) · [FINAL_ANALYSIS_ZH.md](../medtrace_cm_eval146_v1_20261016/FINAL_ANALYSIS_ZH.md) · [TRAINING_REPORT_ZH.md](../medtrace_cm_eval146_v1_20261016/TRAINING_REPORT_ZH.md) · [CM_RESULTS.json](../medtrace_cm_eval146_v1_20261016/CM_RESULTS.json) · [FINAL_EXECUTION_AUDIT.json](../medtrace_cm_eval146_v1_20261016/FINAL_EXECUTION_AUDIT.json)

## E108 · medtrace_cp_init_ablation_v1_20261005

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[CP 初始化消融 V1：完成审阅](../medtrace_cp_init_ablation_v1_20261005/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：结论：无 H 条件下，本轮支持训练过的 CP 初始化再展开到 FREE 优于直接 FREE 初始化；有 H 条件下未见同样优势。证据仅覆盖事先固定、已暴露的 H8，不能外推到原146全部编辑。

原聚合文件状态字段：`REPORTED_PUBLICATION_PENDING`；`SUPPORTED_ON_EXPOSED_H8_ONLY`。

全部结果/审阅/分析及聚合入口：[SCIENTIFIC_REVIEW_ZH.md](../medtrace_cp_init_ablation_v1_20261005/SCIENTIFIC_REVIEW_ZH.md) · [final_metrics/REPORT_ZH.md](../medtrace_cp_init_ablation_v1_20261005/final_metrics/REPORT_ZH.md) · [final_metrics/PAIRED_RESULTS.json](../medtrace_cp_init_ablation_v1_20261005/final_metrics/PAIRED_RESULTS.json)

## E109 · medtrace_curriculum146_v1_20261017

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[三臂540步课程实验：完整负结果](../medtrace_curriculum146_v1_20261017/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：原146编辑的训练、single生成和评分全部完成。5223个唯一payload均FORMAT_VALID，覆盖19829个消费者，无缺失、待评分或未结算预约。两名CPU控制器正常EXIT0。冻结选择中无人满足全部条件，因此连续编辑为NOT_ADMITTED，未启动GPU生成或Judge调用。

原聚合文件状态字段：`COMPLETE_NOT_ADMITTED`。

全部结果/审阅/分析及聚合入口：[CURRICULUM_REPORT_ZH.md](../medtrace_curriculum146_v1_20261017/CURRICULUM_REPORT_ZH.md) · [FINAL_ANALYSIS_ZH.md](../medtrace_curriculum146_v1_20261017/FINAL_ANALYSIS_ZH.md) · [CONTINUOUS_SELECTION.json](../medtrace_curriculum146_v1_20261017/CONTINUOUS_SELECTION.json) · [CURRICULUM_RESULTS.json](../medtrace_curriculum146_v1_20261017/CURRICULUM_RESULTS.json) · [FINAL_EXECUTION_AUDIT.json](../medtrace_curriculum146_v1_20261017/FINAL_EXECUTION_AUDIT.json)

## E110 · medtrace_fisher_balance146_v1_20261017

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[M13B完成：保持原M13保护条件并改善T2G，预注册联合门槛通过](../medtrace_fisher_balance146_v1_20261017/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：本轮唯一候选M13B在原146 single DEV上通过全部预注册条件：FIT逐编辑保持，T2L及fresh Base正确集损伤逐支持编辑不退化，T2G宏平均严格提高，且评估无缺失。T2G相对M13增加12.6142个百分点，配对edit区间[9.3037,16.0959]pp、source组区间[9.3171,16.1939]pp均为正。这是本次有限DEV实验的成功，不等同于独立确认、完整bank验证或普遍优于无Fisher方案。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[BALANCE_REPORT_ZH.md](../medtrace_fisher_balance146_v1_20261017/BALANCE_REPORT_ZH.md) · [FINAL_ANALYSIS_ZH.md](../medtrace_fisher_balance146_v1_20261017/FINAL_ANALYSIS_ZH.md) · [BALANCE_RESULTS.json](../medtrace_fisher_balance146_v1_20261017/BALANCE_RESULTS.json) · [FINAL_EXECUTION_AUDIT.json](../medtrace_fisher_balance146_v1_20261017/FINAL_EXECUTION_AUDIT.json)

## E111 · medtrace_fisher_direction146_v1_20261017

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[M13P：146独立单次编辑完整结果与机制分析](../medtrace_fisher_direction146_v1_20261017/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：本轮唯一M13P候选已完整结束，预注册的M13B参照五项AND全部通过。相同原始监督、种子、146编辑、160步与7168参数；不含额外跨图训练监督。M13P先保留M13B的Fisher强度上限，再逐RMS组删除惩罚梯度与source反向的平行分量，alpha不重算，不放大正交分量。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[BALANCE_REPORT_ZH.md](../medtrace_fisher_direction146_v1_20261017/BALANCE_REPORT_ZH.md) · [FINAL_ANALYSIS_ZH.md](../medtrace_fisher_direction146_v1_20261017/FINAL_ANALYSIS_ZH.md) · [BALANCE_RESULTS.json](../medtrace_fisher_direction146_v1_20261017/BALANCE_RESULTS.json) · [FINAL_EXECUTION_AUDIT.json](../medtrace_fisher_direction146_v1_20261017/FINAL_EXECUTION_AUDIT.json)

## E112 · medtrace_fisher_min_v1_20261014

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[完整结果与剩余泛化缺口](../medtrace_fisher_min_v1_20261014/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：八例160步、939 forced＋219 natural、1158消费者/350精确payload全部有效，无缺失/PENDING/RESERVED，四worker完成，FULL_EXIT0/COMPLETE与执行审计齐备。87个本轮自有包在最后消费者后释放，历史C/D不动；模型源b41292a、CPU父d689aa0、机制71980c5分别保留。新增GPU进程时17.95244619h，累计179.15215480h；

原聚合文件状态字段：`PASS`；`COMPLETE`；`JOINT_COMPATIBILITY_NOT_ESTABLISHED`。

全部结果/审阅/分析及聚合入口：[FINAL_ANALYSIS_ZH.md](../medtrace_fisher_min_v1_20261014/FINAL_ANALYSIS_ZH.md) · [FINAL_REPORT_ZH.md](../medtrace_fisher_min_v1_20261014/FINAL_REPORT_ZH.md) · [FULL_REPORT_ZH.md](../medtrace_fisher_min_v1_20261014/FULL_REPORT_ZH.md) · [MECHANISM_REPORT_ZH.md](../medtrace_fisher_min_v1_20261014/MECHANISM_REPORT_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_fisher_min_v1_20261014/EXECUTION_AUDIT.json) · [FULL_RESULTS.json](../medtrace_fisher_min_v1_20261014/FULL_RESULTS.json) · [MECHANISM_RESULTS.json](../medtrace_fisher_min_v1_20261014/MECHANISM_RESULTS.json) · [PAIRED_FULL_RESULTS.json](../medtrace_fisher_min_v1_20261014/PAIRED_FULL_RESULTS.json)

## E113 · medtrace_fisher_response_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[全词表局部Fisher响应约束：完成复核](../medtrace_fisher_response_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：冻结结论 **NO_LOCAL_SUPPORT**。没有自由生成语义验证或完整训练/24/146准入。

原聚合文件状态字段：`PASS`；`COMPLETE`；`NO_LOCAL_SUPPORT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_fisher_response_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_fisher_response_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_fisher_response_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_fisher_response_v1_20261009/RESULTS.json)

## E114 · medtrace_fp32_baseline_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[FP32 Base身份复核](../medtrace_fp32_baseline_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：结论：BASELINE_REQUALIFICATION_REQUIRED。六原生控制精确复现旧tokens，157个FP32回答全部EOS结束、无到达上限；BASIS60/61、HELDOUT91/96文本与tokens完全一致。6条变化中原正确5条（BASIS1，主留出4）；变化本身不能判医学错误。必须先完成固定6条独立语义重核验才能判断资格继承。

原聚合文件状态字段：`PASS`；`COMPLETE`；`ORIGINAL_QUALIFICATION_NOT_PRESERVED`；`BASELINE_REQUALIFICATION_REQUIRED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_fp32_baseline_v1_20261009/REPORT_ZH.md) · [REQUALIFICATION_REVIEW_ZH.md](../medtrace_fp32_baseline_v1_20261009/REQUALIFICATION_REVIEW_ZH.md) · [REVIEW_ZH.md](../medtrace_fp32_baseline_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_fp32_baseline_v1_20261009/COMPLETION_AUDIT.json) · [REQUALIFICATION_RESULTS.json](../medtrace_fp32_baseline_v1_20261009/REQUALIFICATION_RESULTS.json) · [RESULTS.json](../medtrace_fp32_baseline_v1_20261009/RESULTS.json)

## E115 · medtrace_fp32_constrained_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[FP32真实编辑约束：局部机制通过，生成验证待完成](../medtrace_fp32_constrained_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：结论LOCAL_MECHANISM_SIGNAL。原63历史正确主面板的同精度编辑增量KL：RAW/MATCHED_RAW 6.2444060479e-08，BOUNDED 2.72776115917e-08，降低56.3167%。

原聚合文件状态字段：`PASS`；`COMPLETE`；`EDIT_BASELINE_QUALIFIED`；`INCOMPLETE_SEMANTIC_EVIDENCE`；`LOCAL_MECHANISM_SIGNAL`。

全部结果/审阅/分析及聚合入口：[GEN_RECOVERY_REVIEW_ZH.md](../medtrace_fp32_constrained_v1_20261009/GEN_RECOVERY_REVIEW_ZH.md) · [GEN_REPORT_ZH.md](../medtrace_fp32_constrained_v1_20261009/GEN_REPORT_ZH.md) · [GEN_REVIEW_ZH.md](../medtrace_fp32_constrained_v1_20261009/GEN_REVIEW_ZH.md) · [REPORT_ZH.md](../medtrace_fp32_constrained_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_fp32_constrained_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_fp32_constrained_v1_20261009/COMPLETION_AUDIT.json) · [GEN_RECOVERY_RESULTS.json](../medtrace_fp32_constrained_v1_20261009/GEN_RECOVERY_RESULTS.json) · [GEN_RESULTS.json](../medtrace_fp32_constrained_v1_20261009/GEN_RESULTS.json) · [RESULTS.json](../medtrace_fp32_constrained_v1_20261009/RESULTS.json)

## E116 · medtrace_fp32_damage_audit_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[合格源修复参照：泛化成立，但保留损伤可测](../medtrace_fp32_damage_audit_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮有限链已完成，审计 PASS。全部80条重建源输出与 PR52 接受结果一致，两个固定160步终点仍各为 native8/8、FIT32/32。未训练输入上存在语义修复，同时固定保留集出现原本正确回答变错。因此，这轮建立了有实际编辑效果、可测遗忘的未保护参照，尚未验证任何保护方法。

原聚合文件状态字段：`COMPLETE`；`QUALIFIED_UNPROTECTED_REFERENCE_AUDITED`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_fp32_damage_audit_v1_20261009/REVIEW_ZH.md) · [RESULTS.json](../medtrace_fp32_damage_audit_v1_20261009/RESULTS.json)

## E117 · medtrace_fp32_optimizer160_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定160步两臂均完成训练内修复；RAW32失败不能归因于TT容量](../medtrace_fp32_optimizer160_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮按预登记完整完成，判定 **BOTH_HAVE_NATIVE_REPAIR**。同原8个零残差TT88专家，RAW与标准Adam均在唯一160步终点达到native8/8、FIT32/32；两臂对应40条回答文本和tokens全部相同。40个独立payload经一次Astra medium评分全部有效，覆盖80个消费者，0missing、0pending、0reserved，无重试。

原聚合文件状态字段：`COMPLETE`；`BOTH_HAVE_NATIVE_REPAIR`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_fp32_optimizer160_v1_20261009/REVIEW_ZH.md) · [SUPERVISION_REVIEW_ZH.md](../medtrace_fp32_optimizer160_v1_20261009/SUPERVISION_REVIEW_ZH.md) · [RESULTS.json](../medtrace_fp32_optimizer160_v1_20261009/RESULTS.json)

## E118 · medtrace_fp32_protection160_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[全TT RAW160保护实验：最终审计](../medtrace_fp32_protection160_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮执行、评分与审计已完成，科学判定为 **EDIT_OR_GENERALIZATION_NOT_PRESERVED**。保护约束减少了本轮开发保留集上的损伤数量，但未保住预先冻结的逐编辑T2G泛化门槛，不能晋升为保护成功或独立确认。没有追加调参、重试或下一轮实验。

原聚合文件状态字段：`COMPLETE`；`EDIT_OR_GENERALIZATION_NOT_PRESERVED`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_fp32_protection160_v1_20261009/REVIEW_ZH.md) · [RESULTS.json](../medtrace_fp32_protection160_v1_20261009/RESULTS.json)

## E119 · medtrace_fp32_short32_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[RAW32门槛未通过：回答已有变化，但native仍未修复](../medtrace_fp32_short32_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮按预登记结束，判定 `STOP_RAW32_NO_NATIVE_REPAIR`。原8专家全部保留，每个从原seed零残差TT出发训练固定32步；没有挑选成功案例、增加步数、启动保护训练或访问保护BASIS/HELDOUT。本轮回答的是固定32步RAW是否已有实际修复能力，不是完整160步训练或TT容量的最终结论。

原聚合文件状态字段：`COMPLETE`；`STOP_RAW32_NO_NATIVE_REPAIR`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_fp32_short32_v1_20261009/REVIEW_ZH.md) · [RAW32_RESULTS.json](../medtrace_fp32_short32_v1_20261009/RAW32_RESULTS.json)

## E120 · medtrace_freshstart_20260922

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[medtrace_freshstart_20260922](../medtrace_freshstart_20260922/)。

## E121 · medtrace_full_method_comparison_20261003

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[既有候选的原始来源视野审阅](../medtrace_full_method_comparison_20261003/H_FRAME_REVIEW_POLICY_ZH.md)。

原报告结论节选：此步不扩展检索，只审阅 final 来源清单的 UNKNOWN 项。原始 VQA-RAD `evaluated/freeform` QA、原 `image_organ` 标签、已通过的测试/角色/病例隔离，以及服务器缓存原图均作为证据。审阅者为 Codex，不能签为新临床医师验证或患者级独立确认。

全部结果/审阅/分析及聚合入口：[H_FRAME_REVIEW_POLICY_ZH.md](../medtrace_full_method_comparison_20261003/H_FRAME_REVIEW_POLICY_ZH.md) · [P0_REPORT_ZH.md](../medtrace_full_method_comparison_20261003/P0_REPORT_ZH.md)

## E122 · medtrace_full_method_comparison_v2_20261003

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[原146完整方法比较：最终审阅](../medtrace_full_method_comparison_v2_20261003/FINAL_REVIEW_ZH.md)。

原报告结论节选：本轮已完成原146单例及真实专家库顺序插入/全库生成，所有共同新Astra评分终结，指标绑定审计通过。结果支持“native保持良好、相比BELoRA泛化更好”，但不支持“全面优于BalancEdit”或“局部性更好”。本轮没有追加训练、重判或改变门槛。

原聚合文件状态字段：`REPORTED_PUBLICATION_PENDING`。

全部结果/审阅/分析及聚合入口：[FINAL_REVIEW_ZH.md](../medtrace_full_method_comparison_v2_20261003/FINAL_REVIEW_ZH.md) · [final_metrics/COMPARISON_REPORT_ZH.md](../medtrace_full_method_comparison_v2_20261003/final_metrics/COMPARISON_REPORT_ZH.md) · [final_metrics/COMPARISON_RESULTS.json](../medtrace_full_method_comparison_v2_20261003/final_metrics/COMPARISON_RESULTS.json)

## E123 · medtrace_gradient_probe_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[诊断结论：历史回放对照由来源梯度主导，不能据此否定原方法](../medtrace_gradient_probe_v1_20261008/REVIEW_ZH.md)。

原报告结论节选：用户明确：回放不属于方法。本次没有训练回放方案、改写方法、选择checkpoint或启动后续干预。历史SOURCE_REPLAY结果仅作为诊断对照，不得列为用户方法的必要模块。

原聚合文件状态字段：`COMPLETE_DIAGNOSTIC_ONLY`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_gradient_probe_v1_20261008/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_gradient_probe_v1_20261008/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_gradient_probe_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_gradient_probe_v1_20261008/RESULTS.json)

## E124 · medtrace_h8_causal_v1_20261004

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[H8 因果消融最终科学审阅](../medtrace_h8_causal_v1_20261004/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：本轮完成原 146 个编辑、H8 四臂 32 次续训、138 个共享专家、真实全库 prefix1/50/100/146、冻结优先队列的 oracle 诊断与共同 Qwen 评分。3,524 个唯一评分对象全部有效，无永久 missing，全部 GPU lease 已结束。单次完整键只尝试一次，没有混入历史 Astra/SOL 判分。

原聚合文件状态字段：`REPORTED_PUBLICATION_PENDING`。

全部结果/审阅/分析及聚合入口：[SCIENTIFIC_REVIEW_ZH.md](../medtrace_h8_causal_v1_20261004/SCIENTIFIC_REVIEW_ZH.md) · [final_metrics/FINAL_REVIEW_ZH.md](../medtrace_h8_causal_v1_20261004/final_metrics/FINAL_REVIEW_ZH.md) · [final_metrics/PAIRED_RESULTS.json](../medtrace_h8_causal_v1_20261004/final_metrics/PAIRED_RESULTS.json)

## E125 · medtrace_m13_146_preparation_v1_20261014

准备与中期资料；当前146终态另见正文。

入口：[MedTRACE 已有实验结果总结（2026-10-15）](../medtrace_m13_146_preparation_v1_20261014/EXPERIMENT_SUMMARY_20261015_ZH.md)。

原报告结论节选：本总结覆盖与当前 M13 / FISHER_MIN 主线直接相关的历史146基线、模块消融、保护约束及修正实验、前20例中期评测和独立加速验证。它是已有证据的汇总，不是新实验或完整146最终报告。运行快照截至北京时间 **2026-10-15 21:23**；历史终态按各原报告记录，保留阴性与缺失。

全部结果/审阅/分析及聚合入口：[SOURCE_REVIEW_PROGRESS_ZH.md](../medtrace_m13_146_preparation_v1_20261014/SOURCE_REVIEW_PROGRESS_ZH.md) · [TRAINING_REPORT_INTEGRATION_ZH.md](../medtrace_m13_146_preparation_v1_20261014/TRAINING_REPORT_INTEGRATION_ZH.md)

## E126 · medtrace_m13_v1_20261012

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[M1+M3 固定补充实验：完整结果审阅](../medtrace_m13_v1_20261012/REVIEW_ZH.md)。

原报告结论节选：M13 未实现编辑、泛化和保护三者兼顾。它保住了 E 的源编辑成绩，显著减少 Base 正确题的新损伤，但 T2G 与自然路由泛化低于 E，且第8例新损伤变差。M13 对 M3 没有确立综合优势，不能据此把组合替换为主方法。执行完整结束与科学门槛通过是两件事；本次结论为 `EDIT_OR_GENERALIZATION_NOT_PRESERVED`。

原聚合文件状态字段：`COMPLETE_EXECUTION_AUDIT_PASS`；`COMPLETE`；`EDIT_OR_GENERALIZATION_NOT_PRESERVED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_m13_v1_20261012/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_m13_v1_20261012/REVIEW_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_m13_v1_20261012/EXECUTION_AUDIT.json) · [RESULTS.json](../medtrace_m13_v1_20261012/RESULTS.json)

## E127 · medtrace_m1_guard_v1_20261012

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[M1源状态与算子保护纠偏：阶段A终结报告](../medtrace_m1_guard_v1_20261012/REPORT_ZH.md)。

原报告结论节选：2026-10-12，结论 **STOP_MECHANISM_GATE_NOT_MET**。原八例、原seed、M1REF/GUARD各80步、20/80节点全部完成，固定机制门槛失败；不进入81–160条件后续，不生成、不运行Judge，不调整预算系数、例子或门槛。

原聚合文件状态字段：`TERMINAL_VERIFIED`；`STOP_MECHANISM_GATE_NOT_MET`；`PHASE_A_COMPLETE`。

全部结果/审阅/分析及聚合入口：[MEMORY_ADMISSION_REVIEW_ZH.md](../medtrace_m1_guard_v1_20261012/MEMORY_ADMISSION_REVIEW_ZH.md) · [REPORT_ZH.md](../medtrace_m1_guard_v1_20261012/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_m1_guard_v1_20261012/EXECUTION_AUDIT.json) · [RESULTS.json](../medtrace_m1_guard_v1_20261012/RESULTS.json)

## E128 · medtrace_minimal_mechanism_v1_20261011

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[最小验证结论：TT 坐标敏感性成立，视觉问题需要分题型解释](../medtrace_minimal_mechanism_v1_20261011/REVIEW_ZH.md)。

原报告结论节选：本轮已完整执行：8 个固定 owner × 初始态/C160 终态 × 3 个等价 TT 表示，48 个临时单步；同时覆盖 56 条源题的原图、固定错配图、视觉嵌入置零。所有 16 个 owner/state 单元完成，864 次语言前向、144 次反向、0.240383 GPU 小时，退出码均为 0。新增 checkpoint、自由生成和 Judge 均为 0。旧实验仍停止，原 C 候选保留。

原聚合文件状态字段：`PASS`。

全部结果/审阅/分析及聚合入口：[NUMERICAL_REPORT_ZH.md](../medtrace_minimal_mechanism_v1_20261011/NUMERICAL_REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_minimal_mechanism_v1_20261011/REVIEW_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_minimal_mechanism_v1_20261011/EXECUTION_AUDIT.json)

## E129 · medtrace_module_transfer_v1_20261011

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[三模块实验：完整结果审阅](../medtrace_module_transfer_v1_20261011/REVIEW_ZH.md)。

原报告结论节选：完整执行已结束：48个候选全部160步，6948/6948消费者有有效评分，1026/1026完整payload有效，其中333个新增评分、693个历史payload精确继承，0missing/0pending/0reserved。工程执行完成；科学判定为 `EDIT_OR_GENERALIZATION_NOT_PRESERVED`，没有满足兼顾编辑、泛化与保护的预登记联合门槛。

原聚合文件状态字段：`PASS`；`COMPLETE`；`EDIT_OR_GENERALIZATION_NOT_PRESERVED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_module_transfer_v1_20261011/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_module_transfer_v1_20261011/REVIEW_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_module_transfer_v1_20261011/EXECUTION_AUDIT.json) · [RESULTS.json](../medtrace_module_transfer_v1_20261011/RESULTS.json)

## E130 · medtrace_multicore_v1_20261013

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[修正机制完整160步性能比较](../medtrace_multicore_v1_20261013/MULTI_FULL_REPORT_ZH.md)。

原报告结论节选：执行状态：JOINT_COMPATIBILITY_NOT_ESTABLISHED。主条件为同时保留M1泛化和M3保护，并完整比较M13/E；本次全部属于已暴露开发数据。

原聚合文件状态字段：`PASS`；`COMPLETE`；`JOINT_COMPATIBILITY_NOT_ESTABLISHED`。

全部结果/审阅/分析及聚合入口：[FINAL_REPORT_ZH.md](../medtrace_multicore_v1_20261013/FINAL_REPORT_ZH.md) · [MULTI_FULL_REPORT_ZH.md](../medtrace_multicore_v1_20261013/MULTI_FULL_REPORT_ZH.md) · [ROOT_FIX_FULL_REPORT_ZH.md](../medtrace_multicore_v1_20261013/ROOT_FIX_FULL_REPORT_ZH.md) · [TRAJECTORY_ANALYSIS_ZH.md](../medtrace_multicore_v1_20261013/TRAJECTORY_ANALYSIS_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_multicore_v1_20261013/EXECUTION_AUDIT.json) · [MECHANISM_RESULTS.json](../medtrace_multicore_v1_20261013/MECHANISM_RESULTS.json) · [MULTI_FULL_RESULTS.json](../medtrace_multicore_v1_20261013/MULTI_FULL_RESULTS.json) · [PAIRED_FULL_RESULTS.json](../medtrace_multicore_v1_20261013/PAIRED_FULL_RESULTS.json) · [ROOT_FIX_FULL_RESULTS.json](../medtrace_multicore_v1_20261013/ROOT_FIX_FULL_RESULTS.json)

## E131 · medtrace_native146_interim20_v1_20261015

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[前20例授权补判结果：全部有效](../medtrace_native146_interim20_v1_20261015/INTERIM20_RECOVERED_REPORT_ZH.md)。

原报告结论节选：用户授权仅补原缺失200份；保留原4批失败、原717次尝试和费用，不重生成，GPU0/1原训练不变。独立补判4批各50串行全部FORMAT_VALID，新增200 Judge，零GPU/生成/训练；与原517有效合并后717/717份评分、1077/1077条消费者均有效，0缺失/PENDING/RESERVED。

原聚合文件状态字段：`INTERIM20_AUTHORIZED_RECOVERY_RESOLVED`；`INTERIM20_SCORING_RESOLVED`。

全部结果/审阅/分析及聚合入口：[INTERIM20_RECOVERED_REPORT_ZH.md](../medtrace_native146_interim20_v1_20261015/INTERIM20_RECOVERED_REPORT_ZH.md) · [INTERIM20_REPORT_ZH.md](../medtrace_native146_interim20_v1_20261015/INTERIM20_REPORT_ZH.md) · [INTERIM20_RECOVERED_RESULTS.json](../medtrace_native146_interim20_v1_20261015/INTERIM20_RECOVERED_RESULTS.json) · [INTERIM20_RESULTS.json](../medtrace_native146_interim20_v1_20261015/INTERIM20_RESULTS.json)

## E132 · medtrace_online_fine2x2_v1_20261011

用户停止的部分交付，非完整2x2效应。

入口：[在线2×2：用户要求停止后的部分交付](../medtrace_online_fine2x2_v1_20261011/PARTIAL_REVIEW_ZH.md)。

原报告结论节选：状态为PARTIAL_USER_STOPPED。用户明确要求停止本轮、保留恢复状态，将其作为部分结果交付，并把研究重点转向算法验证/开发。停止原因是研究方向调整，不是达到64 GPU小时上限，也不是观察到N/T/NT科学失败。

原聚合文件状态字段：`PARTIAL_USER_STOPPED`。

全部结果/审阅/分析及聚合入口：[ALGORITHM_REVIEW_ZH.md](../medtrace_online_fine2x2_v1_20261011/ALGORITHM_REVIEW_ZH.md) · [PARTIAL_REVIEW_ZH.md](../medtrace_online_fine2x2_v1_20261011/PARTIAL_REVIEW_ZH.md) · [PARTIAL_RESULTS.json](../medtrace_online_fine2x2_v1_20261011/PARTIAL_RESULTS.json)

## E133 · medtrace_paper_transfer_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[TT-only TIME / M-ORE 模块迁移：完成结论](../medtrace_paper_transfer_v1_20261008/REVIEW_ZH.md)。

原报告结论节选：本轮于 2026-10-08 12:27:08（北京时间）结束。固定矩阵全部完成生成，但评分终态含缺失；没有建立可替换原基线的稳定收益。CP 在正式训练开始前已按用户指令取消。

原聚合文件状态字段：`PASS_EXECUTION_TERMINAL_WITH_MISSING_SCORES`；`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_paper_transfer_v1_20261008/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_paper_transfer_v1_20261008/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_paper_transfer_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_paper_transfer_v1_20261008/RESULTS.json)

## E134 · medtrace_pooling_20260907T060435Z

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[GPT Pro 审阅：CP-independent pooling，尚未启动的开发实验](../medtrace_pooling_20260907T060435Z/GPT_PRO_REVIEW.md)。

原报告结论节选：本次交付是**代码和启动前诊断审阅包，不是完成的实验结果**。21 个任务包、63 个 image-head fits 均未开始；没有新模型结果、新 Judge 调用或新科学结论。历史 `scientific_gain=false` 和 LoRA `QUAL_VALIDATION_FAIL` 不变。

全部结果/审阅/分析及聚合入口：[CPU_REGRESSION_REPORT.md](../medtrace_pooling_20260907T060435Z/CPU_REGRESSION_REPORT.md) · [GPT_PRO_REVIEW.md](../medtrace_pooling_20260907T060435Z/GPT_PRO_REVIEW.md)

## E135 · medtrace_post31_scope_hold_v1_20261006

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[PR31 后续实验：A/B 阶段结果与交付](../medtrace_post31_scope_hold_v1_20261006/REPORT_ZH.md)。

原报告结论节选：本次交付完成的机制检查、文本语义诊断和冻结权重路由诊断。完整用户协议尚未完成：C 未运行，C2 未准入，独立确认 BLOCKED_CONFIRMATION。本轮没有新 TT 训练，因此无法回答 HOLD_U 与 U_ONLY 的续训价值问题。

原聚合文件状态字段：`DEVELOPMENT_DIAGNOSTICS_COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_post31_scope_hold_v1_20261006/REPORT_ZH.md) · [B_RESULTS.json](../medtrace_post31_scope_hold_v1_20261006/B_RESULTS.json)

## E136 · medtrace_post32_closeout_v1_20261007

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[Astra medium完整评分：补齐50份后的最终版本](../medtrace_post32_closeout_v1_20261007/ASTRA_MEDIUM_RECOVERED_REPORT_ZH.md)。

原报告结论节选：用户明确授权对原超时50份各追加一次评分。50份全部成功，合并后2,137份不同输入全部有效、0缺失，覆盖9,501条消费者记录且0缺失。原2087份有效分数保持不变，原50份超时记录、原Qwen队列和首次缺失报告均保留；只在独立恢复队列及新汇总版本中补齐。

原聚合文件状态字段：`COMPLETE_NO_MISSING`；`Retrospective development; not independent confirmation`；`SCORING_TERMINAL_WITH_50_MISSING`；`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[ASTRA_MEDIUM_RECOVERED_REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/ASTRA_MEDIUM_RECOVERED_REPORT_ZH.md) · [ASTRA_MEDIUM_REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/ASTRA_MEDIUM_REPORT_ZH.md) · [COMBO24_REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/COMBO24_REPORT_ZH.md) · [COMBO24_REVIEW_ZH.md](../medtrace_post32_closeout_v1_20261007/COMBO24_REVIEW_ZH.md) · [LORA146_REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/LORA146_REPORT_ZH.md) · [LORA146_REVIEW_ZH.md](../medtrace_post32_closeout_v1_20261007/LORA146_REVIEW_ZH.md) · [REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/REPORT_ZH.md) · [RESEARCH_REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/RESEARCH_REPORT_ZH.md) · [RESEARCH_REVIEW_ZH.md](../medtrace_post32_closeout_v1_20261007/RESEARCH_REVIEW_ZH.md) · [WEAK24_REPORT_ZH.md](../medtrace_post32_closeout_v1_20261007/WEAK24_REPORT_ZH.md) · [WEAK24_REVIEW_ZH.md](../medtrace_post32_closeout_v1_20261007/WEAK24_REVIEW_ZH.md) · [ASTRA_MEDIUM_RECOVERED_RESULTS.json](../medtrace_post32_closeout_v1_20261007/ASTRA_MEDIUM_RECOVERED_RESULTS.json) · [ASTRA_MEDIUM_RESULTS.json](../medtrace_post32_closeout_v1_20261007/ASTRA_MEDIUM_RESULTS.json) · [BENCHMARK146_RESULTS.json](../medtrace_post32_closeout_v1_20261007/BENCHMARK146_RESULTS.json) · [COMBO24_RESULTS.json](../medtrace_post32_closeout_v1_20261007/COMBO24_RESULTS.json) · [CORE_RESULTS.json](../medtrace_post32_closeout_v1_20261007/CORE_RESULTS.json) · [LORA146_RESULTS.json](../medtrace_post32_closeout_v1_20261007/LORA146_RESULTS.json) · [RESEARCH_RESULTS.json](../medtrace_post32_closeout_v1_20261007/RESEARCH_RESULTS.json) · [WEAK24_RESULTS.json](../medtrace_post32_closeout_v1_20261007/WEAK24_RESULTS.json)

## E137 · medtrace_prefix_cache_v1_20261016

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[C/M13 固定前缀缓存验收](../medtrace_prefix_cache_v1_20261016/REPORT_ZH.md)。

原报告结论节选：独立验收 PASS。原前两例、C/M13 两方法、初始/终态克隆共8组，每组4步配对；五种 teacher-forcing 输入的全部 logits/loss、训练梯度、更新状态、Adam/RMS、审计字段、路由 mask 与 Python/PyTorch/CUDA/NumPy RNG 全部逐项一致。无容差放宽。CPU 另验证缓存命中、输入版本修改拒绝和恢复原方法。

原聚合文件状态字段：`PASS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_prefix_cache_v1_20261016/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_prefix_cache_v1_20261016/EXECUTION_AUDIT.json) · [RESULTS.json](../medtrace_prefix_cache_v1_20261016/RESULTS.json)

## E138 · medtrace_protected_initial_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[零初始化保护投影：未通过局部机制门槛](../medtrace_protected_initial_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮从原种子零残差TT出发，4832前向/784反向/8次临时候选更新全部完成，808组未编辑Base与零TT完整logits逐元素相同。没有生成、Judge或持久新checkpoint。初始化路径及内容/EOS拆分检查通过。

原聚合文件状态字段：`PASS`；`COMPLETE`；`NO_LOCAL_SUPPORT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_protected_initial_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_protected_initial_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_protected_initial_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_protected_initial_v1_20261009/RESULTS.json)

## E139 · medtrace_protected_response_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[保护响应方向：复核结论](../medtrace_protected_response_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：**观察到等幅下的响应稳定性信号，但预登记联合判据未通过。** 自动结果 `INCONCLUSIVE_SMALL_RESPONSE` 保留原样；实际原因是 **MATCHED_RAW 的平均编辑进展为负**，不是保护侧响应太小。此标签将两个可测性条件合并命名，阅读时应以具体数值及本复核原因为准。不得将结果重标为完整算法成功。

原聚合文件状态字段：`PASS`；`COMPLETE`；`INCONCLUSIVE_SMALL_RESPONSE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_protected_response_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_protected_response_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_protected_response_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_protected_response_v1_20261009/RESULTS.json)

## E140 · medtrace_qualified_response_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[正确完整响应保护：完成复核](../medtrace_qualified_response_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：冻结结论 **NO_LOCAL_SUPPORT**。4552前向、504反向、8临时候选更新全部完成，6个GPU会话结束；808零TT/Base logits精确一致、8重复基线、8状态恢复和8实际映射等幅检查通过。0新增生成/Judge/持久checkpoint，无失败或重试。源约束均为61个非零方向，浮点回写后的源线性响应范数约为RAW的1.7–2.6e-8，投影执行本身成立。

原聚合文件状态字段：`PASS`；`COMPLETE`；`NO_LOCAL_SUPPORT`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_qualified_response_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_qualified_response_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_qualified_response_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_qualified_response_v1_20261009/RESULTS.json)

## E141 · medtrace_replay_diagnosis_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定轨迹结论：未观察到无损保护窗口](../medtrace_replay_diagnosis_v1_20261008/REVIEW_ZH.md)。

原报告结论节选：16个固定调用专家、174条既有查询和48条来源CHECK的有限轨迹已完成。0/192步复用历史输出，12/48/96步新生成；本轮没有选择或晋级任何checkpoint。完整表见TRAJECTORY_REPORT_ZH.md，所有配对区间与支持集见TRAJECTORY_RESULTS.json。

原聚合文件状态字段：`TERMINAL_WITH_INHERITED_MISSING_NOT_PROMOTED`；`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_replay_diagnosis_v1_20261008/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_replay_diagnosis_v1_20261008/REVIEW_ZH.md) · [TRAJECTORY_REPORT_ZH.md](../medtrace_replay_diagnosis_v1_20261008/TRAJECTORY_REPORT_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_replay_diagnosis_v1_20261008/COMPLETION_AUDIT.json) · [TRAJECTORY_RESULTS.json](../medtrace_replay_diagnosis_v1_20261008/TRAJECTORY_RESULTS.json)

## E142 · medtrace_response_calibration_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定候选响应校准：完成复核](../medtrace_response_calibration_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：结论：**CALIBRATION_ONLY_NO_METHOD_PROMOTION**。本轮定位了局部响应预测失配的重要数值来源；旧PR47的NO_LOCAL_SUPPORT不变，不授予留出集、自由生成或完整训练准入。

原聚合文件状态字段：`PASS`；`COMPLETE`；`CALIBRATION_ONLY_NO_METHOD_PROMOTION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_response_calibration_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_response_calibration_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_response_calibration_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_response_calibration_v1_20261009/RESULTS.json)

## E143 · medtrace_root_joint_v1_20261013

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[同根空间联合可行性：终结报告](../medtrace_root_joint_v1_20261013/REPORT_ZH.md)。

原报告结论节选：结论：**NO_VERIFIED_SEARCH_MISS_WITNESS**。八个固定首拒绝状态、32个局部问题全部完成，未找到满足原真实验收条件的搜索漏解见证。工程执行通过，机制检验为阴性；不能据此断言真实非线性模型或完整 TT 空间无解。

原聚合文件状态字段：`PASS`；`COMPLETE`；`NO_VERIFIED_SEARCH_MISS_WITNESS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_root_joint_v1_20261013/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_root_joint_v1_20261013/EXECUTION_AUDIT.json) · [RESULTS.json](../medtrace_root_joint_v1_20261013/RESULTS.json)

## E144 · medtrace_root_refine_v1_20261013

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[修正机制完整160步性能比较](../medtrace_root_refine_v1_20261013/FULL_REPORT_ZH.md)。

原报告结论节选：完整八例160步训练、1158个评测消费者和评分已结束，0缺失。冻结结论为 **JOINT_COMPATIBILITY_NOT_ESTABLISHED**：本轮显著牺牲了编辑和泛化，没有同时保留M1能力与M3保护，因此条件146不会启动。所有数据均为已暴露DEV。

原聚合文件状态字段：`PASS`；`COMPLETE`；`JOINT_COMPATIBILITY_NOT_ESTABLISHED`；`REFINEMENT_WITNESS_FOUND`。

全部结果/审阅/分析及聚合入口：[FULL_REPORT_ZH.md](../medtrace_root_refine_v1_20261013/FULL_REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_root_refine_v1_20261013/REVIEW_ZH.md) · [SOURCE_EXTRA6_REVIEW_ZH.md](../medtrace_root_refine_v1_20261013/SOURCE_EXTRA6_REVIEW_ZH.md) · [EXECUTION_AUDIT.json](../medtrace_root_refine_v1_20261013/EXECUTION_AUDIT.json) · [FULL_RESULTS.json](../medtrace_root_refine_v1_20261013/FULL_RESULTS.json) · [PAIRED_PRIOR_RESULTS.json](../medtrace_root_refine_v1_20261013/PAIRED_PRIOR_RESULTS.json) · [RESULTS.json](../medtrace_root_refine_v1_20261013/RESULTS.json)

## E145 · medtrace_route_gate_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[冻结 TT 路由分解：完成结论](../medtrace_route_gate_v1_20261008/REVIEW_ZH.md)。

原报告结论节选：本轮于2026-10-08 14:52:16（北京时间）完成。原146个TT88 W0、Base、解码均冻结，CP未恢复，没有新增训练。7个固定条件全部执行；不是按评分筛选后的子集。

原聚合文件状态字段：`PASS_EXECUTION_TERMINAL_WITH_MISSING_SCORES`；`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_route_gate_v1_20261008/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_route_gate_v1_20261008/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_route_gate_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_route_gate_v1_20261008/RESULTS.json)

## E146 · medtrace_route_margin_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[模态排序修正完成报告](../medtrace_route_margin_v1_20261008/REPORT_ZH.md)。

原报告结论节选：主条件：**原半径候选集＋模态分差至少 0.02 才切换，否则保留 R0**。已对全部 1513 输入完成精确回放；146 个 TT 专家权重、任务分母和生成配置不变。

原聚合文件状态字段：`PASS_COMPLETED_EXACT_REPLAY`；`TERMINAL_WITH_INHERITED_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_route_margin_v1_20261008/REPORT_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_route_margin_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_route_margin_v1_20261008/RESULTS.json)

## E147 · medtrace_scope_anchor_audit_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[八专家作用域锚点准入：数据门槛未通过，未启动GPU训练](../medtrace_scope_anchor_audit_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：2026-10-09。按用户授权完成PR54已有输出配对复盘及本地来源锚点资格审计。终态为 **DATA_GATE_NOT_MET**。六张GPU使用授权有效，但该授权带有“数据资格通过后才启动C/D”的依赖；本轮未取得全八专家合格锚点集，故未训练、未重建候选或新增自然路由生成。不是GPU故障或训练失败。

原聚合文件状态字段：`PASS`；`COMPLETE`；`DATA_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_scope_anchor_audit_v1_20261009/REVIEW_ZH.md) · [PAIRED_RESULTS.json](../medtrace_scope_anchor_audit_v1_20261009/PAIRED_RESULTS.json) · [RESULTS.json](../medtrace_scope_anchor_audit_v1_20261009/RESULTS.json)

## E148 · medtrace_scope_anchor_supplement_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[来源补充审阅](../medtrace_scope_anchor_supplement_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本次有限CPU补充完成，结论为 **SOURCE_EVIDENCE_GATE_PASS**：8项各2条正锚点，共16条、14个不同图像；第1/8项共享2张训练图，未冒充16个独立病例。第8项已另立一般风险因素版本，并冻结native、4FIT、4T1G和4T2G输入。其他7项源任务逐字段保持一致。尚未运行新版Base、训练或评分，不能称为方法成功。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_scope_anchor_supplement_v1_20261009/REVIEW_ZH.md)

## E149 · medtrace_scope_cd160_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[C/D 双组保护实验：完成，但未通过联合科学门槛](../medtrace_scope_cd160_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：结论为 **EDIT_OR_GENERALIZATION_NOT_PRESERVED**。18条固定160步轨迹全部完成；强制评价和原R0自然路由完成；2567个消费者、765个唯一评分输入全部有效，无缺失。D减少附带损伤，但未保持训练锚点与跨图泛化，且一位owner损伤增加，因此不能宣称合格保护成功或独立验证。

原聚合文件状态字段：`COMPLETE`；`EDIT_OR_GENERALIZATION_NOT_PRESERVED`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_scope_cd160_v1_20261009/REVIEW_ZH.md) · [RESULTS.json](../medtrace_scope_cd160_v1_20261009/RESULTS.json)

## E150 · medtrace_selective_write_20260907

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[GPT Pro review](../medtrace_selective_write_20260907/GPT_PRO_REVIEW.md)。

原报告结论节选：See the numerical summary for all eleven A2/P4/L16 conditions and limitations. Key findings: calibrated P4 W1 (lambda 0.1) retains 100% evaluation positive semantics and reduces U Base-correct damage from 31.46% to 10.63…

全部结果/审阅/分析及聚合入口：[CPU_REGRESSION_REPORT.md](../medtrace_selective_write_20260907/CPU_REGRESSION_REPORT.md) · [GPT_PRO_REVIEW.md](../medtrace_selective_write_20260907/GPT_PRO_REVIEW.md) · [SELECTIVE_WRITE_REPORT.md](../medtrace_selective_write_20260907/SELECTIVE_WRITE_REPORT.md) · [TRANSFER_CPU_RESULTS.json](../medtrace_selective_write_20260907/TRANSFER_CPU_RESULTS.json)

## E151 · medtrace_single_factorial146_v1_20261016

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[原146 single四臂机制实验：完成，主门槛未通过](../medtrace_single_factorial146_v1_20261016/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：本轮全部训练、生成和评分完成，无缺失。M3相对M13的T2G宏平均增加0.7991个百分点，但T2L下降1.0753个百分点、fresh Base正确样本新增损伤从3增至4。因此“保持FIT/T2L及逐owner保护，同时改善T2G”的预注册AND门槛失败。不能将小幅T2G上升单独作为成功。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_ANALYSIS_ZH.md](../medtrace_single_factorial146_v1_20261016/FINAL_ANALYSIS_ZH.md) · [FACTORIAL_RESULTS.json](../medtrace_single_factorial146_v1_20261016/FACTORIAL_RESULTS.json) · [FINAL_EXECUTION_AUDIT.json](../medtrace_single_factorial146_v1_20261016/FINAL_EXECUTION_AUDIT.json)

## E152 · medtrace_source_answer_audit_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[保护目标应对应真实回答，而非把严格短答案匹配当知识资格](../medtrace_source_answer_audit_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：**已有足够的Base语义正确支持，但原短标注目标的严格token匹配不能代表知识正确性。** 固定192题均生成并获得原隔离Astra medium有效评分，124正确、68错误、0缺失；没有补判或修改标注。

原聚合文件状态字段：`COMPLETE`；`SEMANTIC_SUPPORT_AVAILABLE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_source_answer_audit_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_source_answer_audit_v1_20261009/REVIEW_ZH.md) · [RESULTS.json](../medtrace_source_answer_audit_v1_20261009/RESULTS.json)

## E153 · medtrace_source_endpoint_v1_20261010

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[E 源终态保持：通过开发集联合门槛，保护优势仍不确定](../medtrace_source_endpoint_v1_20261010/REVIEW_ZH.md)。

原报告结论节选：本轮完成，结论 **DEVELOPMENT_PROTECTION_SIGNAL_NOT_INDEPENDENT_CONFIRMATION**。预注册E−C的源/泛化逐owner保持、整体损伤下降、无owner损伤增加及原63正确率不下降均通过；但保护差值的置信区间包含零，不能称统计上已确认的保护优势，更不能称独立验证、临床有效性或完整方法成功。

原聚合文件状态字段：`COMPLETE`；`DEVELOPMENT_PROTECTION_SIGNAL_NOT_INDEPENDENT_CONFIRMATION`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../medtrace_source_endpoint_v1_20261010/REVIEW_ZH.md) · [RESULTS.json](../medtrace_source_endpoint_v1_20261010/RESULTS.json)

## E154 · medtrace_source_replay_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[补评分结果：保护压力测试改善，但主泛化退化](../medtrace_source_replay_v1_20261008/RECOVERED_REPORT_ZH.md)。

原报告结论节选：用户授权对首轮302条缺失各追加一次gpt-6-astra medium评分。本次200条成功，102条再次连接失败；合并2256/2358有效，102缺失，零待评分/预约。首轮2056条有效评分完全不变，原302条缺失队列和失败成本完整保留。所有新有效结果通过同输入绑定、模型/推理强度、格式和隔离证据校验。

原聚合文件状态字段：`COMPLETE_WITH_MISSING_NOT_PROMOTED`；`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[RECOVERED_REPORT_ZH.md](../medtrace_source_replay_v1_20261008/RECOVERED_REPORT_ZH.md) · [REPORT_ZH.md](../medtrace_source_replay_v1_20261008/REPORT_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_source_replay_v1_20261008/COMPLETION_AUDIT.json) · [RECOVERED_RESULTS.json](../medtrace_source_replay_v1_20261008/RECOVERED_RESULTS.json) · [RESULTS.json](../medtrace_source_replay_v1_20261008/RESULTS.json)

## E155 · medtrace_stage10_20260910

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage10 anatomical evidence development results](../medtrace_stage10_20260910/GPT_PRO_REVIEW.md)。

原报告结论节选：Compute is complete on the frozen supported subset: 5 of the 15 Stage9 edits, 10 E1/E2 trajectories at step160, Judge387/387 (386 reused, 1 new). The other10 edits have no eligible evidence and were not trained. E0 is ex…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage10_20260910/GPT_PRO_REVIEW.md)

## E156 · medtrace_stage11_20260911

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage11 joint fact writer comparison](../medtrace_stage11_20260911/GPT_PRO_REVIEW.md)。

原报告结论节选：All45 planned final trajectories and90 step endpoints completed; Judge1171/1171, missing0 (1017 reused,154 new). Recorded wall time78.60 minutes; process-interval GPU time upper bound2.563 hours. Execution source2569831.…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage11_20260911/GPT_PRO_REVIEW.md)

## E157 · medtrace_stage12_v2_20260911

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage12 V2 focused closeout](../medtrace_stage12_v2_20260911/GPT_PRO_REVIEW.md)。

原报告结论节选：H-source supervision has a positive incremental result on the frozen writer-level core objective: native correctness stays15/15, selected FIT pairs increase0/23 to23/23, and EVAL pairs increase0/19 to11/19. Both methods …

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage12_v2_20260911/GPT_PRO_REVIEW.md)

## E158 · medtrace_stage13_20260911

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage13 frozen new-edit evaluation: source closeout](../medtrace_stage13_20260911/GPT_PRO_REVIEW.md)。

原报告结论节选：**Actual N=0. No student training, generation or Judge was run. This is not a negative scientific result.**

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage13_20260911/GPT_PRO_REVIEW.md)

## E159 · medtrace_stage13r_20260911

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage13R frozen new-source comparison](../medtrace_stage13r_20260911/GPT_PRO_REVIEW.md)。

原报告结论节选：All 7 new edits and 21 writer branches completed on GPU0. Training, generation, Judge preparation, Judge and reporting exited successfully. Wall time was 59.10 minutes; GPU process time was 0.975 GPU-hours. Judge coverag…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage13r_20260911/GPT_PRO_REVIEW.md)

## E160 · medtrace_stage14_20260911

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage14 fixed supervision-matched closeout](../medtrace_stage14_20260911/GPT_PRO_REVIEW.md)。

原报告结论节选：Stage14 supports a targeted benefit of the selected H facts beyond this particular matched-count extra-QA control, not universal superiority or a better fixed-RC deployed system. Extra correct QA improves U behavior but …

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage14_20260911/GPT_PRO_REVIEW.md)

## E161 · medtrace_stage15_20260911

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage15 external evaluation](../medtrace_stage15_20260911/GPT_PRO_REVIEW.md)。

原报告结论节选：Both implemented branches completed all 200 independent edits. Native target output match is 100% for both, and R0 retains high textual and cross-image generalization. The primary frozen RC system does **not** transfer w…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage15_20260911/GPT_PRO_REVIEW.md)

## E162 · medtrace_stage16_20260912

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage16 实验报告：冻结系统覆盖补全与 Astra 评分汇总](../medtrace_stage16_20260912/GPT_PRO_REVIEW.md)。

原报告结论节选：日期：2026-09-12。用途：交给 GPT Pro 审阅现有证据、判断后续研究方向；不是新实验启动指令。

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage16_20260912/GPT_PRO_REVIEW.md) · [STAGE16_A_READONLY_REPORT.md](../medtrace_stage16_20260912/STAGE16_A_READONLY_REPORT.md)

## E163 · medtrace_stage17_20260912

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage17-A：材料衔接与接口检查](../medtrace_stage17_20260912/GPT_PRO_REVIEW.md)。

原报告结论节选：已完成本次**元数据衔接、预执行协议登记、CPU 合同检查和 GPU3 开发样本机械检查**。这不是正式 single/sequential 实验的完成报告：Stage17 的正式队列、Base 正确性分层和 H/G/U 支持成员仍未冻结，不能把历史 179 条写成本轮已经合法确认的 N。尚未训练或评分任何 Stage17 正式编辑。

原聚合文件状态字段：`MAIN_T0_COHORT_REPORTED_NOT_YET_PUBLISHED`；`C_NO_H_REPORTED_NOT_YET_PUBLISHED`；`COMPLETED_PHASES_REPORTED_NOT_YET_PUBLISHED`。

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage17_20260912/GPT_PRO_REVIEW.md) · [campaign_closeout/GPT_PRO_REVIEW.md](../medtrace_stage17_20260912/campaign_closeout/GPT_PRO_REVIEW.md) · [cnoh_priority_closeout/GPT_PRO_REVIEW.md](../medtrace_stage17_20260912/cnoh_priority_closeout/GPT_PRO_REVIEW.md) · [cnoh_pro_review_20260915/REPORT_ZH.md](../medtrace_stage17_20260912/cnoh_pro_review_20260915/REPORT_ZH.md) · [formal/SCHEDULED_RESUME_REPORT.md](../medtrace_stage17_20260912/formal/SCHEDULED_RESUME_REPORT.md) · [ready_methods_closeout/GPT_PRO_REVIEW.md](../medtrace_stage17_20260912/ready_methods_closeout/GPT_PRO_REVIEW.md) · [single_be_closeout/GPT_PRO_REVIEW.md](../medtrace_stage17_20260912/single_be_closeout/GPT_PRO_REVIEW.md) · [campaign_closeout/CAMPAIGN_RESULTS.json](../medtrace_stage17_20260912/campaign_closeout/CAMPAIGN_RESULTS.json) · [cnoh_priority_closeout/CAMPAIGN_RESULTS.json](../medtrace_stage17_20260912/cnoh_priority_closeout/CAMPAIGN_RESULTS.json) · [ready_methods_closeout/CAMPAIGN_RESULTS.json](../medtrace_stage17_20260912/ready_methods_closeout/CAMPAIGN_RESULTS.json)

## E164 · medtrace_stage18_cfact_20260917

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Astra 审阅与新 Base 筛选（2026-09-17）](../medtrace_stage18_cfact_20260917/ASTRA_REVIEW_AND_BASE_ZH.md)。

原报告结论节选：本轮按用户“直接用 astra 审阅”完成了看图审阅，并在 GPU3 完成 13 条新 Base 生成及统一 Astra 判定。**当前严格合格的新增训练队列为 0；没有启动新的 student 训练。** 算力与训练权限已具备，阻塞来自具体数据资格。

原聚合文件状态字段：`DEV16_SCORED_EXPLORATORY`；`CPU_AND_REAL_GPU_SMOKE_PASS`。

全部结果/审阅/分析及聚合入口：[ASTRA_REVIEW_AND_BASE_ZH.md](../medtrace_stage18_cfact_20260917/ASTRA_REVIEW_AND_BASE_ZH.md) · [DEV16_RESULTS.json](../medtrace_stage18_cfact_20260917/DEV16_RESULTS.json) · [TEST_RESULTS.json](../medtrace_stage18_cfact_20260917/TEST_RESULTS.json)

## E165 · medtrace_stage19_20260917

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage19：DEV16 专家库诊断](../medtrace_stage19_20260917/DIAGNOSTIC_REPORT_ZH.md)。

原报告结论节选：本轮有界诊断、评分和数据建设已完成，新七配置训练尚未启动。Stage18 原主表保持不变，证据起点为公开提交 `2cedcd7dcc130e2cf301a87d640ac38837bf34c9`。

全部结果/审阅/分析及聚合入口：[DIAGNOSTIC_REPORT_ZH.md](../medtrace_stage19_20260917/DIAGNOSTIC_REPORT_ZH.md)

## E166 · medtrace_stage19_fasttrack_20260917

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

全部结果/审阅/分析及聚合入口：[RESULTS.json](../medtrace_stage19_fasttrack_20260917/RESULTS.json)

## E167 · medtrace_stage19_pure_20260918

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

全部结果/审阅/分析及聚合入口：[RESULTS.json](../medtrace_stage19_pure_20260918/RESULTS.json)

## E168 · medtrace_stage20_20260918

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

全部结果/审阅/分析及聚合入口：[NEW_SOURCE_RESULTS.json](../medtrace_stage20_20260918/NEW_SOURCE_RESULTS.json)

## E169 · medtrace_stage21_20260918

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage21 A机制分析（B待澄清）](../medtrace_stage21_20260918/STAGE21_ANALYSIS_ZH.md)。

原报告结论节选：**NOT SUPPORTED** — 并非逐指标成立。native同为45/45；旧H保持 FACT 2/4 (50.0%) vs NO_H 1/4 (25.0%)，旧U则 FACT 3/7 (42.9%) vs NO_H 6/7 (85.7%)。

全部结果/审阅/分析及聚合入口：[STAGE21_ANALYSIS_ZH.md](../medtrace_stage21_20260918/STAGE21_ANALYSIS_ZH.md) · [NO_H_PURE45_RESULTS.json](../medtrace_stage21_20260918/NO_H_PURE45_RESULTS.json)

## E170 · medtrace_stage21b_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage21B 完成分析](../medtrace_stage21b_20260919/STAGE21B_ANALYSIS_ZH.md)。

原报告结论节选：B 的45条编辑、当次写入与真实11/19/32/45前缀全部生成并评分，298项正式输出全部覆盖。两次机械回放另计，不当独立样本。

全部结果/审阅/分析及聚合入口：[STAGE21B_ANALYSIS_ZH.md](../medtrace_stage21b_20260919/STAGE21B_ANALYSIS_ZH.md) · [FACT_L31_RESULTS.json](../medtrace_stage21b_20260919/FACT_L31_RESULTS.json)

## E171 · medtrace_stage22_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage22C：近邻支持优于随机扩展，但未超过原支持](../medtrace_stage22_20260919/COVERAGE_ANALYSIS_ZH.md)。

原报告结论节选：2026-09-19。S1/S2各19个writer、各66条自然路由输出全部完成并评分，退出码0。固定Stage22B选出的H=.25/U=.01；S0精确复用E2的同配置原支持writer与相同DEV输出。所有有效臂均交付，未按中途成绩删除。最终按冻结门槛与同分时优先简单方案的规则，选择 **S0**，Stage23锁定E0/E1/E2三库。

全部结果/审阅/分析及聚合入口：[COVERAGE_ANALYSIS_ZH.md](../medtrace_stage22_20260919/COVERAGE_ANALYSIS_ZH.md) · [COVERAGE_PAIRED_RESULTS.json](../medtrace_stage22_20260919/COVERAGE_PAIRED_RESULTS.json) · [COVERAGE_RESULTS.json](../medtrace_stage22_20260919/COVERAGE_RESULTS.json) · [WEIGHT_SCREENING_RESULTS.json](../medtrace_stage22_20260919/WEIGHT_SCREENING_RESULTS.json)

## E172 · medtrace_stage23_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage23 结果、机制与证据边界](../medtrace_stage23_20260919/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：Stage23执行代码为 `d1ebfc9af388b088fd79bf66e3a64a9f253f4d30`，使用冻结最终锁 `6d93cd1bc574866ab1659185608044a754ccf63c8dbdcb68ef11469291e5dec9`。GPU正常退出0；控制器为 `23_COMPLETE_SCORED`，最高共同完整评分端点N=45。E0/E1复用经兼容审计的完整45 bank；

全部结果/审阅/分析及聚合入口：[FINAL_ANALYSIS_ZH.md](../medtrace_stage23_20260919/FINAL_ANALYSIS_ZH.md) · [PAIRED_SOURCE_RESULTS.json](../medtrace_stage23_20260919/PAIRED_SOURCE_RESULTS.json) · [RESULTS.json](../medtrace_stage23_20260919/RESULTS.json)

## E173 · medtrace_stage24_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage24审计、HSIC机械验证与条件停止报告](../medtrace_stage24_20260919/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：依据用户Stage24–25文件，使用Stage23公开c85df12和相应私有执行环境，在新服务器物理GPU0完成有界审计及机械检查。GPU UUID、各执行commit与耗费见账本，未启动旧任务，未重判历史答案，未追加数据、模型、七配置或超参数搜索。Stage24C需要先通过机械/成本门槛；Stage25还依赖DEV晋级、数据与资源，当前均未启动。

全部结果/审阅/分析及聚合入口：[FINAL_ANALYSIS_ZH.md](../medtrace_stage24_20260919/FINAL_ANALYSIS_ZH.md)

## E174 · medtrace_stage24c_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

全部结果/审阅/分析及聚合入口：[RESULTS.json](../medtrace_stage24c_20260919/RESULTS.json)

## E175 · medtrace_stage24d_20260920

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage24D 机制诊断收口](../medtrace_stage24d_20260920/FINAL_ANALYSIS_ZH.md)。

原报告结论节选：Stage24D 已完成，未启动 Stage24E 或 Stage25。固定起点为 `36baf4897c6f9b1d46fdc1f3f3efa3285fa70693`；RESULTS、RESOURCE_PROFILE、环境验收与R2修复验收逐文件核对通过，历史Stage22–24报告和负结果未改写。

全部结果/审阅/分析及聚合入口：[FINAL_ANALYSIS_ZH.md](../medtrace_stage24d_20260920/FINAL_ANALYSIS_ZH.md)

## E176 · medtrace_stage24e_20260920

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE_SCORED`。

全部结果/审阅/分析及聚合入口：[R1_RESULTS.json](../medtrace_stage24e_20260920/R1_RESULTS.json) · [R2_DECOMPOSITION_RESULTS.json](../medtrace_stage24e_20260920/R2_DECOMPOSITION_RESULTS.json) · [R2_RESULTS.json](../medtrace_stage24e_20260920/R2_RESULTS.json)

## E177 · medtrace_stage24e_r2_audit

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

全部结果/审阅/分析及聚合入口：[CPU_TEST_RESULTS.json](../medtrace_stage24e_r2_audit/CPU_TEST_RESULTS.json)

## E178 · medtrace_stage24e_r2a2

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`CPU_INTEGRATION_PASS_NOT_DEPLOYED`。

全部结果/审阅/分析及聚合入口：[CPU_INTEGRATION_TEST_RESULTS.json](../medtrace_stage24e_r2a2/CPU_INTEGRATION_TEST_RESULTS.json)

## E179 · medtrace_stage24r2_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`PASS`。

全部结果/审阅/分析及聚合入口：[CPU_VALIDATION_RESULTS.json](../medtrace_stage24r2_20260919/CPU_VALIDATION_RESULTS.json)

## E180 · medtrace_stage24r_20260919

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage24R：数值修复有恢复，验收未通过](../medtrace_stage24r_20260919/ANALYSIS_ZH.md)。

原报告结论节选：已执行包内补丁、现有PyTorch2.6 CPU测试、两例真实GPU分层诊断、3/20步机械更新、重复性与成本定位。未新增正式bank、未做语义评分、未启动24C/25。历史Stage20–24正式工件保持原样。

原聚合文件状态字段：`PASS`。

全部结果/审阅/分析及聚合入口：[ANALYSIS_ZH.md](../medtrace_stage24r_20260919/ANALYSIS_ZH.md) · [CPU_VALIDATION_RESULTS.json](../medtrace_stage24r_20260919/CPU_VALIDATION_RESULTS.json)

## E181 · medtrace_stage2_20260908

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage2 V2 review](../medtrace_stage2_20260908/GPT_PRO_REVIEW.md)。

原报告结论节选：Status: COMPLETE. BalancEdit old DEV16 16/16 judged; source candidates=16, frozen runnable new episodes N=16.

全部结果/审阅/分析及聚合入口：[BALANCEDIT_MATCHED_DEV_REPORT.md](../medtrace_stage2_20260908/BALANCEDIT_MATCHED_DEV_REPORT.md) · [GPT_PRO_REVIEW.md](../medtrace_stage2_20260908/GPT_PRO_REVIEW.md) · [NEW_EDIT_CONFIRMATION_REPORT.md](../medtrace_stage2_20260908/NEW_EDIT_CONFIRMATION_REPORT.md) · [SOURCE_PREPARATION_REPORT.md](../medtrace_stage2_20260908/SOURCE_PREPARATION_REPORT.md)

## E182 · medtrace_stage3_20260908

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage3 factual review](../medtrace_stage3_20260908/GPT_PRO_REVIEW.md)。

原报告结论节选：Status: PARTIAL; compute=COMPLETE_EXECUTABLE; Judge=COMPLETE_CURRENT_TUPLES; publication=PENDING. S1 = BE_ROUTE + P4-W1_KL_0.1; S0 = same route + P4-W0_TASK_ONLY; BE = native BalancEdit V4 adaptation.

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage3_20260908/GPT_PRO_REVIEW.md)

## E183 · medtrace_stage4_20260908

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage4 factual review](../medtrace_stage4_20260908/GPT_PRO_REVIEW.md)。

原报告结论节选：Status: COMPLETE_EXECUTABLE_CONFIRMATION_UNAVAILABLE. Judge missing=0. Publication state here is the generator-time snapshot; verify the Git commit for later public delivery.

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage4_20260908/GPT_PRO_REVIEW.md) · [UNSEEN_CONFIRMATION_REPORT.md](../medtrace_stage4_20260908/UNSEEN_CONFIRMATION_REPORT.md)

## E184 · medtrace_stage5_20260909

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage5 factual review](../medtrace_stage5_20260909/GPT_PRO_REVIEW.md)。

原报告结论节选：Status: COMPUTE_COMPLETE; publication tracked separately in PUBLICATION_STATUS.json when available.

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage5_20260909/GPT_PRO_REVIEW.md)

## E185 · medtrace_stage6_20260909

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE Stage6 fixed threshold transfer](../medtrace_stage6_20260909/GPT_PRO_REVIEW.md)。

原报告结论节选：Status: COMPUTE_COMPLETE

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage6_20260909/GPT_PRO_REVIEW.md)

## E186 · medtrace_stage7_20260909

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage7 APR development results](../medtrace_stage7_20260909/GPT_PRO_REVIEW.md)。

原报告结论节选：All 23 edits and 46 closed-form solves completed; Judge covered 1754/1754 tuples (1417 reused, 337 new), with all pipeline exit codes zero. Wall time was 1647.93 seconds (27.47 minutes); recorded GPU-hour upper bound was…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage7_20260909/GPT_PRO_REVIEW.md)

## E187 · medtrace_stage8_20260909

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage8 bounded APR closeout](../medtrace_stage8_20260909/GPT_PRO_REVIEW.md)。

原报告结论节选：**Decision: PARTIAL_TRADEOFF.** The fixed-radius repair does not resolve generalization versus locality. No further APR grid, new facts or automatic Stage9 is authorized or launched. Do not promote APR/TR as the default …

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage8_20260909/GPT_PRO_REVIEW.md)

## E188 · medtrace_stage9_20260909

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[Stage9 fact-contrast development results](../medtrace_stage9_20260909/GPT_PRO_REVIEW.md)。

原报告结论节选：Primary training/generation/Judge completed for the supported subset:15 SLAKE edit tasks x3 conditions =45 trajectories, each160 additional optimizer steps (7200 total). Eight planned edits/24 trajectories remain uniform…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_stage9_20260909/GPT_PRO_REVIEW.md)

## E189 · medtrace_structured_h8_v1_20261004

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[CP/Tucker H8 结构替换验证：最终审阅](../medtrace_structured_h8_v1_20261004/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：本轮完成48条320步续训、六个原146背景全库的prefix1/50/100/146评估，以及统一Qwen评分。结论为 **NOT_SUPPORTED**：当前证据不支持按冻结条件用TK_H1替换FREE_H1，也未显示Tucker相对CP具有更好的H效应。紧凑存储收益真实，但伴随本轮泛化性能下降。未追加训练、种子或参数搜索。

原聚合文件状态字段：`REPORTED_PUBLICATION_PENDING`。

全部结果/审阅/分析及聚合入口：[REPORT_STAT_RECOVERY_ZH.md](../medtrace_structured_h8_v1_20261004/REPORT_STAT_RECOVERY_ZH.md) · [SCIENTIFIC_REVIEW_ZH.md](../medtrace_structured_h8_v1_20261004/SCIENTIFIC_REVIEW_ZH.md) · [final_metrics/FINAL_REVIEW_ZH.md](../medtrace_structured_h8_v1_20261004/final_metrics/FINAL_REVIEW_ZH.md) · [final_metrics/PAIRED_RESULTS.json](../medtrace_structured_h8_v1_20261004/final_metrics/PAIRED_RESULTS.json)

## E190 · medtrace_textjoint_r2_20260925

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE TextJoint-R2 实验报告](../medtrace_textjoint_r2_20260925/REPORT_ZH.md)。

原报告结论节选：状态：COMPLETE；阶段：REPORT_READY。

全部结果/审阅/分析及聚合入口：[FINAL_REPORT_ZH.md](../medtrace_textjoint_r2_20260925/FINAL_REPORT_ZH.md) · [REPORT_ZH.md](../medtrace_textjoint_r2_20260925/REPORT_ZH.md) · [supplement/REPORT_ZH.md](../medtrace_textjoint_r2_20260925/supplement/REPORT_ZH.md) · [supplement/SUPPLEMENT_RESULTS.json](../medtrace_textjoint_r2_20260925/supplement/SUPPLEMENT_RESULTS.json)

## E191 · medtrace_textjoint_r3_20260926

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE TextJoint-R3 中文报告](../medtrace_textjoint_r3_20260926/FINAL_REPORT_ZH.md)。

原报告结论节选：核心矩阵与 A48 扩展已完成。结论：未验证到满足全部约束的联合收益。

全部结果/审阅/分析及聚合入口：[FINAL_REPORT_ZH.md](../medtrace_textjoint_r3_20260926/FINAL_REPORT_ZH.md)

## E192 · medtrace_textjoint_r4_20260926

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE TextJoint-R4 中文报告](../medtrace_textjoint_r4_20260926/FINAL_REPORT_ZH.md)。

原报告结论节选：训练、生成与可提交评分已结束。核心配对完整；两条新臂的 DEV sequential 压力指标存在 missing。正式联合工程目标未达到。

全部结果/审阅/分析及聚合入口：[FINAL_REPORT_ZH.md](../medtrace_textjoint_r4_20260926/FINAL_REPORT_ZH.md)

## E193 · medtrace_token_response_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[逐 token 保护约束：完成复核](../medtrace_token_response_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：冻结结论 **INCONCLUSIVE_SMALL_RESPONSE**。本轮不是保护成功：投影和等幅RAW的编辑NLL进展都为0，主保护KL也均为0；原RAW仍有0.00616349770490的平均正编辑进展，8/8专家为正。不能把停止写入造成的零扰动解释成在编辑过程中保住旧知识。

原聚合文件状态字段：`PASS`；`COMPLETE`；`INCONCLUSIVE_SMALL_RESPONSE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_token_response_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_token_response_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_token_response_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_token_response_v1_20261009/RESULTS.json)

## E194 · medtrace_tt_adam_direction_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定末段方向对照复盘](../medtrace_tt_adam_direction_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮出现编辑泛化和保护同时改善的点估计，但未达到预登记的可信开发信号标准，结论保持 **INCONCLUSIVE_POSITIVE_POINT_ONLY**。不能宣布方法成功、独立确认或医学保护有效。

原聚合文件状态字段：`COMPLETE`；`INCONCLUSIVE_POSITIVE_POINT_ONLY`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_adam_direction_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_adam_direction_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_adam_direction_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_adam_direction_v1_20261009/RESULTS.json)

## E195 · medtrace_tt_direction24_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定24开发复验：收益方向复现，证据仍未过晋级线](../medtrace_tt_direction24_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮已完整结束，结论为 **INCONCLUSIVE_POSITIVE_POINT_ONLY**；`eligible_for_preregistered146=false`。全部24和新增16的点安全线、保护点估计及生成上限检查均通过，但全部24的T2G编辑CI下界为0，不满足预登记的严格正下界。停止此方法的规模扩展，不启动146，也不修改原判据。

原聚合文件状态字段：`COMPLETE`；`INCONCLUSIVE_POSITIVE_POINT_ONLY`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_direction24_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_direction24_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_direction24_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_direction24_v1_20261009/RESULTS.json)

## E196 · medtrace_tt_direction_h_v1_20261006

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[TT 可行方向与 H 作用域：最终科学审阅](../medtrace_tt_direction_h_v1_20261006/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：P0、P1和依门槛选择的P2B已完成，80条续训均到320次尝试终点，评分无缺失。结论为 **NOT_CONFIRMED，不推进完整方法或自动扩展实验**。DIR_H主要保持已有编辑/泛化能力，未达到H学习门槛；提高中间秩的小试没有带来任务收益。P3因数据资格不足为BLOCKED_DATA，独立确认没有完成。

全部结果/审阅/分析及聚合入口：[MECHANICAL_TEST_REPORT.md](../medtrace_tt_direction_h_v1_20261006/MECHANICAL_TEST_REPORT.md) · [SCIENTIFIC_REVIEW_ZH.md](../medtrace_tt_direction_h_v1_20261006/SCIENTIFIC_REVIEW_ZH.md) · [FROZEN_W0_RESULTS.json](../medtrace_tt_direction_h_v1_20261006/FROZEN_W0_RESULTS.json) · [P0_RESULTS.json](../medtrace_tt_direction_h_v1_20261006/P0_RESULTS.json) · [P1_RESULTS.json](../medtrace_tt_direction_h_v1_20261006/P1_RESULTS.json) · [P2B_RESULTS.json](../medtrace_tt_direction_h_v1_20261006/P2B_RESULTS.json) · [PAIRED_RESULTS.json](../medtrace_tt_direction_h_v1_20261006/PAIRED_RESULTS.json) · [SOURCE_GROUP_RESULTS.json](../medtrace_tt_direction_h_v1_20261006/SOURCE_GROUP_RESULTS.json)

## E197 · medtrace_tt_functionmatch_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[固定方向/函数幅度交叉实验复盘](../medtrace_tt_functionmatch_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：执行已完成：16条历史轨迹最终四核均精确复现；16条新轨迹的5120次函数幅度匹配全部达到预注册精度，最大相对误差0.0009997393。18个GPU会话全部结束，220条新生成、7012消费者、1855条评分全部有效，无缺失、重试或运行失败。以下判定保留原预注册标准。

原聚合文件状态字段：`COMPLETE`；`INCONCLUSIVE_POSITIVE_POINT_ONLY`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_functionmatch_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_functionmatch_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_functionmatch_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_functionmatch_v1_20261009/RESULTS.json)

## E198 · medtrace_tt_initialwrite_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[初始训练顺序与末段优化：完成复盘](../medtrace_tt_initialwrite_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮主候选JOINT_220未达预登记标准，不晋级。训练和报告均终结；540步历史端点全部核精确复现，六卡历史输出检查通过，没有未解决的工程失败。

原聚合文件状态字段：`TERMINAL_WITH_MISSING`；`FAIL_EDIT_PRESERVATION_SCREEN`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_initialwrite_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_initialwrite_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_initialwrite_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_initialwrite_v1_20261009/RESULTS.json)

## E199 · medtrace_tt_mechanism_v1_20261009

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[同状态方向诊断与自然保护配对复盘](../medtrace_tt_mechanism_v1_20261009/REVIEW_ZH.md)。

原报告结论节选：本轮按预注册完成。8条RAW重建的最终四个TT核全部与历史端点逐元素一致；32个指定状态、128个候选写回和840次诊断前向齐全。6个GPU会话均正常结束，无运行失败、无重试、无新增答案生成或Judge。CPU读回复算了全部已报告中位比。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_mechanism_v1_20261009/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_mechanism_v1_20261009/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_mechanism_v1_20261009/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_mechanism_v1_20261009/RESULTS.json)

## E200 · medtrace_tt_minwrite_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[第二轮无回放机制对照复盘](../medtrace_tt_minwrite_v1_20261008/REVIEW_ZH.md)。

原报告结论节选：结论：拒绝晋级。native/FIT目标约束满足，并不足以保证未见编辑问题的泛化。主候选NATIVE_FIT的保护点估计改善伴随明显的T2G下降，未满足预登记-1pp开发筛选线。不能把训练可行性或CHECK改善称为方法成功。

原聚合文件状态字段：`TERMINAL_WITH_MISSING`；`FAIL_EDIT_PRESERVATION_SCREEN`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_minwrite_v1_20261008/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_minwrite_v1_20261008/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_minwrite_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_minwrite_v1_20261008/RESULTS.json)

## E201 · medtrace_tt_only_budget_h_v1_20261005

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[TT-only H8 三种子实验：科学审阅与交付](../medtrace_tt_only_budget_h_v1_20261005/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：本轮完成96次原生TT暖启动、240条320次外层尝试的续训及全部生成、评分和统计。TT84值得作为低参数开发候选继续考虑，TT44保留最低预算价值；本轮没有足够证据选择稳定优胜者，也不自动切换主线或追加实验。H带来的编辑退化没有通过增大TT参数量解决。保护更新保住了编辑指标，但主要代价是大量拒绝与极弱H学习，不能宣布完整方法成功。

原聚合文件状态字段：`REPORTED_PUBLICATION_PENDING`；`DESCRIPTIVE_PARETO_OR_INCONCLUSIVE_NO_AUTOMATIC_PROMOTION`。

全部结果/审阅/分析及聚合入口：[SCIENTIFIC_REVIEW_ZH.md](../medtrace_tt_only_budget_h_v1_20261005/SCIENTIFIC_REVIEW_ZH.md) · [results/REPORT_ZH.md](../medtrace_tt_only_budget_h_v1_20261005/results/REPORT_ZH.md) · [results/PAIRED_RESULTS.json](../medtrace_tt_only_budget_h_v1_20261005/results/PAIRED_RESULTS.json)

## E202 · medtrace_tt_scopedwrite_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[第一轮复盘：保持了W0指标，未产生ScopeWrite收益](../medtrace_tt_scopedwrite_v1_20261008/REVIEW_ZH.md)。

原报告结论节选：本轮已完成40条训练轨迹/7680更新、1942条新生成及13327个评分消费者。147条新增Astra medium载荷全部有效；共1887载荷中1885有效、2条历史继承缺失、0待提交/预约。六卡推理于北京时间2026-10-08 23:27:50完成，报告于23:31:10生成。源码与报告完成交付后，本轮机械执行结束；科学结论是未达到改进目标。

原聚合文件状态字段：`TERMINAL_WITH_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_tt_scopedwrite_v1_20261008/REPORT_ZH.md) · [REVIEW_ZH.md](../medtrace_tt_scopedwrite_v1_20261008/REVIEW_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_tt_scopedwrite_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_tt_scopedwrite_v1_20261008/RESULTS.json)

## E203 · medtrace_tt_svd_h8_v1_20261005

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[TT / SVD H8 八臂实验：最终审阅](../medtrace_tt_svd_h8_v1_20261005/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：64条320步续训、完整生成、Qwen评分和CPU报告均完成。236个完整评分键全部有效（175继承、61新attempt），804消费者完整绑定，0 missing。冻结判断为描述性H8证据，不自动选赢家或追加实验。

原聚合文件状态字段：`REPORTED_PUBLICATION_PENDING`；`DESCRIPTIVE_EXPOSED_H8_ONLY_NO_AUTOMATIC_PROMOTION`。

全部结果/审阅/分析及聚合入口：[SCIENTIFIC_REVIEW_ZH.md](../medtrace_tt_svd_h8_v1_20261005/SCIENTIFIC_REVIEW_ZH.md) · [final_metrics/REPORT_ZH.md](../medtrace_tt_svd_h8_v1_20261005/final_metrics/REPORT_ZH.md) · [final_metrics/PAIRED_RESULTS.json](../medtrace_tt_svd_h8_v1_20261005/final_metrics/PAIRED_RESULTS.json)

## E204 · medtrace_tt_u_continuation_v1_20261006

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[TT-U 续训价值与保护收益：科学报告](../medtrace_tt_u_continuation_v1_20261006/SCIENTIFIC_REVIEW_ZH.md)。

原报告结论节选：本轮为固定开发实验，不是独立确认。W0是已经完成编辑的TT专家；CE-only与带U条件是不同必要对照。所有节点及失败/缺失完整保留。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[SCIENTIFIC_REVIEW_ZH.md](../medtrace_tt_u_continuation_v1_20261006/SCIENTIFIC_REVIEW_ZH.md) · [BANK_RESULTS.json](../medtrace_tt_u_continuation_v1_20261006/BANK_RESULTS.json) · [P1_RESULTS.json](../medtrace_tt_u_continuation_v1_20261006/P1_RESULTS.json) · [P2_RESULTS.json](../medtrace_tt_u_continuation_v1_20261006/P2_RESULTS.json)

## E205 · medtrace_type_guard_v1_20261008

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[问答类型 OFF 保护：完成，但不替换主方法](../medtrace_type_guard_v1_20261008/REPORT_ZH.md)。

原报告结论节选：明确问答类型不匹配时回退 Base，在历史 T2L 上从 62.366% 提升到 100%，T0/T1G/T2G 保持 100% / [99.144%,99.315%] / 93.322%。四条 CHECK 的 Base token 一致率从 0 提升到 50%，KL 从 2.231 降到 1.809；两条开放式 CHECK 仍受干扰。

原聚合文件状态字段：`COMPLETE_NOT_PROMOTED`；`TERMINAL_WITH_INHERITED_MISSING`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../medtrace_type_guard_v1_20261008/REPORT_ZH.md) · [COMPLETION_AUDIT.json](../medtrace_type_guard_v1_20261008/COMPLETION_AUDIT.json) · [RESULTS.json](../medtrace_type_guard_v1_20261008/RESULTS.json)

## E206 · medtrace_visual_verifier_recovery_20260907T045800Z

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[GPT Pro review: frozen-C1 visual-verifier recovery](../medtrace_visual_verifier_recovery_20260907T045800Z/GPT_PRO_REVIEW.md)。

原报告结论节选：Computation and evaluation are complete: 21 edit/seed tasks, 42 verifier fits closed. Seven source facts are repeated across three seeds. This is viewed development diagnosis, not full TIME/MedTRACE or clinical validatio…

全部结果/审阅/分析及聚合入口：[GPT_PRO_REVIEW.md](../medtrace_visual_verifier_recovery_20260907T045800Z/GPT_PRO_REVIEW.md) · [ROUTER_PAIRED_E2E_REPORT.md](../medtrace_visual_verifier_recovery_20260907T045800Z/ROUTER_PAIRED_E2E_REPORT.md) · [ROUTER_TRAINING_AND_PARAMETER_REPORT.md](../medtrace_visual_verifier_recovery_20260907T045800Z/ROUTER_TRAINING_AND_PARAMETER_REPORT.md) · [STARTUP_RECOVERY_REPORT.md](../medtrace_visual_verifier_recovery_20260907T045800Z/STARTUP_RECOVERY_REPORT.md)

## E207 · medtrace_wording_pair_v1_20261015

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[前20例问法配对诊断：完成，未达到预定改善标准](../medtrace_wording_pair_v1_20261015/REPORT_ZH.md)。

原报告结论节选：本轮234份新输出和234个消费者评分全部有效、0缺失。将原同图T2G问句改成保留任务与限定的直接请求后，M13宏平均提高5.0个百分点，FISHER_MIN提高7.5个百分点；两者均低于预定10个百分点门槛，99%配对区间均跨0。不能据此认定问法是当前T2G低分的主要原因，也不能据此证明问法没有影响。

原聚合文件状态字段：`COMPLETE_READONLY_POSTHOC_ANALYSIS`；`COMPLETE_VALID`。

全部结果/审阅/分析及聚合入口：[CROSS78_REPORT_ZH.md](../medtrace_wording_pair_v1_20261015/CROSS78_REPORT_ZH.md) · [REPORT_ZH.md](../medtrace_wording_pair_v1_20261015/REPORT_ZH.md) · [CROSS78_RESULTS.json](../medtrace_wording_pair_v1_20261015/CROSS78_RESULTS.json) · [EXECUTION_AUDIT.json](../medtrace_wording_pair_v1_20261015/EXECUTION_AUDIT.json) · [RESULTS.json](../medtrace_wording_pair_v1_20261015/RESULTS.json)

## E208 · method_decision_v79_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V79 方法定型：精确复现通过，未建立 TT 相对 LoRA 的优势](../method_decision_v79_20261023/REPORT_ZH.md)。

原报告结论节选：本有限计划已完成。V69 的10条160步训练曲线及全部219个输出token和绑定精确复现；相同训练信息的归一化rank1 LoRA完成后，T2G为36/40，V69为37/40，原生LoRA也是37/40。V69与匹配LoRA均未通过原联合门槛。现有证据支持实现可复现，但不足以支持TT优于LoRA或医学泛化/保护优势。没有追加候选、门控训练或146扩展。

原聚合文件状态字段：`FINITE_PLAN_COMPLETE_NO_JOINT_PROMOTION`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../method_decision_v79_20261023/REPORT_ZH.md) · [RESULTS.json](../method_decision_v79_20261023/RESULTS.json)

## E209 · multiroute_full_v86_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V86五臂完整实验结果](../multiroute_full_v86_20261023/REPORT_ZH.md)。

原报告结论节选：固定四核TT相对曲率迁移未得到净泛化提升；终态零空间修正改善了Fisher代理，但没有改变任何自由回答。简单取消保护项U有2题开发集增益，仍未达到预登记继续门槛。本轮不扩大半径、扫描系数或按结果挑编辑/seed。

原聚合文件状态字段：`ALL_REGISTERED_CONSUMERS_AND_LIFECYCLE_COMPLETE`；`SCORED_ALL_775`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../multiroute_full_v86_20261023/REPORT_ZH.md) · [FINAL_EXECUTION_AUDIT.json](../multiroute_full_v86_20261023/FINAL_EXECUTION_AUDIT.json) · [RESULTS.json](../multiroute_full_v86_20261023/RESULTS.json)

## E210 · multiroute_v85_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V85非教师机制实验结果](../multiroute_v85_20261023/REPORT_ZH.md)。

原报告结论节选：2026-10-23，完整16/16格完成，未选择编辑或锚点。此次实验在多路线队列中独立执行，不以教师资格为前置。

原聚合文件状态字段：`COMPLETED_LOCAL_PROTECTION_TRADEOFF`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../multiroute_v85_20261023/REPORT_ZH.md) · [RESULTS.json](../multiroute_v85_20261023/RESULTS.json)

## E211 · noncanonical_mechanisms_20261023

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[noncanonical_mechanisms_20261023](../noncanonical_mechanisms_20261023/)。

## E212 · norm_pair_scope_v114_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V114：范数加成对坐标仍有一个整组留一错误](../norm_pair_scope_v114_20261025/REPORT_ZH.md)。

原报告结论节选：按预登记只增加范数通道、保留原abs坐标，λ=1、符号0阈值、来源组/类别平衡权重和8折整组留一保持。58判别错1条（T06），较V113的3条减少，但严格零错误准入仍FAIL。最终head未拟合、额外6组未评价、没有缓存输出重放；不按该错误调λ或阈值。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../norm_pair_scope_v114_20261025/REPORT_ZH.md) · [RESULTS.json](../norm_pair_scope_v114_20261025/RESULTS.json)

## E213 · output_init_v69_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V69：输出侧初始化提高泛化，但保护联合门槛未通过](../output_init_v69_20261023/REPORT_ZH.md)。

原报告结论节选：`OUTPUT_INITIALIZATION_NOT_PROMOTED`。全部10编辑、219新正式输出完成。T2G为37/40，相对V64 A净增4题、分布于3个编辑；相对B增加2题，原B的35题全部保留。这是曝光DEV上的局部正向信号，不是整体通过或独立确认。

原聚合文件状态字段：`OUTPUT_INITIALIZATION_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../output_init_v69_20261023/REPORT_ZH.md) · [RESULTS.json](../output_init_v69_20261023/RESULTS.json)

## E214 · owner5_autonomous_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[Owner 5 覆盖与轨迹诊断结果](../owner5_autonomous_20261018/REPORT_ZH.md)。

原报告结论节选：本轮两个预注册诊断均已完成，仍未获得增量语义保护收益的证据。三个保留负例不是可用的Base正确保护分母；第40、80、120步的构造负例也未显示语义损伤。不能把本结果解释为方法普遍无效、保护有效或整条训练路径无损伤。

原聚合文件状态字段：`TWO_PHASES_AND_SUCCESSOR_PANEL96_COMPLETE_ALL_OWNED_PROCESSES_ENDED`；`NO_INCREMENTAL_SEMANTIC_PROTECTION_BENEFIT_OBSERVED_ON_FIXED96`；`NO_INCREMENTAL_SEMANTIC_PROTECTION_EVIDENCE`。

全部结果/审阅/分析及聚合入口：[PANEL96_REPORT_ZH.md](../owner5_autonomous_20261018/PANEL96_REPORT_ZH.md) · [REPORT_ZH.md](../owner5_autonomous_20261018/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../owner5_autonomous_20261018/EXECUTION_AUDIT.json) · [PANEL96_RESULTS.json](../owner5_autonomous_20261018/PANEL96_RESULTS.json) · [RESULTS.json](../owner5_autonomous_20261018/RESULTS.json)

## E215 · owner5_ct_expand_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[同模态跨图正检查补充：数据准入未通过](../owner5_ct_expand_v1_20261018/REPORT_ZH.md)。

原报告结论节选：本轮已在物理GPU0/1完整执行冻结候选的Base资格检查。13个新增CT正检查全部被Base正确回答，0个明确错误，因此未满足“至少1个Base原错同模态跨图正检查”的预定门槛。六组对照未获准入、未启动、训练更新为0。这是有限候选补充的阴性结果，不能据此判断语义保护有效或无效。

原聚合文件状态字段：`DATA_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../owner5_ct_expand_v1_20261018/REPORT_ZH.md) · [RESULTS.json](../owner5_ct_expand_v1_20261018/RESULTS.json)

## E216 · owner5_mechanism31_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[31步机制审计与同模态准入：终态结果](../owner5_mechanism31_v1_20261018/REPORT_ZH.md)。

原报告结论节选：**机制审计通过；数据联合准入未通过，六组对照没有启动。** 本轮回答了保护梯度是否真正影响更新，并补齐了三个当前Base正确的同模态CT负例；但三个可取得原图的CT正检查在Base下全部已经正确，缺少预注册要求的新增跨图迁移支持。不能据此宣称保护有效或六组方法失败。

原聚合文件状态字段：`TERMINAL_MECHANISM_PASS_DATA_GATE_NOT_MET`；`MECHANISM_PASS_DATA_GATE_NOT_MET`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../owner5_mechanism31_v1_20261018/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../owner5_mechanism31_v1_20261018/EXECUTION_AUDIT.json) · [RESULTS.json](../owner5_mechanism31_v1_20261018/RESULTS.json)

## E217 · owner5_protection_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[Owner 5 固定两臂机制诊断结果](../owner5_protection_v1_20261018/REVIEW_ZH.md)。

原报告结论节选：结论：无新增保护控制在两个预定负例上均无损伤，未显示新增保护收益。保护臂满足构造NLL约束，却几乎阻断编辑：原native/FIT仅2/5正确，控制臂和原M13P160参考均5/5。两个跨图正例在未编辑Base下已正确，不能把训练后答对解释成新增迁移收益。本轮结束，不自动扩展或调参。

原聚合文件状态字段：`COMPLETE_NO_ADDITIONAL_PROTECTION_BENEFIT_AND_EDIT_FAILURE`。

全部结果/审阅/分析及聚合入口：[REVIEW_ZH.md](../owner5_protection_v1_20261018/REVIEW_ZH.md) · [RESULTS.json](../owner5_protection_v1_20261018/RESULTS.json)

## E218 · owner5_semantic_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[Owner 5 语义保护最小对照结果](../owner5_semantic_v1_20261018/REPORT_ZH.md)。

原报告结论节选：两组均完成160步并通过语义验收，但没有观察到额外的语义保护收益。Control与Semantic-Protected的全部9条最终回答逐token一致，且均符合固定参考结论。独立负检查上Control本来就没有受损，因此本轮不能证明新保护项避免了损伤。

原聚合文件状态字段：`NO_INCREMENTAL_SEMANTIC_PROTECTION_BENEFIT_OBSERVED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../owner5_semantic_v1_20261018/REPORT_ZH.md) · [EXECUTION_AUDIT.json](../owner5_semantic_v1_20261018/EXECUTION_AUDIT.json) · [RESULTS.json](../owner5_semantic_v1_20261018/RESULTS.json)

## E219 · owner5_token_audit_v1_20261018

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[构造负例逐 token 损失与 Yes/No margin 补证](../owner5_token_audit_v1_20261018/REPORT_ZH.md)。

原报告结论节选：本次只测同一构造负例上的原生 Base 与重建零专家 Base。两者三个预测位置的全部 32000 维 logits 逐元素相同，labels 相同；平均 NLL 均为 5.913671016693115，与历史 guard Base 完全一致。此结论仅覆盖本输入与三个 teacher-forcing 位置。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../owner5_token_audit_v1_20261018/REPORT_ZH.md) · [RESULTS.json](../owner5_token_audit_v1_20261018/RESULTS.json)

## E220 · pair_scope_v113_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V113：成对线性判别未通过整来源组留一](../pair_scope_v113_20261025/REPORT_ZH.md)。

原报告结论节选：固定原单位置Base特征、abs成对差异、λ=1、来源组/类别平衡权重和符号阈值0，按预登记完成8折整来源组留一。58个判别出现3个错误，分布于T01、T06、T07；其余5组无错。记录PAIR_SCOPE_GROUP_CV_FAILED，未拟合最终head、未评价额外6组、未执行缓存输出重放，后继语义结果NA。没有根据错误调整本轮λ或阈值。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../pair_scope_v113_20261025/REPORT_ZH.md) · [RESULTS.json](../pair_scope_v113_20261025/RESULTS.json)

## E221 · path_extension_v119_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V119：格式扩展增大保护基，但没有回答收益](../path_extension_v119_20261025/REPORT_ZH.md)。

原报告结论节选：预注册一次统一问题前缀的扩展已完成：28次Base格式生成、182次固定终态候选生成，全部210次EOS、0cap，0更新/反向/Judge。14个父TT终态在生成前后逐元素相同；全部扩展Q保存加载通过，在全部Q冻结后才开始候选评价。guard和worker正常退出，没有重训、重缩放或试第二种格式。

原聚合文件状态字段：`COMPLETE_DEVELOPMENT_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../path_extension_v119_20261025/REPORT_ZH.md) · [RESULTS.json](../path_extension_v119_20261025/RESULTS.json)

## E222 · path_null_v118_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V118：恢复5条保护损伤，严格推进门仍失败](../path_null_v118_20261025/REPORT_ZH.md)。

原报告结论节选：两臂28条训练轨迹和931次生成完成，全部终态保存加载通过，所有训练先冻结再评价，guard与worker正常退出。PATH_NULL保持目标来源70/70及目标未训练表达28/28，并将56条保护新表达中Base原正确身份的损伤从6条降至1条，恢复5条、相对对照无新增损伤；但预登记的**相对Base零损伤**门失败，不晋级。不能用平均增益或构造内全保持替代该门。

原聚合文件状态字段：`COMPLETE_DEVELOPMENT_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../path_null_v118_20261025/REPORT_ZH.md) · [RESULTS.json](../path_null_v118_20261025/RESULTS.json)

## E223 · prospective_confirmation_v111_20261025

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[prospective_confirmation_v111_20261025](../prospective_confirmation_v111_20261025/)。

## E224 · prospective_source_v95_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V95：新来源候选核查与临床片段加权准入结果](../prospective_source_v95_20261024/REPORT_ZH.md)。

原报告结论节选：本次固定候选未达到临床片段加权的事前可识别条件，未启动GPU、训练或生成。它是数据与机制准入失败，不能作为该方法的效果负结果，也不能据此否定所有医学结构监督。

原聚合文件状态字段：`CLOSED_IDENTIFIABILITY_ADMISSION_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../prospective_source_v95_20261024/REPORT_ZH.md) · [RESULTS.json](../prospective_source_v95_20261024/RESULTS.json)

## E225 · protection_admission_v98_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V98：找到操作层面保护事实，配对辨别门槛仍未通过](../protection_admission_v98_20261024/REPORT_ZH.md)。

原报告结论节选：依用户要求，先冻结新协议，再完成 30 名患者、15 组配对、90 问的 Base 资格验证。**找到 9 组“持续侧别错误＋稳定正确阴性保护”的操作层面候选；通过预设阳性辨别检查的较强候选为 0。** 因此未启动方法训练，未扩样、换问法或换保护事实。

原聚合文件状态字段：`TASK_NOT_ADMITTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../protection_admission_v98_20261024/REPORT_ZH.md) · [RESULTS.json](../protection_admission_v98_20261024/RESULTS.json)

## E226 · protection_reachability_audit_v127_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V127终态：保护请求梯度未被Q归零，剩余失败原因仍未确定](../protection_reachability_audit_v127_20261025/REPORT_ZH.md)。

原报告结论节选：全部84个保护TRAIN请求、14个终态模型完成Q与EMPTY_Q只读比较。308次语言forward/真实损失反向、14次Fisher算子反向、308次视觉forward，0更新/生成/Judge；有限性、缓存计数、参考一阶界和权重前后逐元素一致全部通过，worker/guard退出0。未使用目标held、保护挑战或封存8组。

原聚合文件状态字段：`PROTECTION_REACHABILITY_AUDIT_COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../protection_reachability_audit_v127_20261025/REPORT_ZH.md) · [RESULTS.json](../protection_reachability_audit_v127_20261025/RESULTS.json)

## E227 · protection_source_fidelity_v128_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V128终态：未发现保护TRAIN改写引入明确语义范围变化](../protection_source_fidelity_v128_20261025/REPORT_ZH.md)。

原报告结论节选：28个原问/参考/两个旧改写三联已全部读完，56条改写均保持原问的事实关系、量词和图像范围：EQUIVALENT56、SCOPE_CHANGED0、UNDETERMINED0。28个原问题是恒等支持，不能把它们再算成28次独立语义审阅。图像与参考绑定一致性、56唯一表达映射通过；先冻结文本判断，再映射既有TRAIN正确性。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../protection_source_fidelity_v128_20261025/REPORT_ZH.md) · [RESULTS.json](../protection_source_fidelity_v128_20261025/RESULTS.json)

## E228 · question_mean_scope_v112_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V112：实际问题范围均值未通过来源几何准入](../question_mean_scope_v112_20261025/REPORT_ZH.md)。

原报告结论节选：预登记只改变Base同层读出位置：实际问题token的原始激活均值、CPU double及L2归一；最大负例约束半径、40来源留一正例、18不同谓词负例和全部后继门保持。真实前缀中参考替换不改变inputs_embeds/attention/position，问题范围通过实际换行上下文匹配与多模态位置偏移检查，排除图像patch、提示控制token与答案。58个校准输入全部完成，无工程失败。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../question_mean_scope_v112_20261025/REPORT_ZH.md) · [RESULTS.json](../question_mean_scope_v112_20261025/RESULTS.json)

## E229 · rankjb_v90_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[Rank/JB V90：固定参考前缀蒸馏的完整实验](../rankjb_v90_20261023/REPORT_ZH.md)。

原报告结论节选：Rank缓解了均匀反向KL的损伤，但没有超过CE。T2G：CE25/32，RKL7/32，RANK19/32，JB5/32。RANK相对RKL净增12题、跨5编辑，满足预登记的相对开发门槛；同时相对匹配CE是0胜6负。因此不能将相对受损对照的恢复写成主线增益，当前结果不支持直接扩大该蒸馏方案。

原聚合文件状态字段：`SCORED_ALL_775`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../rankjb_v90_20261023/REPORT_ZH.md) · [RESULTS.json](../rankjb_v90_20261023/RESULTS.json)

## E230 · reference_eos_gradient_audit_v125_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V125终态：参考与结束监督局部冲突，达到优先级机制试验准入门](../reference_eos_gradient_audit_v125_20261025/REPORT_ZH.md)。

原报告结论节选：全部28个固定状态完成，448次真实语言forward/损失反向、28次算子反向、308次视觉forward；0更新/生成/Judge。权重前后逐元素相同、有限性、缓存计数及数学一阶不等式全部通过，worker/guard退出0。仅使用70目标TRAIN、同70个已冻结prefix上下文及84保护TRAIN，未使用目标held、保护挑战或封存新8组作梯度或准入判断。

原聚合文件状态字段：`REFERENCE_EOS_AUDIT_COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../reference_eos_gradient_audit_v125_20261025/REPORT_ZH.md) · [RESULTS.json](../reference_eos_gradient_audit_v125_20261025/RESULTS.json)

## E231 · reference_priority_completion_v126_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V126终态：98/98完整目标与零触顶，保护门仍失败](../reference_priority_completion_v126_20261025/REPORT_ZH.md)。

原报告结论节选：全部14×160新训练和252新生成完成，worker/guard退出0。98条目标核心和完整支持均全正确、252条全部EOS且0触顶；固定同权重的完整参考优先机制恢复了V124的目标事实损失，并覆盖了V122未完整结束的7条目标。但保护训练81/84核心正确、保护挑战完整48/56低于固定49门，**本轮仍无候选晋级**。不降低保护门，不打开封存新来源。

原聚合文件状态字段：`COMPLETE_DEVELOPMENT_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../reference_priority_completion_v126_20261025/REPORT_ZH.md) · [RESULTS.json](../reference_priority_completion_v126_20261025/RESULTS.json)

## E232 · relation_init_v96_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V96 完整结论：BAL未超过CE，当前面板不能证明新医学纠错能力](../relation_init_v96_20261024/REPORT_ZH.md)。

原报告结论节选：V96已完成资格、CPU机制核验、三臂训练、全部生成及全文评分。TT_BAL和TT_CE在原问、FIT、T2G及其他关系的汇总结果完全相同，八组T2G配对差值均为零；预注册的至少+3/32门槛失败。关闭本固定均衡假设，不追加权重、种子、rank或确认组搜索。CR-TTI因固定数据准入失败而关闭，不能用CPU PASS替代真实保护证据。

原聚合文件状态字段：`COMPLETE_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[LAUNCH_REPORT_ZH.md](../relation_init_v96_20261024/LAUNCH_REPORT_ZH.md) · [REPORT_ZH.md](../relation_init_v96_20261024/REPORT_ZH.md) · [RESULTS.json](../relation_init_v96_20261024/RESULTS.json)

## E233 · relation_scope_v109_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V109：范围门保住28保护事实，但误拒两个未训练表达，晋级失败](../relation_scope_v109_20261025/REPORT_ZH.md)。

原报告结论节选：预登记的58个校准点确实可分：40个来源留一正例最大距离0.5164723，18个不同事实谓词负例最小距离0.7942818，间隔0.2778096，固定中点半径0.6553771。半径在读取其余输入的范围决定之前固定，未按回答调整。所有126查询的路由冻结后完成140次完整生成，0更新/0反向，正常退出。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../relation_scope_v109_20261025/REPORT_ZH.md) · [RESULTS.json](../relation_scope_v109_20261025/RESULTS.json)

## E234 · relation_source_v97_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V97 执行结果：发现真实侧别错误，但异关系保护未准入](../relation_source_v97_20261024/REPORT_ZH.md)。

原报告结论节选：已经完成新的来源审查、候选冻结和一次 Base 资格推理。此次不再使用 V96 的 MRI 简洁度面板，转向项目已有 PadChest-GR 的报告支持侧别事实。结果是 **20 人中 0 个满足预先规定的纠错与双保护组合**，因此没有启动 CR-TTI 或其他参数训练，也没有补抽病例。

原聚合文件状态字段：`CLOSED_NO_QUALIFIED_CORRECTION_PROTECTION_PAIRS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../relation_source_v97_20261024/REPORT_ZH.md) · [RESULTS.json](../relation_source_v97_20261024/RESULTS.json)

## E235 · replay_gradient_audit_v121_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V121：辅助梯度局部冲突门通过，不是目标失败的因果证明](../replay_gradient_audit_v121_20261025/REPORT_ZH.md)。

原报告结论节选：固定14任务、两臂、初始化/实际160步终态的56个状态全部完成。616次实际teacher-forcing forward/损失反向、56次TT算子反向、308视觉forward，0更新/生成/Judge；所有权重前后逐元素不变，梯度有限、缓存计数和一阶数值不等式通过。guard与worker均正常退出。

原聚合文件状态字段：`GRADIENT_AUDIT_COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../replay_gradient_audit_v121_20261025/REPORT_ZH.md) · [RESULTS.json](../replay_gradient_audit_v121_20261025/RESULTS.json)

## E236 · request_state_v63_20261022

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

材料目录：[request_state_v63_20261022](../request_state_v63_20261022/)。

## E237 · scope_information_audit_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_information_audit_20261001/FINAL_EXECUTION_AUDIT.json)

## E238 · scope_m3bench_candidate_audit_20261001

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[私有 scope 审核指南](../scope_m3bench_candidate_audit_20261001/REVIEW_GUIDE_ZH.md)。

原报告结论节选：审核单位为(edit, image, question)，医学答案和edit适用关系分别记录。图片/QA/病例链接只在合法私有环境查看，不上传公共仓库。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[REVIEW_GUIDE_ZH.md](../scope_m3bench_candidate_audit_20261001/REVIEW_GUIDE_ZH.md) · [FINAL_EXECUTION_AUDIT.json](../scope_m3bench_candidate_audit_20261001/FINAL_EXECUTION_AUDIT.json)

## E239 · scope_m3bench_exposure_audit_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_m3bench_exposure_audit_20261001/FINAL_EXECUTION_AUDIT.json)

## E240 · scope_m3bench_review_packet_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_m3bench_review_packet_20261001/FINAL_EXECUTION_AUDIT.json)

## E241 · scope_positive_guard_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`CLOSED_NOT_ADMITTED`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_positive_guard_20261001/FINAL_EXECUTION_AUDIT.json)

## E242 · scope_safe_residual_20260929

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`EXECUTED_MATRIX_COMPLETE`；`COMPLETE`。

全部结果/审阅/分析及聚合入口：[CORE_MATRIX_RESULTS.json](../scope_safe_residual_20260929/CORE_MATRIX_RESULTS.json) · [FINAL_EXECUTION_AUDIT.json](../scope_safe_residual_20260929/FINAL_EXECUTION_AUDIT.json) · [HOLDOUT_RESULTS.json](../scope_safe_residual_20260929/HOLDOUT_RESULTS.json)

## E243 · scope_safe_router_20260930

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[实现自审](../scope_safe_router_20260930/IMPLEMENTATION_REVIEW_ZH.md)。

原报告结论节选：沿用冻结 source/runtime、H expert tensor、R0 router、Base输入绑定、生成配置及Sol Judge rubric。没有训练/损失/optimizer/gradient surgery入口；推理worker禁止backward，expert requires_grad=False。

原聚合文件状态字段：`CLOSED_NOT_ADMITTED`。

全部结果/审阅/分析及聚合入口：[IMPLEMENTATION_REVIEW_ZH.md](../scope_safe_router_20260930/IMPLEMENTATION_REVIEW_ZH.md) · [FINAL_EXECUTION_AUDIT.json](../scope_safe_router_20260930/FINAL_EXECUTION_AUDIT.json)

## E244 · scope_selective_anchor_20260930

历史MedTRACE与前驱材料；不并入当前TT配方。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`CLOSED_NOT_ADMITTED`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_selective_anchor_20260930/FINAL_EXECUTION_AUDIT.json)

## E245 · scope_text_guard_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_text_guard_20261001/FINAL_EXECUTION_AUDIT.json)

## E246 · scope_text_margin_failure_audit_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_text_margin_failure_audit_20261001/FINAL_EXECUTION_AUDIT.json)

## E247 · scope_text_margin_robustness_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_text_margin_robustness_20261001/FINAL_EXECUTION_AUDIT.json)

## E248 · scope_text_transfer_audit_20261001

无叙述性结果入口；按JSON/方案区分，不能补造性能结论。

未找到叙述性结果报告；存在方案或其他材料不等于性能实验已经执行。

原聚合文件状态字段：`COMPLETE`。

全部结果/审阅/分析及聚合入口：[FINAL_EXECUTION_AUDIT.json](../scope_text_transfer_audit_20261001/FINAL_EXECUTION_AUDIT.json)

## E249 · scro_v88_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[SCRO V88：完整方法训练与自由回答结果](../scro_v88_20261023/REPORT_ZH.md)。

原报告结论节选：固定 TT 迁移没有增益：SCRO 的 T2G 为13/32，普通Frobenius压缩24/32，直接CE25/32。SCRO相对压缩对照0胜11负，相对CE0胜12负，不满足预登记继续门槛。本冻结版本结束，不追加半径、谱阈值或seed搜索。

原聚合文件状态字段：`SCORED_ALL_620`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../scro_v88_20261023/REPORT_ZH.md) · [RESULTS.json](../scro_v88_20261023/RESULTS.json)

## E250 · semantic_scope_v116_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V116：文字关系判别校准通过，仍拒绝一个源改写，缓存评价未准入](../semantic_scope_v116_20261025/REPORT_ZH.md)。

原报告结论节选：固定现有冻结LLaVA-Med/Mistral、文字问题比较提示、0/1强制选项、分差>0规则，医学参考/答案/图像字段从实际输入删除。原DEV8的58对全正确，其中8个原题全文相同为定义命中、32个非同文字源表达及18个不同谓词负例为50次真实模型forward。未训练模型或按此面板改提示。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../semantic_scope_v116_20261025/REPORT_ZH.md)

## E251 · sgb_identifiable_v91_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[SGB来源秩修复V91：完整训练与严格自由回答结果](../sgb_identifiable_v91_20261023/REPORT_ZH.md)。

原报告结论节选：来源秩修复仍未带来主线增益：共享对照SHARED的T2G为23/32，IDENT_ZERO18/32，IDENT_LS21/32。IDENT_LS相对IDENT_ZERO净增3题、跨3编辑，满足这个次要相对比较的预登记开发门槛；但相对共享对照仍0胜2负，主比较IDENT_ZERO相对共享0胜5负。因此不能以弱对照上的恢复宣称方法优越，本固定SGB分支不扩大，不追加谱阈值、初始化或seed搜索。

原聚合文件状态字段：`SCORED_ALL_620`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../sgb_identifiable_v91_20261023/REPORT_ZH.md) · [RESULTS.json](../sgb_identifiable_v91_20261023/RESULTS.json)

## E252 · sgb_v89_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V89谱分支完整训练与评分结果](../sgb_v89_20261023/REPORT_ZH.md)。

原报告结论节选：固定有符号输入路由迁移失败：相对同标称容量共享分支，T2G净损13/32，FIT损3/32。结果不支持本实现的增量收益，不外推原论文或所有有符号组合。

原聚合文件状态字段：`SCORED_ALL_465`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../sgb_v89_20261023/REPORT_ZH.md) · [RESULTS.json](../sgb_v89_20261023/RESULTS.json)

## E253 · source_eos_audit_v123_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V123终态：参考EOS拟合与自然前缀结束之间有差异，但不是触顶专属证据](../source_eos_audit_v123_20261025/REPORT_ZH.md)。

原报告结论节选：70条固定目标TRAIN全部完成，70个字面完整参考边界全部匹配；TRAINED_Q重算下一token与原实际生成70/70精确一致。三条件共420语言forward、140视觉forward，0反向/更新/生成/Judge，TT运行前后逐元素不变，worker/guard退出0。未读取28目标held、84保护、56挑战或封存8组作为本轮输入。

原聚合文件状态字段：`EOS_AUDIT_COMPLETE`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_eos_audit_v123_20261025/REPORT_ZH.md) · [RESULTS.json](../source_eos_audit_v123_20261025/RESULTS.json)

## E254 · source_independence_audit_20261021

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[既有来源病例独立性与第二关系审计](../source_independence_audit_20261021/REPORT_ZH.md)。

原报告结论节选：2026-10-21。结论：现有证据尚不足以冻结并启动新面板；本次新增模型调用、训练、生成和 Judge 均为零。保留 JOINT 默认及 V30/V31 回归负结果，不继续围绕原六例调参。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_independence_audit_20261021/REPORT_ZH.md)

## E255 · source_init_v66_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V66：源前缀初始化未改善泛化](../source_init_v66_20261023/REPORT_ZH.md)。

原报告结论节选：`SOURCE_INITIALIZATION_NOT_PROMOTED`。完整10编辑、219个新消费者：T2G为31/40，低于V64 A的33/40和B的35/40；相对B丢5题、增1题。T0=10/10、T1G=40/40、改写40/40、T1L=33/84均保持，但T2L从1/5降至0/5，损失唯一Base原正确题。无触顶输出。联合门槛失败，关闭此固定初始化候选，不搜索rank/seed/方向。

原聚合文件状态字段：`SOURCE_INITIALIZATION_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_init_v66_20261023/REPORT_ZH.md) · [RESULTS.json](../source_init_v66_20261023/RESULTS.json)

## E256 · source_prefix_completion_v124_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V124终态：零触顶伴随严重目标事实损失，无候选晋级](../source_prefix_completion_v124_20261025/REPORT_ZH.md)。

原报告结论节选：全部14×160训练和252次新生成完成，worker/guard退出0。252次均EOS、0触顶，原V122的7次触顶消失，但目标核心从98降至57/98，完整支持从91降至57/98。固定配方以提前结束的收益换来大量目标事实失败，不能晋级，也不能只报告停止率。关闭当前将prefix-EOS混入主目标的配方，不通过调整EOS权重、边界、步数或seed救分。

原聚合文件状态字段：`COMPLETE_DEVELOPMENT_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_prefix_completion_v124_20261025/REPORT_ZH.md) · [RESULTS.json](../source_prefix_completion_v124_20261025/RESULTS.json)

## E257 · source_priority_replay_v122_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V122终态：核心覆盖恢复，完整终止未恢复，保护拟合门失败](../source_priority_replay_v122_20261025/REPORT_ZH.md)。

原报告结论节选：SOURCE_PRIORITY_PATH完成全部14×160新训练、2240更新，全部训练冻结后完成238评价，另14零函数核验。预设推进门失败，无候选晋级。相对V120 PATH_REPLAY，98条目标核心覆盖从91升至98，但完整支持回答仍为91；7条同一任务回答先覆盖参考中的全部核心事实，随后退化重复至1024 token。不得裁掉重复、强制EOS、重试或把核心覆盖当作稳定完整编辑成功。

原聚合文件状态字段：`COMPLETE_DEVELOPMENT_GATE_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_priority_replay_v122_20261025/REPORT_ZH.md) · [RESULTS.json](../source_priority_replay_v122_20261025/RESULTS.json)

## E258 · source_span_v74_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V74：源子空间门控改善狭窄保护指标但严重损伤泛化](../source_span_v74_20261023/REPORT_ZH.md)。

原报告结论节选：`SOURCE_SPAN_NOT_PROMOTED`。全10编辑、1600更新、219输出及真实准入完成。T0 10/10、T1G40/40、改写40/40、T1L33/84保持，但T2G21/40：相对B35丢14增0，相对V69 37丢16增0，联合FAIL。

原聚合文件状态字段：`SOURCE_SPAN_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_span_v74_20261023/REPORT_ZH.md) · [RESULTS.json](../source_span_v74_20261023/RESULTS.json)

## E259 · source_stop_v62_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V62：源题达标早停未通过冻结开发门槛](../source_stop_v62_20261022/REPORT_ZH.md)。

原报告结论节选：本轮已完成，结论为 `TT_SOURCE_STOP_NOT_PROMOTED`。TT早停的T2G由33/40降至28/40（−12.5个百分点），归一化LoRA由37/40降至30/40（−17.5个百分点）。两者均无任何编辑取得T2G净提升。保持固定160步作为本轮对照；关闭源题达标停止候选，不启动新实验，不扫描检查间隔、阈值、层位、rank或新损失。

原聚合文件状态字段：`TT_SOURCE_STOP_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_stop_v62_20261022/REPORT_ZH.md) · [RESULTS.json](../source_stop_v62_20261022/RESULTS.json)

## E260 · source_sufficient_v108_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V108：来源停止节省更新并部分修复保护，严格晋级仍失败](../source_sufficient_v108_20261025/REPORT_ZH.md)。

原报告结论节选：14个已曝光开发任务全部按预登记来源规则停止，来源原题/FIT70和未训练表达28均保持核心与全文正确。更新从P160的2240降至900（减少59.82%）；保护全文从P160的24/28改善到26/28，恢复2个损伤绑定且不新增P损伤；但固定强参照B160为27/28，因此未通过“至少B保护水平且无B新增损伤身份”的冻结晋级门槛。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_sufficient_v108_20261025/REPORT_ZH.md) · [RESULTS.json](../source_sufficient_v108_20261025/RESULTS.json)

## E261 · source_supported8_v83_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V83结果：监督资格修复，但未发现医学字段增量，混合监督泛化下降](../source_supported8_v83_20261023/REPORT_ZH.md)。

原报告结论节选：全部8编辑、32训练和620原合同自由回答已完成，0缺失。原S0的8×160步源NLL/实际算子范数及155个历史生成token均精确匹配；路由恢复保留2次失败、43份有效旧输出且未重训。108个唯一完整回答先隐藏arm审核并冻结，再解映射；112个唯一Base回答单独按同标准审核，随后才解owner映射。评分为同一assistant对源参考的完整语义审核，不是临床专家真值或独立确认。

原聚合文件状态字段：`SCORED_ALL_620`。

全部结果/审阅/分析及聚合入口：[PRO_ANALYSIS_REVIEW_ZH.md](../source_supported8_v83_20261023/PRO_ANALYSIS_REVIEW_ZH.md) · [PRO_HANDOFF_REPORT_ZH.md](../source_supported8_v83_20261023/PRO_HANDOFF_REPORT_ZH.md) · [REPORT_ZH.md](../source_supported8_v83_20261023/REPORT_ZH.md) · [RESULTS.json](../source_supported8_v83_20261023/RESULTS.json)

## E262 · source_surprisal_v73_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V73：固定源token惊讶度加权未获益](../source_surprisal_v73_20261023/REPORT_ZH.md)。

原报告结论节选：`SOURCE_SURPRISAL_NOT_PROMOTED`。全10编辑、219输出、1600更新完成；T0 8/10、T1G32/40、T2G30/40、改写32/40、T1L31/84、T2L1/5。相对B的T2G丢7增2，相对V69丢8增1。Base唯一正确T2L仍丢失，另增1题；总数相同不代表保护。源断言36，较A新增身份2；触顶2，较A/B新增身份1。联合FAIL。

原聚合文件状态字段：`SOURCE_SURPRISAL_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_surprisal_v73_20261023/REPORT_ZH.md) · [RESULTS.json](../source_surprisal_v73_20261023/RESULTS.json)

## E263 · source_trajectory_v84_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V84：源轨迹错误覆盖为0，纠错训练未准入](../source_trajectory_v84_20261023/REPORT_ZH.md)。

原报告结论节选：原8编辑、原40训练输入各一次S0实际采样（固定温度.7/128/EOS/原seed规则）已全部完成。隐藏owner和teacher排序的40条完整回答逐条审核，先冻结标签再解映射：40/40与源参考完全一致，全部EOS、0触顶、0额外断言，每编辑5/5正确。

原聚合文件状态字段：`CLOSED_NO_SOURCE_ERRORS`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_trajectory_v84_20261023/REPORT_ZH.md) · [RESULTS.json](../source_trajectory_v84_20261023/RESULTS.json)

## E264 · source_view_v64_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V64：泛化净增两题，但未通过源事实传播保护门槛](../source_view_v64_20261022/REPORT_ZH.md)。

原报告结论节选：状态：`SOURCE_VIEW_COVERAGE_NOT_PROMOTED`。源问法覆盖使TT的T2G从33/40提升至35/40（+5个百分点），编辑7、8各增加1题，未丢失原正确T2G；然而参考未支持的源事实断言消费者从34增加至36，新增2题，因此冻结的联合门槛失败。这个局部正向信号不能包装成整体通过。

原聚合文件状态字段：`SOURCE_VIEW_COVERAGE_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../source_view_v64_20261022/REPORT_ZH.md) · [RESULTS.json](../source_view_v64_20261022/RESULTS.json)

## E265 · swap_anchor_v68_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V68：换图首token锚定损伤泛化与生成稳定性](../swap_anchor_v68_20261023/REPORT_ZH.md)。

原报告结论节选：`SWAPPED_PREFIX_ANCHOR_NOT_PROMOTED`。全部10编辑、219新正式输出已评分。相对V64 B，T2G从35/40降至27/40，丢8题、无新增正确题；T0、T1G、改写和T1L也下降。虽然参考未支持的源断言减少，联合门槛仍明确失败，不将单项辅助指标改善当作整体成功。

原聚合文件状态字段：`SWAPPED_PREFIX_ANCHOR_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../swap_anchor_v68_20261023/REPORT_ZH.md) · [RESULTS.json](../swap_anchor_v68_20261023/RESULTS.json)

## E266 · symmetric_semantic_scope_v117_20261025

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V117四视图范围判别：校准失败](../symmetric_semantic_scope_v117_20261025/REPORT_ZH.md)。

原报告结论节选：冻结规则将问题顺序正反与0/1含义正反的四个同事实分差取等均值。源码/协议在GPU运行前以102e1553公开；未调整提示、权重、阈值或失败标签。

原聚合文件状态字段：`SYMMETRIC_SEMANTIC_CALIBRATION_FAILED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../symmetric_semantic_scope_v117_20261025/REPORT_ZH.md) · [RESULTS.json](../symmetric_semantic_scope_v117_20261025/RESULTS.json)

## E267 · teacher_diagnosis_v82_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V82：教师目标失败的原因与监督改进依据](../teacher_diagnosis_v82_20261023/REPORT_ZH.md)。

原报告结论节选：280份同参考前缀概率记录、56份native greedy回答完成。只使用原8编辑40个训练输入，0训练/评价/Judge。各条件全部240个事实提示经tokenized prompt核验，最长展开前缀829<模型2048上限；条件对Base的KL非零、参考NLL下降，没有发现事实未传入或截断。47个未在首token终止的回答执行了缓存与同前缀完整重算校验，全部通过；首token即EOS的9个回答未进入该检查。

原聚合文件状态字段：`TEACHER_FAILURE_COMPONENTS_IDENTIFIED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../teacher_diagnosis_v82_20261023/REPORT_ZH.md) · [RESULTS.json](../teacher_diagnosis_v82_20261023/RESULTS.json)

## E268 · textjoint_20260924

历史MedTRACE与前驱材料；不并入当前TT配方。

入口：[MedTRACE 文本泛化与保持联合实验：结项核验](../textjoint_20260924/final/AUDIT_REPORT_ZH_PUBLIC.md)。

原报告结论节选：本轮在 my-gpu 的 A100 GPU 2、3 上完成六臂 DEV12、阶段诊断、B0/P 的 single 与真实专家库 sequential 配对、BalancEdit 对照、第二种子和逆序对照。最终状态为 **COMPLETE_EXPLORATORY**，没有可宣称通过工程约束并经独立确认的候选。P 只是后续研究的开发集参考，不能称为已验证改进。

全部结果/审阅/分析及聚合入口：[final/AUDIT_REPORT_ZH_PUBLIC.md](../textjoint_20260924/final/AUDIT_REPORT_ZH_PUBLIC.md)

## E269 · token_path_v61_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V61：相同编辑信息下的 TT 写入路径](../token_path_v61_20261022/REPORT_ZH.md)。

原报告结论节选：按用户认可的 Pro 建议，新立项只比较残差进入模型的位置。当前 V59 是所有token注入；原V22 SOURCE是答案预测位置注入，两者不混名。此次A使用当前V59所有token实现，在原始编辑包上重新训练；B只作用于输入前缀，C只作用于实际视觉token。三者均固定层21、7168参数、rank4、seed225034559、160步及原优化器、普通CE，不添加数据或损失。

原聚合文件状态字段：`NO_DEVELOPMENT_SIGNAL`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../token_path_v61_20261022/REPORT_ZH.md) · [RESULTS.json](../token_path_v61_20261022/RESULTS.json)

## E270 · token_path_v61_analysis_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V61 冻结输出归因与历史结构对照核查](../token_path_v61_analysis_20261022/REPORT_ZH.md)。

原报告结论节选：本次仅分析已有本地输出和历史协议/代码/结果，新增模型调用0、训练0、评分0；不连接服务器、不恢复监测。V61原始判决、冻结标记和SUMMARY保持原字节内容。以下错误分类是事后描述，不能作新的训练标签或因果结论。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../token_path_v61_analysis_20261022/REPORT_ZH.md)

## E271 · tt_native_canon_v93_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V93：原生TT候选仍无增益，来源门全部回退CE](../tt_native_canon_v93_20261023/REPORT_ZH.md)。

原报告结论节选：775/775正式自由回答和40条来源资格完成，0缺失；原CE的155条控制tokens精确复现。完整314个唯一回答中311份按完整绑定继承冻结标签，3份新全文在新臂隐藏状态下逐条审核、冻结后解映射。新全文均为不完整弃答，其中一份还增加未支持的病例/病因陈述，严格拒绝。评分为单研究助手来源参考审核，非独立临床裁决。

原聚合文件状态字段：`SCORED_ALL_775`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../tt_native_canon_v93_20261023/REPORT_ZH.md) · [RESULTS.json](../tt_native_canon_v93_20261023/RESULTS.json)

## E272 · tt_research_review_20261023

阶段综合/研究审阅，非独立实验计数。

入口：[TT 医学知识编辑研究回顾与待审问题（供 Pro 审阅）](../tt_research_review_20261023/PRO_REVIEW_ZH.md)。

原报告结论节选：日期：2026-10-23。本文汇总已完成工作的研究内容与证据，不是新实验协议，也不授权启动实验。证据截至 V80 完成；未进行新的文献查新，不据此断言方法新颖性或顶会、顶刊录用前景。

全部结果/审阅/分析及聚合入口：[PRO_REVIEW_ZH.md](../tt_research_review_20261023/PRO_REVIEW_ZH.md)

## E273 · v62_followup_20261022

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V62现有结果补充分析：差异题、源事实传播与数据曝光](../v62_followup_20261022/REPORT_ZH.md)。

原报告结论节选：本次只读取已有输出、冻结评分与本地历史记录，0模型调用、0训练、0新增数据、0权重恢复。V62所有原评分和主报告保持不变。这是看过结果后的探索性补充，不是新的盲评或预注册实验。

原聚合文件状态字段：`NO_EXPLICIT_EXCLUSION_DENOMINATOR`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../v62_followup_20261022/REPORT_ZH.md) · [SCOPE_RESULTS.json](../v62_followup_20261022/SCOPE_RESULTS.json)

## E274 · visual_path_v99_20261024

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V99 诊断结论：视觉链路有效，当前任务的辨别仍不足](../visual_path_v99_20261024/REPORT_ZH.md)。

原报告结论节选：**本轮证据更支持当前冻结 Base 在这些病例与问法上的辨别不足。没有发现“图片未进入模型”或“视觉特征没有传给语言模型”的证据。** 图像内容能够改变内部特征、首 token 分布，去掉图像还改变了结节问题的回答方向；但原图与相反来源的换图都未得到正确的阳性/阴性区分。

原聚合文件状态字段：`VISUAL_CHAIN_ACTIVE_DISCRIMINATION_NOT_RECOVERED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../visual_path_v99_20261024/REPORT_ZH.md) · [RESULTS.json](../visual_path_v99_20261024/RESULTS.json)

## E275 · worst_view_v75_20261023

结果/机制/资格资料；具体终态以对应原报告为准。

入口：[V75：较差视图优先保留泛化收益，但未解决保护损伤](../worst_view_v75_20261023/REPORT_ZH.md)。

原报告结论节选：`WORST_VIEW_NOT_PROMOTED`。T0 10/10、T1G40/40、T2G36/40、改写40/40、T1L33/84、T2L1/5。保留B35全部正确T2G且新增1，相对A净增3、分布3编辑；相对同初始化V69丢1增0。仍损失Base唯一正确T2L，另增1题，源断言35且相对A新增身份1，联合FAIL。触顶0不抵消保护损伤。

原聚合文件状态字段：`WORST_VIEW_NOT_PROMOTED`。

全部结果/审阅/分析及聚合入口：[REPORT_ZH.md](../worst_view_v75_20261023/REPORT_ZH.md) · [RESULTS.json](../worst_view_v75_20261023/RESULTS.json)

## 根目录独立材料

- [LIVEEDIT_MED_REPRODUCTION_SUMMARY_20260814.md](../LIVEEDIT_MED_REPRODUCTION_SUMMARY_20260814.md)
- [LIVEEDIT_MED_ROUTER_R1_ORACLE_FINAL_RESULTS_20260817.md](../LIVEEDIT_MED_ROUTER_R1_ORACLE_FINAL_RESULTS_20260817.md)
- [M3BENCH_EDITOR_PAPERSPEC_FINAL_STATUS_20260828.md](../M3BENCH_EDITOR_PAPERSPEC_FINAL_STATUS_20260828.md)
- [M3BENCH_V4_FOUNDATION_CLOSURE_20260828.md](../M3BENCH_V4_FOUNDATION_CLOSURE_20260828.md)
