# 原146指标汇总实现与执行条件

2026-10-04本地01:10左右，本轮训练和共同评分仍在进行，完整性能尚未读取。此文落实既有冻结计划与原Stage17评价合同，不另改队列、数据、训练、路由、Judge、门槛或预算。CPU工具`reporting.py`单独部署到run/private/tools；运行GPU的私有SOURCE和当前健康scorer保持原版本。

执行前要求完整146 single、真实sequential prefix1/50/100/146生成、controller结束和SCORER_DONE回执，payload全部FORMAT_VALID或永久MISSING。汇总入口以只读事务读取SQLite，并核验完整输出/消费者/payload/Judge批次证据绑定及一次attempt账本。不以未完成子集形成完整结果。该入口不能训练、生成、请求Judge或改评分状态。

按原事件和probe出现顺序统计，先在每个有合格支持的编辑内取平均，再在编辑间宏平均；微平均另列完整probe数、已知正确、missing出现及不同missing key数。原Base eligibility/masks固定。T0始终保留原146 native的PostAcc，包含新Base判为正确的native；任何新Base敏感性均不重选native队列。T1G/T2G Fix只选原Base错误，T1L/T2L Retention只选原Base正确；PostAcc及c2w另报，空支持为NA。T4G只是附加诊断，不能作为十任务整体结论。

原完整146面板合格分母：T0为146编辑/146probe，T1G为146/578，T2G为146/557，T1L为1/2，T2L为31/54；这些分母不能替换为全部594 task的支持数。原prefix active-target映射机械审计为1/50/100/146个native query，没有参考答案改变，也没有合格locality碰撞；汇总仍保留预声明的active-locality处理，并在实际报告中计数。

主设置为sequential最终prefix146；single和四prefix另列。MedTRACE分别减BalancEdit、BELoRA，成对支持逐项一致。宏/微差值用有理数系数合并同完整payload key，missing是一个共享二元变量；系数抵消后再求极值，不把共享missing当独立样本，也不删除它。无missing时报告点值。

10000次paired edit bootstrap与native source-group cluster敏感性继续用seed20260912、原Stage17排序位置249/9749。完整判分时已用合成例与原`stage17_report.interval`核对一致。若存在missing，每次抽样先合并共享key系数，报告逐次下界分布的2.5%与上界分布的97.5%所形成的描述性包络；这不是某一个missing补全的bootstrap，也不是独立确认或优越性检验。来源组重采样保持编辑加权，不能替代患者独立或新顺序验证。

插入到最终native保持以全部146对输入审核。不同native完整key不交叉共享；每对枚举其合法二元状态，再用计数动态规划求`插入正确且最终正确 / 插入正确`的精确可行界，并明确零分母是否可能。两时刻同key的missing必须同值。

新Base掩码只另列敏感性和与原mask的分歧计数。若新Base还有永久missing，敏感性资格不强行填补，也不丢弃unknown后宣称完整新mask结果，面板记NA及原因；原mask主表照完整分母报告。H_eval为NA，患者独立UNKNOWN，8编辑/12关系H与其余138原C_NO_H策略如实披露。

既有验证建议门槛只在最终共同结果上按共享missing点差下界计算，满足也不自动新seed或新参数。报告将列出全部失败成本、GPU租约、Judge载荷/批次/永久missing及历史基线复用限制。每小时存储快照不是连续峰值计量，精确存储峰值不编造；最后消费者后的实际owned文件清单和删除回执另行审核，当前不删训练/银行消费者仍需的权重。

机械检查包括共享missing抵消、有理权重宏/微、穷举missing极值、零支持NA、成对支持、共享插入保持及原完整判分bootstrap一致。当前元数据准入PASS、零GPU/生成/Judge新增；实际报告执行和公开交付仍待完整结果。公开交付仅源码/配置/必要检查/匿名汇总及限制，不能发布逐题私有绑定或医疗文本。
