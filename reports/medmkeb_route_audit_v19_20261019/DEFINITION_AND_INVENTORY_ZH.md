# 定义核查与已有证据盘点

官方M3Bench当前README：T1G=同问题、语义匹配新图；T2G=同图、语义保持的问题改写。本次经显式代理读取。历史146队列与原查询ledger逐配对元数据实查：T0 146均同图同问；T1G 584均异图同问；T2G 584均同图异问。因此输入构造没有T1G/T2G对调，旧报告部分中文解释写反。尤其LORA146_REVIEW将T2G优势称为图像泛化优势，应解释为问题改写泛化；历史数值、缺失界、评分与协议不改写。后续表同时显示完整中文名。

附带发现T1L 960关系中1条问题字符串不同，保留为已有定义异常，未删除样本或重算历史评分。T2L 98均同图异问。上述结构配对总数与原Base错误/正确资格过滤后的评分分母不同，不互相替代。

已有原146 TT自然门控single_R0 2372消费者、bank_R0四个前缀3204、insertion_R0 146；原Base已有冻结ledger/输出，旧归档输出与评分不重Judge。single_R0不是指定正确专家，不能作为oracle。原完整Astra报告T1G single100.000%、bank14686.644%，578/578到501/578，支持库竞争的开发信号；其中原76条既有配对loss经Astra仍1→0，既有审计记录始终ON、专家和tokens变化，但无临床唯一正确owner标注，不能直接全部称医学错误路由。T2G相应94.349%→93.151%，是改写结果。

当前V18：56 Base已完整，350 forced方法回答仍在执行；TT自然路由未执行。两版BE已登记single路由+FORCED/BASE复用路径，待350消费者完整时核验100条，不新增这些生成。V18固定两任务每项cross均异图同问，rephrase均同图异问；不会重新训练/更换题目。

补测只用当前尚未退休的V18状态，先在两专家库测五TT臂的自然选择，并与原Base/指定原拥有专家比较。旧V11/V14等已退休权重不重建。缺乏适用关系的新病例/同图其他概念组合仍NA，本轮不是完整临床范围验证或146自然库重跑。

官方定义来源：https://github.com/BioMed-AI-Lab-U-Michgan/M3Bench/blob/main/README.md
历史数值来源：reports/medtrace_post32_closeout_v1_20261007/ASTRA_MEDIUM_RECOVERED_REPORT_ZH.md 与 ASTRA_MEDIUM_RECOVERED_RESULTS.json；关系核查私有元数据回执独立保存。
