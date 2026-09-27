# MedTRACE TextJoint-R4 中文报告

训练、生成与可提交评分已结束。核心配对完整；两条新臂的 DEV sequential 压力指标存在 missing。正式联合工程目标未达到。

本轮只新增 beta=0.125/0.25，复用完整绑定的 P80(beta=0)与P+S80(beta=0.5)。总KL权重0.01不变；共享P-W0、支持与采样器，L30/rank4/80步。
CAL_PLUS 合格跨图正例不足，因此 Rcal/Rscope 未获资格，保留R0；这不是标量阈值家族不可行的实验证明。A类只复用了R3 CAL，不能宣称完成了全新的四类校准。
新数据前置盘点发现授权core9目录11088行均存在历史Base暴露；全历史独立性未认证。使用旧VERIFY24作回归，不是新N24或CONFIRM。

DEV冻结选择：P+S。点估计门槛通过不等于统计非劣证明。

## 结论与边界

本轮新增两档较弱 S 没有在 DEV 胜出，冻结选择仍为原 P+S（beta=0.5）。DEV single 压力保持 edit-macro：P 66.67%，beta=0.125/0.25 均76.39%，beta=0.5为79.86%；四臂 T1G均100%，T2G均98.96%。beta=0.25 的T2G发生1退化/1恢复，均值不变不能解释为逐题不变。

冻结后复用完整执行绑定的旧 VERIFY24 回归：P+S 相对P的压力保持从63.89%到82.64%（+18.75个百分点，本轮配对bootstrap区间[9.03,29.17]），但T2G从96.88%到94.79%（-2.08个百分点，3退化/1恢复），违反最多下降1个百分点的工程要求。seq@24压力保持从19.44%到65.97%，仍不抵消泛化门槛失败。上述VERIFY输出来自已验证的历史执行，未新增独立病例或确认数据。未入选的弱S不在VERIFY另行挑优，因此不能声称它们已补回VERIFY泛化。

跨图CAL_PLUS只有1条合格正例、没有编辑达到2条，未达到12编辑目标；Rcal/Rscope不具备校准资格。R*=R0，合并重复矩阵，不把退化为零的路由差值或交互解释成路由改善。

## 执行、评分与资源

48个新专家均完成80步及保存恢复核验；2776条历史R0消费者通过完整绑定复用。正式消费者共3848条，其中2条缺评分：同一新连接失败请求影响beta=0.125/0.25的DEV sequential压力面板。其分子仅为已知正确数，主macro/micro不得用已评分子集替代。DEV single筛选和选中P/P+S的VERIFY核心配对完整。

Judge缓存1740个唯一请求，1738已评分、2个missing；其中1个为继承的R3 A48失败，1个为本轮连接失败。最初CLI路径失效的一项由用户明确授权补评，已成功；原失败消耗保留。无未提交或在途请求。未获新授权的失败不重试。

本轮GPU驻留3923.37秒（1.09 GPU小时），历史加本轮41678.27秒（11.58 GPU小时），低于16小时上限；累计Judge尝试1868项，本轮37项，含失败及获准补评，低于6000项。双卡分别累计。队列于2026-09-26 15:31 UTC（北京时间23:31）结束，无活跃本轮GPU会话；墙钟从初始13:21 UTC计约2小时10分钟，未以空占补足12小时。数据资格不足与矩阵复用使队列提前结束，未扩大搜索。发布核验晚于运行结束，不新增GPU或Judge任务。

原始失败证据、逐题变化、精确支持/teacher内容及目标log-prob仅私有保存。公开逐编辑梯度使用匿名索引；原始梯度冲突不是评测错误的因果证明。T1L仅2/2，不能当作稳健性证据。

|面板/模式/前缀|臂|任务|分子/分母|edit-macro|missing|
|---|---|---|---|---|---|
|DEV24_sequential_12|P+S@80_R0|T0|12/12|100.00%|0|
|DEV24_sequential_12|P+S@80_R0|T1G|43/45|95.83%|0|
|DEV24_sequential_12|P+S@80_R0|T2G|45/46|97.92%|0|
|DEV24_sequential_12|P+S@80_R0|T2L|3/3|100.00%|0|
|DEV24_sequential_12|P+S@80_R0|T2L_PRESSURE|21/26|80.56%|0|
|DEV24_sequential_12|P@80_R0|T0|12/12|100.00%|0|
|DEV24_sequential_12|P@80_R0|T1G|43/45|95.83%|0|
|DEV24_sequential_12|P@80_R0|T2G|46/46|100.00%|0|
|DEV24_sequential_12|P@80_R0|T2L|3/3|100.00%|0|
|DEV24_sequential_12|P@80_R0|T2L_PRESSURE|9/26|36.11%|0|
|DEV24_sequential_24|B125@80_R0|T0|24/24|100.00%|0|
|DEV24_sequential_24|B125@80_R0|T1G|86/93|92.71%|0|
|DEV24_sequential_24|B125@80_R0|T2G|92/93|98.96%|0|
|DEV24_sequential_24|B125@80_R0|T2L|4/8|50.00%|0|
|DEV24_sequential_24|B125@80_R0|T2L_PRESSURE|30/50|null|1|
|DEV24_sequential_24|B25@80_R0|T0|24/24|100.00%|0|
|DEV24_sequential_24|B25@80_R0|T1G|86/93|92.71%|0|
|DEV24_sequential_24|B25@80_R0|T2G|92/93|98.96%|0|
|DEV24_sequential_24|B25@80_R0|T2L|6/8|72.22%|0|
|DEV24_sequential_24|B25@80_R0|T2L_PRESSURE|34/50|null|1|
|DEV24_sequential_24|P+S@80_R0|T0|24/24|100.00%|0|
|DEV24_sequential_24|P+S@80_R0|T1G|86/93|92.71%|0|
|DEV24_sequential_24|P+S@80_R0|T2G|92/93|98.96%|0|
|DEV24_sequential_24|P+S@80_R0|T2L|6/8|72.22%|0|
|DEV24_sequential_24|P+S@80_R0|T2L_PRESSURE|36/50|72.22%|0|
|DEV24_sequential_24|P@80_R0|T0|24/24|100.00%|0|
|DEV24_sequential_24|P@80_R0|T1G|86/93|92.71%|0|
|DEV24_sequential_24|P@80_R0|T2G|92/93|98.96%|0|
|DEV24_sequential_24|P@80_R0|T2L|3/8|33.33%|0|
|DEV24_sequential_24|P@80_R0|T2L_PRESSURE|12/50|25.00%|0|
|DEV24_single_1|B125@80_R0|T0|24/24|100.00%|0|
|DEV24_single_1|B125@80_R0|T1G|93/93|100.00%|0|
|DEV24_single_1|B125@80_R0|T2G|92/93|98.96%|0|
|DEV24_single_1|B125@80_R0|T2L|4/8|50.00%|0|
|DEV24_single_1|B125@80_R0|T2L_PRESSURE|38/50|76.39%|0|
|DEV24_single_1|B25@80_R0|T0|24/24|100.00%|0|
|DEV24_single_1|B25@80_R0|T1G|93/93|100.00%|0|
|DEV24_single_1|B25@80_R0|T2G|92/93|98.96%|0|
|DEV24_single_1|B25@80_R0|T2L|6/8|72.22%|0|
|DEV24_single_1|B25@80_R0|T2L_PRESSURE|38/50|76.39%|0|
|DEV24_single_1|P+S@80_R0|T0|24/24|100.00%|0|
|DEV24_single_1|P+S@80_R0|T1G|93/93|100.00%|0|
|DEV24_single_1|P+S@80_R0|T2G|92/93|98.96%|0|
|DEV24_single_1|P+S@80_R0|T2L|6/8|72.22%|0|
|DEV24_single_1|P+S@80_R0|T2L_PRESSURE|40/50|79.86%|0|
|DEV24_single_1|P@80_R0|T0|24/24|100.00%|0|
|DEV24_single_1|P@80_R0|T1G|93/93|100.00%|0|
|DEV24_single_1|P@80_R0|T2G|92/93|98.96%|0|
|DEV24_single_1|P@80_R0|T2L|3/8|33.33%|0|
|DEV24_single_1|P@80_R0|T2L_PRESSURE|33/50|66.67%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P+S@80_R0|T0|12/12|100.00%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P+S@80_R0|T1G|46/48|95.83%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P+S@80_R0|T2G|45/48|93.75%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P+S@80_R0|T2L|4/6|62.50%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P+S@80_R0|T2L_PRESSURE|23/36|63.89%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P@80_R0|T0|12/12|100.00%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P@80_R0|T1G|46/48|95.83%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P@80_R0|T2G|46/48|95.83%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P@80_R0|T2L|3/6|37.50%|0|
|R2_VERIFY24_REGRESSION_sequential_12|P@80_R0|T2L_PRESSURE|13/36|36.11%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P+S@80_R0|T0|24/24|100.00%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P+S@80_R0|T1G|91/96|94.79%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P+S@80_R0|T1L|2/2|100.00%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P+S@80_R0|T2G|89/94|94.79%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P+S@80_R0|T2L|5/9|58.33%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P+S@80_R0|T2L_PRESSURE|40/60|65.97%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P@80_R0|T0|24/24|100.00%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P@80_R0|T1G|91/96|94.79%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P@80_R0|T1L|2/2|100.00%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P@80_R0|T2G|91/94|96.88%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P@80_R0|T2L|4/9|41.67%|0|
|R2_VERIFY24_REGRESSION_sequential_24|P@80_R0|T2L_PRESSURE|12/60|19.44%|0|
|R2_VERIFY24_REGRESSION_single_1|P+S@80_R0|T0|24/24|100.00%|0|
|R2_VERIFY24_REGRESSION_single_1|P+S@80_R0|T1G|96/96|100.00%|0|
|R2_VERIFY24_REGRESSION_single_1|P+S@80_R0|T1L|2/2|100.00%|0|
|R2_VERIFY24_REGRESSION_single_1|P+S@80_R0|T2G|89/94|94.79%|0|
|R2_VERIFY24_REGRESSION_single_1|P+S@80_R0|T2L|5/9|58.33%|0|
|R2_VERIFY24_REGRESSION_single_1|P+S@80_R0|T2L_PRESSURE|49/60|82.64%|0|
|R2_VERIFY24_REGRESSION_single_1|P@80_R0|T0|24/24|100.00%|0|
|R2_VERIFY24_REGRESSION_single_1|P@80_R0|T1G|96/96|100.00%|0|
|R2_VERIFY24_REGRESSION_single_1|P@80_R0|T1L|2/2|100.00%|0|
|R2_VERIFY24_REGRESSION_single_1|P@80_R0|T2G|91/94|96.88%|0|
|R2_VERIFY24_REGRESSION_single_1|P@80_R0|T2L|4/9|41.67%|0|
|R2_VERIFY24_REGRESSION_single_1|P@80_R0|T2L_PRESSURE|39/60|63.89%|0|

配对计数、edit bootstrap与来源聚类敏感性见RESULTS_PUBLIC.json；逐题变化只保留私有。标准T2L与压力保持分开，空分母不算PASS。
同路由比较writer；R*=R0时路由差值和交互退化为0，不重复生成或宣称路由创新。历史R3 A48失败请求继续missing，不自动补评。
逐编辑梯度余弦分布见GRADIENT_DISTRIBUTION.json。原始梯度与Adam预条件后的实际更新分开；loss下降不代表性能提升。
历史报告、选择锁、模型与账本均未改写。所有结果仍为探索性，seq60%不是安全阈值。
