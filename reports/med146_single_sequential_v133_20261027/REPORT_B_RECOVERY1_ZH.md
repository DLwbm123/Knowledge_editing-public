# MED 原146 完整顺序编辑报告

状态：COMPLETE。原队列、资格掩码与R0固定；本轮为已暴露开发集。
MED 与历史 M13P 回答在同一 CORE/FULL 来源支持协议下评分；历史 accepted 标签不改写。
没有全146 GEN/FULL_CTX对照，不能把差异归因为医学命题特异增益。

|阶段|方法|T2G CORE宏平均|T2G FULL宏平均|T2L CORE支持|
|---|---|---:|---:|---:|
|single_1|MED|78.9384|75.7420|53/54；UNKNOWN 0|
|single_1|LEGACY|78.0822|74.5434|54/54；UNKNOWN 0|
|single_1|BASE|0.1712|0.0000|54/54；UNKNOWN 0|
|bank_1|MED|75.0000|75.0000|0/0；UNKNOWN 0|
|bank_1|LEGACY|75.0000|75.0000|0/0；UNKNOWN 0|
|bank_1|BASE|0.0000|0.0000|0/0；UNKNOWN 0|
|bank_50|MED|77.8333|74.6667|17/17；UNKNOWN 0|
|bank_50|LEGACY|75.8333|73.6667|17/17；UNKNOWN 0|
|bank_50|BASE|0.0000|0.0000|17/17；UNKNOWN 0|
|bank_100|MED|78.9167|76.3333|33/34；UNKNOWN 0|
|bank_100|LEGACY|78.1667|75.3333|34/34；UNKNOWN 0|
|bank_100|BASE|0.2500|0.0000|34/34；UNKNOWN 0|
|bank_146|MED|77.5114|74.6575|53/54；UNKNOWN 0|
|bank_146|LEGACY|76.6553|73.4589|54/54；UNKNOWN 0|
|bank_146|BASE|0.1712|0.0000|54/54；UNKNOWN 0|

百分比为编辑宏平均；空支持为NA。UNKNOWN后的范围为将缺失全判错或全判对的上下界，不是置信区间；不把缺失评分视为已知错误。逐编辑配对区间、原资格掩码、fresh CORE/FULL损伤与旧伤修复、生成完整性和费用见JSON。
顺序阶段真实计算未编辑Base的R0选择；相同专家状态、完整输入及确定性解码身份允许复用同一回答，每个消费者均保留独立实际路由绑定。
所有146个MED终态专家与复现配置保留。科学负结果不取消已登记顺序阶段。
