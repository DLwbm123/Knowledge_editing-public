# 路由泛化与 U 净收益：两线开发实验

原146和旧CAL/CHECK均已经暴露，结果仅用于开发；不声明独立确认或医学保护。U以同W0、同seed、同CE批次和160步的CE为对照；U多一次KL前反向，真实成本单独记录。24库只更新前8，另16固定W0。路由只增加native及已有4条FIT改写锚点，半径/专家权重/生成不变；仅评估最终146库。

|比较|模式|库大小|任务|差值下界pp|差值上界pp|
|---|---|---:|---|---:|---:|
|U_PAIR_CEU minus U_PAIR_CE|single_R0|1|T0|0|0|
|U_PAIR_CEU minus U_PAIR_CE|single_R0|1|T1G|0|0|
|U_PAIR_CEU minus U_PAIR_CE|single_R0|1|T2G|4.6875|4.6875|
|U_PAIR_CEU minus U_PAIR_CE|single_R0|1|T1L|None|None|
|U_PAIR_CEU minus U_PAIR_CE|single_R0|1|T2L|None|None|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|8|T0|0|0|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|8|T1G|0|0|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|8|T2G|4.6875|4.6875|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|8|T1L|None|None|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|8|T2L|None|None|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|24|T0|0|0|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|24|T1G|0.0|0.0|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|24|T2G|1.5625|1.5625|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|24|T1L|None|None|
|U_PAIR_CEU minus U_PAIR_CE|bank_R0|24|T2L|None|None|
|TT88_FITKEY5 minus TT88_W0_RETRO146|bank_R0|146|T0|0|0|
|TT88_FITKEY5 minus TT88_W0_RETRO146|bank_R0|146|T1G|0.8561643835616438|0.8561643835616438|
|TT88_FITKEY5 minus TT88_W0_RETRO146|bank_R0|146|T2G|0.5136986301369862|0.5136986301369862|
|TT88_FITKEY5 minus TT88_W0_RETRO146|bank_R0|146|T1L|0|0|
|TT88_FITKEY5 minus TT88_W0_RETRO146|bank_R0|146|T2L|0|0|

差值界保留缺失分母；配对置信区间、完整面板、U强制启用/自然路由KL与梯度诊断见RESEARCH_RESULTS.json。不能将保护输入路由关闭产生的零KL称作U有效。所有正负结果均保留。
