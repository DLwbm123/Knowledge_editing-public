# MED-RA 原146 single筛选报告

状态：COMPLETE_SINGLE_ONLY；sequential：SEQUENTIAL_NOT_RUN_SCREENING_FAIL。原V133基线与评分保持。

CORE为核心参考正确；FULL为完整回答的保守来源支持，不是临床真值。以下为编辑宏平均百分比。

|阶段|方法|标准|T0|FIT|T1G|T2G|T1L|T2L|
|---|---|---|---:|---:|---:|---:|---:|---:|
|single_1|MED_RA|CORE|79.4521|78.7671|79.9658|63.4703|100.0000|98.3871|
|single_1|MED_RA|FULL|76.0274|76.1986|76.0274|59.5320|100.0000|0.0000|
|single_1|M13P|CORE|100.0000|100.0000|100.0000|78.0822|100.0000|100.0000|
|single_1|M13P|FULL|100.0000|100.0000|100.0000|74.5434|100.0000|0.0000|
|single_1|MED|CORE|100.0000|100.0000|100.0000|78.9384|100.0000|98.3871|
|single_1|MED|FULL|100.0000|100.0000|100.0000|75.7420|100.0000|0.0000|
|single_1|BASE|CORE|0.0000|1.1986|0.1712|0.1712|100.0000|100.0000|
|single_1|BASE|FULL|0.0000|0.3425|0.0000|0.0000|0.0000|0.0000|

single预定筛选：

- T2G_CORE_not_below_MED：FAIL/UNKNOWN
- CORE_trend_vs_M13P_preserved：FAIL/UNKNOWN
- no_new_fresh_CORE_damage_vs_M13P：FAIL/UNKNOWN
- T2L_CORE_not_below_M13P：FAIL/UNKNOWN
- source_T0_FIT_CORE_FULL_no_new_regression：FAIL/UNKNOWN
- all_scoring_complete：PASS
- actual_route_and_state_checks：PASS

single未通过不运行sequential；缺失评分不会变为错误分数，空支持保持NA。预定+3pp与配对区间另见JSON，未用于放宽筛选。

五种规定T2G差值、编辑/来源组95%区间、fresh损伤/修复、同旧owner遗忘、即时插入/最终保留、路由/复用、alpha/q/费用见JSON及私有逐项审计。

本轮仅改每expert有限更新幅度。raw160终态按实际state hash只读复用，继承训练成本另列。没有全146 GEN/FULL_CTX，不宣称医学独有贡献或独立确认。
