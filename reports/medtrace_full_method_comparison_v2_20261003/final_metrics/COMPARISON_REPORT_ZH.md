# 原146完整流程比较

本表使用共同新Astra/high评分，保留原Base资格掩码。数值按完整合格分母计算；方括号为共享missing精确点界，不是置信区间。独立H_eval为NA。

|方法|任务|编辑数|probe分母|已知正确|missing出现|宏平均界|微平均界|
|---|---|---:|---:|---:|---:|---|---|
|MedTRACE_AVAILABLE_H_R0|T0 PostAcc|146|146|146|0|[100.00%, 100.00%]|[100.00%, 100.00%]|
|BalancEdit_locked_adaptation|T0 PostAcc|146|146|146|0|[100.00%, 100.00%]|[100.00%, 100.00%]|
|BELoRA_V2_effect_repaired|T0 PostAcc|146|146|117|0|[80.14%, 80.14%]|[80.14%, 80.14%]|
|MedTRACE_AVAILABLE_H_R0|T1G Fix|146|578|502|0|[86.82%, 86.82%]|[86.85%, 86.85%]|
|BalancEdit_locked_adaptation|T1G Fix|146|578|500|0|[86.47%, 86.47%]|[86.51%, 86.51%]|
|BELoRA_V2_effect_repaired|T1G Fix|146|578|400|5|[69.35%, 70.21%]|[69.20%, 70.07%]|
|MedTRACE_AVAILABLE_H_R0|T1L Retention|1|2|2|0|[100.00%, 100.00%]|[100.00%, 100.00%]|
|BalancEdit_locked_adaptation|T1L Retention|1|2|2|0|[100.00%, 100.00%]|[100.00%, 100.00%]|
|BELoRA_V2_effect_repaired|T1L Retention|1|2|2|0|[100.00%, 100.00%]|[100.00%, 100.00%]|
|MedTRACE_AVAILABLE_H_R0|T2G Fix|146|557|485|0|[87.44%, 87.44%]|[87.07%, 87.07%]|
|BalancEdit_locked_adaptation|T2G Fix|146|557|527|0|[94.86%, 94.86%]|[94.61%, 94.61%]|
|BELoRA_V2_effect_repaired|T2G Fix|146|557|369|0|[66.78%, 66.78%]|[66.25%, 66.25%]|
|MedTRACE_AVAILABLE_H_R0|T2L Retention|31|54|26|0|[55.91%, 55.91%]|[48.15%, 48.15%]|
|BalancEdit_locked_adaptation|T2L Retention|31|54|28|0|[58.60%, 58.60%]|[51.85%, 51.85%]|
|BELoRA_V2_effect_repaired|T2L Retention|31|54|38|0|[71.51%, 71.51%]|[70.37%, 70.37%]|

全部single/prefix、配对bootstrap、来源组敏感性、Base新掩码、插入保持和消耗见同目录JSON/CSV。

- Original146 exposed diagnostic/regression cohort and one order, not independent confirmation
- 8 edits use qualified H; 138 retain original C_NO_H; not146 full C_FACT
- H_eval unsupported; no ten-task overall
- Patient independence UNKNOWN; original U source shared and future metadata overlap disclosed
- BalancEdit locked adaptation; BELoRA effect-repaired rank16/alpha16/50steps, not paper exact
- No immutable Judge model snapshot; common new scoring epoch does not erase prior exposure
