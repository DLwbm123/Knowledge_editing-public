# MED 扩展4条：Base资格评分

按角色安全规则固定4条后，复用与原运行模型、输入、原图身份、原始token及解码合同一致的历史Base完整回答，以原 `MEDPROP_SMALL4_CORE_FULL_SOURCE_V1`、gpt-6-astra medium作一次新评分。4/4为CORE错误，4/4判定格式有效、未知0，满足新增native的Base错误资格。4次尝试，0自动语义/传输重试，原Stage17 high判定和旧回答不被覆盖。

隔离读取、代理预检及本机资源保护器通过，评分worker和保护器正常退出。该资格仅确认参考合同下的Base判断，不等于150正式方法结果或临床真值。新增4条有旧Base曝光、均为VQA-RAD，且源角色、T1G/T2G开发支持与locality仍需按预登记完成。原146结果、失败记录与M13P保留约定均不改变。
