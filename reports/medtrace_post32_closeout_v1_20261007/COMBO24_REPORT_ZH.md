# 24编辑 U × 路由组合：冻结权重开发比较

复用48份既有160步权重与566个原R0输出，仅新增566个FITKEY5输出。所有24专家均为对应CE/CEU条件，一个既定seed；不是此前仅前8更新的24库。旧CHECK只有4个来源，非独立确认。

|比较|任务|差值下界pp|差值上界pp|
|---|---|---:|---:|
|U at R0|T0|0|0|
|U at R0|T1G|0.0|1.0416666666666665|
|U at R0|T2G|-3.819444444444444|-3.819444444444444|
|U at R0|T1L|None|None|
|U at R0|T2L|None|None|
|U at FITKEY5|T0|0|0|
|U at FITKEY5|T1G|0.0|1.0416666666666665|
|U at FITKEY5|T2G|-3.819444444444444|-3.819444444444444|
|U at FITKEY5|T1L|None|None|
|U at FITKEY5|T2L|None|None|
|FITKEY5 with CE|T0|0|0|
|FITKEY5 with CE|T1G|0|0|
|FITKEY5 with CE|T2G|0|0|
|FITKEY5 with CE|T1L|None|None|
|FITKEY5 with CE|T2L|None|None|
|FITKEY5 with CEU|T0|0|0|
|FITKEY5 with CEU|T1G|0|0|
|FITKEY5 with CEU|T2G|0|0|
|FITKEY5 with CEU|T1L|None|None|
|FITKEY5 with CEU|T2L|None|None|
|Combined versus CE R0|T0|0|0|
|Combined versus CE R0|T1G|0.0|1.0416666666666665|
|Combined versus CE R0|T2G|-3.819444444444444|-3.819444444444444|
|Combined versus CE R0|T1L|None|None|
|Combined versus CE R0|T2L|None|None|

缺失保留分母；已有失败的完全相同payload继承缺失，不隐性重试。全部面板、编辑/来源区间及自然路由CHECK KL/精确Base-token一致性见COMBO24_RESULTS.json。组合效果不得归给单一因素；病例独立性未知、医学保护资格0。
