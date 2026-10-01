# DEV Router-only 结果

所有方法 expert weights 冻结；训练/backward/optimizer 为0。已完成 single、bank4/8/12/24 和 EXPOSED_REGRESSION old47。完整分母保留 missing；原 Base 资格不重定义。

|Router|bank24 T2G已知正确/完整分母|bank24 Pressure已知正确/完整分母|缺失(T2G/Pressure)|
|---|---|---|---|
|R0|89/93|29/50|1/5|
|NEG0|89/93|40/50|1/2|
|PLOO|89/93|40/50|1/2|

route buckets、switch matrix、Positive Rejection、Negative Rescue、Negative Damage、NetRescue，以及bank8→12/12→24固定共享输入面板见 ROUTE_TRANSITION_ANALYSIS.json。NetRescue不替代联合门槛。DEV为暴露诊断，REG为暴露回归，不是独立确认。T1L/T2L资格以原冻结 BASE_MASKS 决定，未合格输入不充入正式locality分母。

PLOO相对NEG0所有989消费者的route与Judge payload均相同，guard额外接受0。single T1G新增7错，bank24新增3错，联合FAIL。
