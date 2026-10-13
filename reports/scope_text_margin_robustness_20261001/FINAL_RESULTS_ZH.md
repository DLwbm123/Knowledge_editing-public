# TXT三点margin稳健性结果

完整989 DEV与1058暴露REG，各tau2047消费者，三个冻结点全部报告。tau0路由/key和完整joint与已执行TXT逐项重现PASS。所有反事实选择仅用原R0 H或Base强绑定的既有输出和评分，无新推理或Judge；原winner/radius/词面规则/支持资格与门槛固定。

|面板|tau|原joint|相对tau0路由变化出现次数|bank24 Pressure已知增益|shared-key missing增益计数界|
|---|---|---|---|---|---|
|DEV|-0.1|PASS|107|+4|[3,5]/50|
|DEV|0.0|PASS|0|+11|[7,12]/50|
|DEV|0.1|PASS|38|+14|[10,15]/50|
|REG|-0.1|PASS|117|+6|[6,6]/60|
|REG|0.0|PASS|0|+11|[11,11]/60|
|REG|0.1|FAIL|97|+13|[13,13]/60|

DEV三个点均PASS；REG中tau+0.1明确FAIL：bank12 T1G新增错误精确界[1,1]/48，bank24为[1,1]/96，门槛均为0。Pressure同时通过，故不能只取正向收益声称整体通过。是否为重复消费者必须在下一独立诊断按unique input核对，不将两个prefix当成两个独立病例。

原tau0仍PASS；本轮未挑选最佳tau、未改变阈值，也未扩大搜索。只证明固定局部扰动在暴露REG可出现误拒，不能声称真实独立数据泛化或临床验证。下一轮按预先声明的失败分支，冻结误拒路由机制诊断；真正临床scope仍等待有来源与审核证据的材料。

两次机械失败均发生于首个参数joint输出之前。第一次误将panel内exact.identity用于完整跨prefix消费者，导致身份键折叠；改为包含mode/prefix的全身份，原exact/joint保持panel内调用。第二次OFF行合法无q字段，调用参数直接取q导致KeyError；改为OFF安全的get，全部活跃R0行必有有效q。原失败源码/日志/回执保留private/recovery/DENOMINATOR_1和OFF_DIAGNOSTICS_2；原manifest计时不重置，参数与科学门槛不变。

成功CPU墙钟1.103秒、进程CPU 0.934秒；失败墙钟保守上界4.157秒，累计CPU任务墙钟上界5.260秒。GPU/Judge/训练/生成0，累计Judge7472，41永久missing未重判。公开匿名聚合/源码/计划/自检与限制，私有QA、逐条keys/路由、tokens与权重留服务器。
