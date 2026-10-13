# 实现自审

沿用冻结 source/runtime、H expert tensor、R0 router、Base输入绑定、生成配置及Sol Judge rubric。没有训练/损失/optimizer/gradient surgery入口；推理worker禁止backward，expert requires_grad=False。

RCAP是float32 nextafter半径收缩，NEG0仅固定tau0 veto，均不做fallback/reroute。新决定只能保留H原expert或OFF；验证历史完整input/model/backend/bank权重/radii/key绑定后复用H或Base输出，payload保持原Sol key。每个effective缓存加锁避免并行重复生成。R0 exact/parity机械验证真实运行。

源文件语法、shared missing、科学门槛、复杂度tie-break、NO_ROUTING_EFFECT、冻结分母、pending controller恢复均通过可运行检查。480步等训练检查不适用于本轮，实际training/backward/optimizer为0。12项机械检查另存实际回执。

历史缺少部分S_fit路由特征缓存，已保留失败回执并用原冻结Base extractor补齐合法支持特征，无新样本、无训练、无Judge。U_bg/正式input与source隔离通过；hazard先冻结，再读暴露damage。CAL独立审计当前冻结资产，缺少足量verified隔离cross-image positives，未借用formal数据或虚构标注。

本轮有限controller只推进DEV及条件REG，不重新选择参数，不自动恢复旧小时监测。最终公开交付在实际收口后验证；当前尚未声称科学通过或完成交付。
