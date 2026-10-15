# PM/PU首源执行

用户已授权启动并每半小时持续监测，允许pro5000 GPU0–7、最多同时两张；本轮绑定GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff，只用一张卡串行PM→PU，科学协议540223f不改。

生产CPU PASS：投影分解/解析梯度、极端logits稳定无floor、实际展开字段构造的prefix掩码与损坏拒绝、PM不反传prefix/PU反传、与SH00同目标候选路径逐项精确、拒绝回滚、状态下一步恢复、clock检查、两臂准入屏障、261消费者与共享payload界、损伤不通过、每臂与总计数/驻留限制。真实processor字段及答案logits一致性还必须现场准入；CPU小型fixture不代表真实VLM通过。

复用上一SH执行链的独立快照，仅替换目标和掩码、两臂预算与报告。训练核心二次solve/native验收不改，仍保留Base16/removed4背景与.01风险门；不执行removed软KL/实际硬约束。所有原始训练视图teacher当前窗临时保存，源结束退役；当前训练源外的病例不进入学习器。

首次安装即激活4GPUh/24wallh、每臂2GPUh，冻结计数和成本不能重置。安装前一次存储/显存检查，正式槽前两臂各Base3/真实W写入/上游结构不变全部准入。每臂新进程2题精确重载及所有GPU租约闭合后才评分；4scorer与有限tail负责完整分析、source-only push和公开验证。无通用中途恢复入口，不盲重放已尝试动作。

范围影响：新source_loss→worker、runtime/controller、bootstrap/deploy、queue/report、scorer/tail及生产CPU夹具均在新experiments/purew_projected_objective_first1_v1；旧SH/R/J/F/S及所有历史报告不改。原146/M保持暂停、O/L等待，不自动下一轮或146。
