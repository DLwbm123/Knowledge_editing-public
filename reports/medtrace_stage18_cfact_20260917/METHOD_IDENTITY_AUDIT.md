# Stage18 方法身份核验

代码提交：`0d54456c96742c1e8d32aef5976c4d55030e9077`。权威为本轮附件和 Stage17 已锁定 C_V1；Stage17 历史状态字符串不代表本轮授权。

|批准规格|实际模块/配置|检查|执行状态|
|---|---|---|---|
|LLaVA-Med Mistral7B、FP16、原生生成|stage17_single.setup + load_real_runtime；FULL_METHOD_LOCK|冻结 source/module/generation 验证|GPU smoke 待回执|
|down_proj 第21层、freeR4、73728 FP32|LowRankExpert(rank=4)|CPU CP 转换、参数/优化器隔离；GPU 参数数断言|CPU通过|
|每编辑 native CP→A2 80→CP-W0 320|复用未修改 stage15.initialize|同一 W0 小型状态哈希，三分支独立 Adam|代码已实现；GPU待执行|
|L0 + H / G extra CE|stage18_cfact.update/train|实际图上 H/G 梯度非零，权重零贡献消失|CPU通过；GPU待执行|
|H/G 自己图像/问题/来源答案|source_batch + smoke source核对|图像/问题/答案替换检查|CPU及原始来源核对通过|
|full-vocab KL Base∥student|原 selective_write.full_vocab_kl + 新缓存绑定|方向、温度1、全词表、shift/EOS、失配拒绝|CPU通过|
|R0 Base-only 最近邻/半径|原 MemoryRouter；清除 hook 后提 key|无第二近回退/稳定tie；GPU OFF key相等|CPU通过；GPU待执行|
|无训练 eval 字段|stage18_support.validate_task|白名单拒绝 H_eval/额外字段/缺H|CPU通过|
|保存/恢复|每20步同一 checkpoint 保存Adam/RNG/curve|恢复下一步与连续执行逐张量相同|CPU通过|

没有加入旧草案额外模块，没有把 C_NO_H 改名为 C_FACT。Stage18 入口独立，未改写 Stage17 代码/报告/结果。GPU3 后续训练已有用户授权；数据门槛独立存在。
