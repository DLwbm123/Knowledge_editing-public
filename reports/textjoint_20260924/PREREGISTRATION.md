# 文本泛化与保持联合优化：执行前锁定

> 结项说明：下文保留执行前的原始预登记。原服务器中断后，用户授权迁往 my-gpu 的 GPU 2/3；科学配置与原时间、GPU、Judge 预算未重置。最终执行范围、负结果和限制见 `final/AUDIT_REPORT_ZH_PUBLIC.md`。

本轮审阅基准为 `c26e046b3ec661be42f865e6727d7a638a320f31`。实际运行复用干净的 FreshStart 执行 overlay `7a9b1f22dc410dd8322b5813aaee935c93c38a79`，必须保存实际 Python 导入路径。Stage17 L21 的 single edit-macro T2G=91.9521%、T2L=48.3871% 只作历史参考。本轮固定 L30、rank=4、FP16 Base、FP32 writer、seed base 20260924；所有候选重训，同轮 B0 是当前无 H 方法，不能套用旧分数。

新服务器只允许物理 GPU 5/6/7，默认只用 5 一张（当前有其他用户任务，但实测剩余显存足够；启动前再检查）。入口、解释器及参数使用中性路径。预算保守从 2026-09-24 13:30 UTC 起计算：14 小时目标，16 小时墙钟/16 GPU 驻留小时硬上限；10.5 小时后不再扩张训练，至少留下 25% 用于评分报告。准备、训练、生成、评分与有效实验时长分别记录。没有有效工作时不能以等待或空驻留凑 12 小时。

数据使用旧 Stage17 已暴露 DEV，先12条，最多扩大至同序36/48条；独立 CONFIRM 缺失，结论为探索。native target、native-only P_fit、隔离辅助 U_fit、official DEV 探针分离；不使用官方探针训练、选择 U 或调路由。本次获取前48条编辑所需的318张图像，辅助来源只有29条 QA / 3个图像源；记录每条编辑真正可用的 hard/diverse U 和唯一来源数，不能重复凑8。原始图像模态保持；T2 的“文本”不能擅自解释为去除图像令牌。

六臂固定基础优化器、种子、生成与路由：B0 原wrapper/原U/原保护时点；P 仅四条语义等价native改写；E 仅从native CP训练起加原U保护；U 仅扩展U（目标hard4+diverse4）；EU early+U；PEU paraphrase+early+U。E保留每阶段旧目标，从共同介入前初始化重训；解析转换不加loss。U teacher为全专家OFF冻结Base，生成前缀全词表 KL(Base||student)，token mean，lambda=.01，跨U求均值。不引入H、HSIC、层/rank搜索或历史回放。

先完成 B0 真实训练/保存重载/生成/Judge/归并闭环，再放大。诊断 Base OFF、native CP、A2、CP_W0、rank4转换、continuation；同一探针对比 OFF、FORCED_ON、真实路由。FORCED_ON仅诊断。检查 causal mask、梯度、长度、冻结Base、转换等价、来源重叠与评分覆盖。

候选选择以同N、同题完整配对为先；最多两个进入扩大阶段。优先 B0+最佳候选完整 single/sequential；每个 sequential 前缀从当前真实专家库重新路由生成，旧专家冻结。然后 BalancEdit（明确论文规格适配实现），预算允许才加 LoRA-Perf 和监督匹配+Aug。仅明确误路由诊断支持时才加固定writer路由验证。不得使用gold ID、未来专家或纯文本一律关闭。

Judge固定既有 gpt-6-astra/high source-answer-agreement 协议，隔离环境，无工具/记忆/项目上下文。预算上限继承6000项；历史31项+未决恢复19项先保守扣除，剩余最多5950项；请求数同样保守预留，语义不重试。失败缺失保持missing，积压超过可回收量停止扩张，不能把已评分子集当全量。

主要edit-macro，另报micro、分子分母、有效编辑、来源组、missing、逐题变化、edit级配对置信区间。Fix使用同轮冻结Base错误集合，Retention使用同轮Base正确集合；输出一致性单独报告。T0≥99%、T1G相对B0不低于-1pp、T1L不新增已知正确变错误，T2G+2pp/T2L+8pp只是工程目标。loss下降不等于性能提高。

有效正负结果全部保留；checkpoint仅在最后必需消费者完成后按生命周期处理，保留活跃恢复状态与最终最佳必要checkpoint及hash。只公开源码与脱敏聚合；私有题目、图像、逐题Judge、权重、凭证不得发布。
