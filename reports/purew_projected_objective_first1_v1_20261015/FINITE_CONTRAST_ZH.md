# 最小有限对照：PM / PU，复用SH00

仅冻结准备；需要后续明确执行授权、生产检查与真实VLM准入。旧SH/R与失败成本保留；原146/M暂停、O/L原屏障等待。无新周期监测。

## 设计与可归因范围

|臂|原始训练视图答案位置|语言prefix|执行|
|---|---|---|---|
|SH00|原CE|无|复用已交付87消费者，不重跑|
|PM|KL(Bern(a*)||Bern(a))|无|未来首源20槽|
|PU|KL(p*||p)|KL(p0||p)，类权重1|未来首源20槽|

a*=max(a0,.95)。两新臂从原Base开始，原首源/seed、native和原4FIT、.5native+.5cyclic FIT，顺序PM→PU。名称不沿用已阻塞的N/NJ，后者不获授权。

PM−SH00检验由CE改为有界投影边际目标；包括梯度尺度/超调惩罚的变化，不能称只有停止阈值效应。PU−PM检验完整非目标保护政策（答案条件分布+语言prefix作为同一政策）；这两部分不单独归因，不额外增加第三臂。PU−SH00检验整体取舍。旧SH10/SH01/R仅提供机制背景，不替代上述同背景参照。

必须逐项验证除loss外的source/seed/tokenization/FP16/native/优化器/二次Fisher/Armijo/计数链与SH00一致；若不一致，该历史比较标NON_IDENTICAL_BACKGROUND，不偷偷增加重跑基线。PM/PU目标都有相同α；只有PU有q和prefix保护。完整CE、实际生成、逐题损伤统一测量，不用各臂训练总loss比较拟合。

## 掩码与teacher

以实际VLM processor展开后的序列位置定义，先做因果shift。答案mask完全复用原labels!=ignore（含原EOS规则，不新增/删除token）。prefix是原system/user/prompt中的普通语言token预测位置，前后attention均有效，且不属于答案、padding、图像占位/patch或任何非语言特殊token；不能仅取“labels==ignore”当prefix。最后prompt位置若预测答案属于答案mask而非prefix。词表分布仍为完整模型词表，只在q中排除该位置目标token。

使用当前源开始前原生FP16 Base在同一5个原始训练视图、相同teacher-forced答案前缀的teacher；detach，不更新。所有原answer token均有p*，不按难度/损伤挑选。prefix为空或映射不明确视为数据/实现准入失败，不默默置零。生产前验证两mask不交、无图像/padding、非空和原答案逐token对齐。teacher forward及缓存占用计费；不是新增生成或Judge。

用稳定log_softmax/masked logsumexp计算a/q和KL，不用1-p直接相减、不用概率floor改变数学目标；非有限值保留失败并停止。不得用评价输入、答案、评分或保护错误构造p*、alpha、prefix或统计。当前窗口teacher仅临时存在；不保留为旧病例重放库。既有只读研究数据/历史输出不因本实验删除。

## 共同控制

沿用SH00 S0H0：无removed软KL、无当前五视图实际硬门；保留Base16与写入前removed4 Fisher、各.01预算，标HARD_KL_NOT_ENFORCED。实际removed KL只读诊断，不能反过来拒绝或训练。layer21 down_proj原FP16W，临时FP32master/梯度，FP64统计；eta .01/clip5/scale128/proposal cap .05/Gamma I；二次solve48迭代/8线搜，8候选回溯，Armijo1e-4按各臂自身目标判断。reject/zero消耗slot且回滚master/native，不补槽。原源末合格统计最多38行规则不变；首源之后不训练。

拟合门仍完整native+allFIT CE<.1；十个旧证据间隔全严格正有限仍保留为证据敏感性资格门，不在新loss中强迫它通过。任何失败保持失败，不借“不是临床证书”移除旧要求。逐题零新增损伤门不变。

## 指标、失败和预定决策

主问题：PU能否在CE<.1及T0/T1G全正确时，达到固定34T2L+2T1L零Base损伤且无未知？同时PU的逐编辑T2G差下界相对SH00、PM都非负。任一不满足，不称科学满意，也不扩146。

slot20固定完整CE、T2G全4分母、所有五任务、固定保护和相对新增/修复同时报告。PM/PU各87消费者=prefix1 46+实际插入5+固定36；共174新，加SH00旧87为261主分析消费者。旧SH10/SH01/R摘要引用、不追加评分。全绑定相同有效和永久missing共享；未生成/未尝试不算永久missing。差界先合并同payload未知变量；10000次seed20260912配对edit/source区间，单源无总体推广解释。prefix1局部性0支持与固定36分开。

每槽记录完整CE、边际KL、加权/未加权qKL、prefixKL、各答案a-a*与超调、当前五视图KL、Fisher各块风险/数值包络、接受拒绝/零、梯度范数与native/量化/path/net。PM的q/prefix仅诊断、无反传；初始和终点记录边际/条件/prefix参数梯度范数及余弦，只有当前训练源，计入反传预算。禁止用评价梯度诊断。记录first CE<1/.5/.1、目标概率达标率、全损伤向量匿名汇总。

CE>=.1且末10槽全零才HIGH_LOSS_FROZEN；高CE但有更新是EDIT_NOT_FIT。连续5次仅数值包络拒绝且高CE仍机制失败；所有.01和原数值门不放宽。有限求解失败不推断无方向。机械/数据/存储/预算异常阻塞链；单臂科学止步在安全终点评价后允许另一个预登记臂，不能更换source或目标。20槽以前不可恢复终止时slot20主指标NA，并交付actual partial，不回放。

若PU改善保护但降低T2G，报告取舍，不能以平均收益补偿损伤；若PM改善但PU无附加收益，反对当前保护政策；若PU的q/prefix均低而外部保护仍失败，优先怀疑覆盖而非自动提高系数；若q项实际贡献过小，只得“投影质量系数不足”的后续假说，不本轮重加权。结果不触发任何自动下一轮。

## 预算与准入

新独立预算只在以后明确授权的唯一安装/激活开始；当前未激活。总4累计GPU进程小时/24墙钟，每臂独立2GPU小时，不转移；40正式槽/含2临时准入最多42写入、600反传、30000前向（含vision不重复相加）、184生成、174新Judge、320候选、0主动集fallback。每臂20/21/300/15000/92/87/160。teacher及诊断、保存、失败、驻留均计费；不借旧预算，不预留硬中断重算。

96GiB自产矩阵含原子临时与当前teacher、96GiB主机可用、8GiB盘余量、45GiB空闲准入/43GiB缓存预留/45GiB自产GPU。GPU未分配，未来硬件必须由执行授权单独绑定；不沿用旧“任意GPU”作为自动启动许可。

生产CPU必须验证p*梯度/极端logit、实际VLM mask、PM保护梯度为零、PU完整分解、候选回滚、完整状态恢复/损坏clock拒绝、原SH00路径一致、261消费共享缺分/损伤门、所有计数硬限。然后存储显存检查，两个臂各Base3精确输出和真实原W更新/上游与结构不变；全部通过才正式槽。当前仅小型合成数学检查，不等任何生产或VLM准入。

全部实际臂终态、各最后W新进程2题精确重载/FINAL_CONSUMED、所有GPU租约关闭后GENERATION_COMPLETE才评分。gpt-6-astra medium/原Stage17隔离、每新payload一次、失败不补、连续3新传输失败停止；四worker与全PENDING/RESERVED0/实际终态评分/ALL_WORKERS_COMPLETE共同闭环。全部实际输出评分后公开源码/匿名报告/缺分边界/成本，验证SHA和匿名访问。本轮不创建这些执行器/进程或队列。
