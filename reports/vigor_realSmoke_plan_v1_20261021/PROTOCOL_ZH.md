# 真实VLM系数与KV缓存烟测：具体配置与停止条件

**方法与预算已固定，具体合法源/原输入绑定仍缺，故不是可执行launch协议。** 本轮只有一次有界静态读取，无模型、GPU、Judge或toy。SMOKE_CONFIG.json是唯一参数清单；新来源未建立前，不调用真实设备。

## 输入与接口

复用既有LLaVA-Med-v1.5-Mistral-7B和runtime commit（见配置），唯一layer6 down_proj 4096×14336、native FP16，原layer6 F0及绑定不重新校准。只取一个用途明确的新源；其用途、split、暴露/排除、原图/问题/答案、规范预处理与既有完整生成参数必须在私有清单绑定。当前source_binding=null，不用旧E01等补位、不改163保留或38未知角色、不生成新题、不下载新数据。

当前源码实际看到：EditBatch.forward_kwargs令use_cache=False；runtime.generate(record,use_cache=...)显式转发配置，并返回raw_token_ids及sequence_contract。定向追加读取已定位adapter和安装版transformers4.51.3：adapter白名单确实将use_cache传给model.generate，Mistral在use_cache且无past时构造DynamicCache，支持get_seq_length。实际加载的LLaVA子类及prepare_inputs_for_generation覆盖仍需在启动时核绑定，不能把库源码等同已观测真实缓存。

烟测不训练。先用原editable_fp32只读路径取得当前源完整NLL梯度G，列范数最大（同分首项）列规范化为u，再将Gᵀu规范化为v，构造一个诊断方向；无SVD、迭代、选层或阈值扫描。若梯度/投影为0或非有限则停止。A=0时检验新coefficient_forward与原FP32路径：source NLL绝对差≤1e-5，系数梯度与uᵀGv比较atol1e-6/rtol1e-3。这个rank1仅为接口诊断维度，不是IND2/JOINT2的编辑秩。

非零诊断系数固定为1e-4×||W0||F，非toy学习率或多样性阈值。计算W0+uAvᵀ再一次cast FP16，实际native写入必须非零，折叠/非有限即失败，不改尺度救一致性。此矩阵仅临时烟测，不当合规编辑候选；原visual/null/F0 trust数值完整记录，但不是本轮纠错/损伤结论。移除临时系数forward后再官方生成，禁止带临时FP32 hook生成。

## 真正的KV路径对照

同一native矩阵、同一问题图像，按序调用runtime.generate(use_cache=True)、runtime.generate(use_cache=False)，do_sample=False、num_beams=1、max_new_tokens=1024，其余精确沿用既有runtime冻结配置；不能缩短输出或修改prompt造长解码。每请求最多128模型forward，必须自然完整终止，触顶/空输出/未终止保留失败，不拼接partial。

只读model前后观察器记录prefill与decode输入长度、past_key_values是否非空、实际Cache类/序列长度，私有token/logits只留RAM。Cache允许已核安装版tuple或Cache接口；不识别则STOP_UNRESOLVED，不能调用未经核实的属性。缓存臂至少一次真实非空past、单token decode；无缓存臂不出现非空past且序列长度增长。自然输出没有decode则CACHE_NOT_EXERCISED，不改prompt重跑。

两完整token序列与终止状态必须精确相同；在相同prefix下对齐logits最大绝对差≤.05且相对RMS≤.001，范数分母max(||reference||F,1e-12)。只报告标量。任何超容差/argmax分歧/缓存结构异常均停止并保留，禁止放宽门或扫系数。FP32分解与native合并的差异单列测量，不能用其数学等价宣称数值精确一致。

## 预算、恢复、失败

整体从首次实际执行起1h；全模型forward硬320（工作300、恢复20），其中完整生成≤256，其他工作≤44；backward≤4、优化0/Judge0/checkpoint0。GPU工作240秒、TERM270秒、KILL300秒，PID出生tick独立guard。模型加载/准入/生成/恢复成本全部计，超过工作预算即进入恢复，不等心跳。

固定物理GPU2及原UUID，启动时≥48GiB可用、工作余量12GiB，owned allocated不得超36GiB；若预估无法容纳完整当前梯度/原矩阵/组合FP32/native副本/模型激活/两请求KV峰值则不启动，不靠toy RSS替代。44非生成工作forward可覆盖基座visual/null、旧FP32梯度、新系数梯度、非零连续/native损失与prefill、异常恢复及最终W0复核；实际逐事件计账，不将这些额度作为必须用完的调用数。

所有远端文件含临时限既定data根；root240MiB+其他16MiB=256MiB，无权重/梯度/激活/文本公开或checkpoint。挂载余量与写读探针、磁盘基线、中性主子argv、ps和nvidia实采、独立guard、独占reservation是实际启动前提。无GPU部署器已运行。

正常/异常/非有限/超预算均恢复W0矩阵精确相同、native dtype、flags、模式、hooks、grad、generation config；临时KV对象释放且不带入第二请求，allocator缓存可存在但不保存KV引用。最终W0 visual/null NLL与初始绝对差≤1e-5。恢复失败优先报告并停止整轮，不重启旧run或重试烟测。

## IND2/JOINT2的边界

合法源和生产设备接口准入后，独立冻结真实两臂方案；双成功是H1结果，不是launch前提。真实优化rank/lr/步/初始化/罚项/多样性单位另立一次冻结，不能从本烟测rank1或1e-4诊断幅度推导，也不照搬Q1 .02/.25/64。关系不足可只准入源端H1，不作H2/H3；真实行为/选择面板隔离后才评价保护。没有合格来源时保留来源缺口，不重复普查、改变角色或用更多toy替代。
