# 医学视觉语言模型知识编辑项目总结

更新日期 2026-11-26。项目仓库名为 Knowledge_editing，MedTRACE 是这一研究系列的早期名称。本文以 2026-11-06 选定的 **完整 MDU150 专家库**为主，采用 **MED 初始化、MDU 续训、原 R0 路由**，论文结果统一使用 CORE 口径。[正式版本登记][selection] · [最终实验报告][final]

## 1. 项目目标与主要结果

项目研究如何在冻结的医学视觉语言模型中修正一条图像问答知识，使修正能够迁移到同义问法和语义保持的图像变化，同时尽量保留其他原本正确的判断。方法为每条编辑训练一个小型 Tensor Train（TT）残差专家，用 MED 约束写入对已有判断的影响，用 MDU 加强同一答案在不同问法下的片段学习，再由 R0 决定是否启用专家。

在已曝光的 **146+4 开发队列**最终全库上，MDU150 的 **T2G 改写泛化为 85.61%**，**T2L 局部性为 96.77%**。相对 MED，两项分别变化 **+7.50** 和 **−1.61** 个百分点。BalancEdit 的 T2G 更高（94.83%），MDU150 的 T2L 更高（96.77% 对 63.44%）。这些分数刻画了两种方法在该开发队列上的泛化与局部性取舍。[最终实验报告][final] · [机器结果与配对统计][main-json]

## 2. 方法细节

### 2.1 写入结构与训练顺序

基础模型为 **LLaVA-Med v1.5 Mistral-7B**，全部参数冻结。每位专家只在 `model.layers.21.mlp.down_proj`（从零计数，第 22 个 Transformer 块）增加一条残差支路。每条编辑使用一张图、原始提问和四个 FIT 同义改写，五个视图共享目标答案。单步取原问及一个按固定顺序轮换的 FIT 问法。

训练顺序为 **C 校准 160 步 → 从相同初态另训 MED 160 步 → 从 MED 终态续训 MDU 160 步 → 冻结并入库**。C 提供校准量和逐步算子更新幅度，MED 重新从零函数初态出发。已具备 MED 终态时，新增 MDU 训练量是每位专家 160 步。[训练预登记][mdu-prereg] · [C/MED 训练代码][med-code] · [MDU 训练代码][mdu-code]

将 $4096\times14336$ 的写入算子拆为输出模式 $64\times64$ 和输入模式 $112\times128$。四个 TT 核的形状为 $1\times64\times8$、$8\times64\times4$、$4\times112\times8$、$8\times128\times1$，共 **7,168 个可训练参数**。收缩前两核得到 $B_e\in\mathbb{R}^{4096\times4}$，后两核得到 $A_e\in\mathbb{R}^{4\times14336}$，因此 $B_eA_e$ 的矩阵秩至多为 4。完整 150 位专家共 **1,075,200 个 TT 参数**，路由与运行时开销另计。[TT 参数化代码][tt]

$$
\bar x=\frac{x}{\sqrt{\operatorname{mean}(x^2)}+10^{-6}},
\qquad h_e=Wx+B_eA_e\bar x.
$$

原支路使用 $x$，残差支路使用归一化输入 $\bar x$。初始化将第一个核置零，其余核按种子做非零初始化，因此初始残差为零。训练时依据移位后的目标标签掩码在预测答案 token 的位置施加残差。[TT 参数化代码][tt] · [C/MED 训练代码][med-code]

### 2.2 MED 初始化

MED 在来源问答的交叉熵写入之外，约束已有医学判断的局部响应。61 个原 BASIS 保护项里，31 个使用已核验的二元解剖结构出现命题，另 30 个沿用原 FULL_CTX 曲率。对每个二元命题，计算每个类别下两个同义表述的平均 token 对数概率（含 EOS），用 log-mean-exp 聚合类别分数，再取两个类别的分数差。令 $\widehat J_a$ 为该分数差对残差算子的导数，经同锚点 FULL_CTX 响应能量定标。保护项对算子变化在这些方向上的投影取平方，再加入其余保护项。

$$
R_{\mathrm{MED}}(\Delta W)
= \sum_a \langle \widehat J_a,\Delta W\rangle_F^2
  + R_{\mathrm{fallback}}(\Delta W),\qquad \Delta W=B_eA_e.
$$

C 只用原问和 FIT 的平均交叉熵训练，记录每条编辑的来源增益、终态 FULL_CTX 惩罚及逐步算子步幅。MED 的保护系数为 $0.1\times\text{C 来源增益}/\text{C 终态 FULL\_CTX 惩罚}$。MED 分别计算来源梯度与保护梯度，对保护梯度统一限幅，并在输入核、输出核两组中移除与来源梯度相冲突的投影分量。随后做全局梯度裁剪、分组标量二阶矩的 Canonical RMS 更新、C 轨迹算子步幅匹配及 TT 规范化。这些约束直接覆盖已有保护样本及零残差附近的响应。[MED 原146预登记][med-prereg] · [医学命题与曲率代码][med-curvature] · [梯度与校准代码][med-math]

### 2.3 MDU 两视图片段稳健续训

MDU 从各自的 MED 终态出发。答案正文按 token 前后各半切成两段，单 token 答案用一段，分段排除 EOS。设 $\ell_{v,k}$ 为视图 $v$ 在第 $k$ 段的平均 token 负对数似然，$\mathrm{CE}_v$ 为原训练标签掩码下的整答交叉熵。固定 $\lambda=\tau=1$，目标为

$$
L_{\mathrm{MDU}}
=\tfrac12(\mathrm{CE}_0+\mathrm{CE}_1)
+\frac{\lambda}{K}\sum_{k=1}^{K}
\tau\log\left(\frac{e^{\ell_{0,k}/\tau}+e^{\ell_{1,k}/\tau}}{2}\right).
$$

逐段对两视图损失做 softmax，较难视图获得较大梯度权重。例如某段原问和 FIT 的损失分别为 0.2、1.2，额外片段梯度的权重约为 27%、73%。不同片段可强调不同视图。实现先用无梯度前向得到权重，再在同一专家状态下重算各视图并累积梯度，同时检查重算损失。分段依据 token 位置，医学实体、病征和否定关系没有显式标注。[稳健目标代码][loss-code] · [MDU 训练代码][mdu-code]

续训使用 Adam，输出核学习率 $10^{-3}$，输入核 $10^{-4}$，$\beta=(0.9,0.999)$，$\epsilon=10^{-8}$，无权重衰减，全局梯度裁剪阈值 1。保护训练的影响通过 MED 初始化继承，续训目标只含整答交叉熵和片段稳健项。[TT 优化器][tt] · [MDU 训练代码][mdu-code]

### 2.4 原 R0 路由与连续编辑

R0 沿用 BalancEdit 的锚点、距离和半径规则。未激活专家的 Base 从末层 MLP `up_proj` 输入取均值特征作为查询键。每条编辑保存原问键和半径，半径用同义改写正锚点和原问配黑图的负锚点，两项欧氏距离按 0.2 和 0.8 加权。推理先在专家库找欧氏距离最近的条目，随后只核查该条目的半径。通过则启用该专家，否则回落 Base。每次最多启用一个专家。连续插入时旧专家权重保持冻结，新增键仍可能改变最近邻归属，从而改变旧查询的回答。[R0 锚点与键提取][anchors] · [R0 最近邻与半径代码][router] · [MED150 路由绑定][med-code]

完整 MDU150 沿用冻结的 MED150 路由，主队列 **2,148/2,148 个事件均为 ON，0 个回落 Base**。因此本面板的局部性是在专家激活时测得的，无法验证半径的范围外拒绝能力。激活后的专家选择仍可能影响局部性。代码对参与注意力的图文 token 求均值，图像 token 数量多只能提出潜在偏重，实际图文贡献需用模态遮蔽或键和路由敏感性核查。[原 R0 激活审计][modal-margin] · [MDU 全库路由绑定][bank-binding] · [均值键代码][pooling]

## 3. 主实验与对照

### 3.1 队列、指标和执行

主队列由原 146 条开发编辑和 4 条新增开发编辑组成。最终库冻结 **2,148 个评测事件**，各角色在资格筛选后有各自的支持量。原 101 个已训练 MDU 专家和对应的 1,513 条预测经绑定复用，其余 49 位各续训 160 步，合计 7,840 次更新、635 条实际生成。随后对 T2G 的 35 个既有未知评分载荷补评，35 个均有效，16 个 CORE 正确。该步骤沿用已冻结的预测。[全库实验报告][bank] · [T2G 补评执行审计][audit]

CORE 判断回答的核心语义是否与参考一致，并检查核心矛盾以及极性、实体、部位、模态、数值等错误。百分比先在每条编辑内计算，再对有该角色支持的编辑等权平均。UNKNOWN 对应的区间表示将未知项分别赋为错与对得到的分数范围。论文统一使用 CORE。T2L 的 CORE 资格下，未编辑 Base 的 FULL 为 0%，该面板缺乏用于衡量严格整答保持的正例支持，FULL 原值留在实验档案。T2G 补评记录使用 `gpt-6-astra` medium，未记录不可变模型快照，历史尝试账本亦不完整。[定版与评分口径][selection] · [Base 复聚合][base] · [评分审计][audit] · [冻结任务定义][benchmark]

| 角色 | 评估内容 | 事件 / 编辑 | MDU150 CORE |
|---|---|---:|---:|
| T0 | 修正 Base 原本答错的原始知识问答 | 150 / 150 | 100.00% |
| FIT | 训练所用的四个同义问法的拟合 | 600 / 150 | 100.00% |
| T1G | 同一问题在语义保持的影像变化上的泛化 | 578 / 146 | 86.99–88.70%（10 项未知） |
| T2G | 同图下合格同义改写题的泛化 | 571 / 150 | **85.61%** |
| T1L | 保留同一问题在其他影像上的原有正确回答 | 2 / 1 | 100.00% |
| T2L | 保留相关病征下其他问题的原有正确回答 | 54 / 31 | **96.77%** |

### 3.2 同队列最终库比较

| 方法 | T0 CORE | T2G CORE | T2L CORE |
|---|---:|---:|---:|
| 未编辑 Base | 0.00% | 0.17% | 100.00% |
| MEMIT-MM 固定适配 | 1.33% | 1.33% | 0.00% |
| AlphaEdit-MM 固定适配 | 84.00% | 28.67% | 0.00% |
| GRACE 多模态修正版 | 90.00% | 0.17–1.33% | 100.00% |
| BalancEdit 固定适配 | 100.00% | 94.83% | 63.44% |
| MED（V133） | 100.00% | 78.11% | 98.39% |
| MDU101＋49 位原 MED 回退 | 100.00% | 83.50% | 96.77% |
| **MDU150** | **100.00%** | **85.61%** | **96.77%** |

- **相对 MED 的逐事件变化。** T2G 有 46 项由错变对、4 项由对变错，编辑等权宏平均提高 7.50 个百分点。按编辑重采样的配对 bootstrap 95% 区间为 +5.28 至 +9.83 个百分点，共享专家和来源间的依赖未纳入重采样。[配对机器结果][main-json]
- **T2L 配对变化。** 相对 MED，MDU150 在 54 个事件／31 条编辑中有 1 项修复、2 项损失，宏平均差为 −1.61 个百分点，按编辑配对 bootstrap 的 95% 区间为 [−6.45, +3.23] 个百分点，方向仍有不确定性。[配对机器结果][main-json]
- **补齐 49 个 MDU 专家的收益。** 相对 MDU101 混合库，全库 T2G 多 13 项修复、2 项损失，提高 2.11 个百分点。按实际选中专家统计，这 49 位覆盖 178 个合格 T2G 事件／49 条编辑，T2G 为 81.97%，原 MED 对应为 75.51%。该路由分组的合格 T1L 和 T2L 支持均为零。[配对与分组结果][main-json]
- **对照适配的限制。** BalancEdit 与 MED 系列的运行精度不同。MEMIT-MM 最终库 2,148 条输出全部达到 1,024 token 上限。修正版 GRACE 在所有 571 个合格 T2G 和 54 个 T2L 事件上均回落 Base，其 T2G 有 7 项未知。AlphaEdit-MM 的 T2L 0% 出自该固定多层多模态适配，应连同保护损伤审计报告，方法原生实现的表现仍需另外验证。[标准对照汇总][comparators] · [GRACE 修正版路由审计][grace] · [AlphaEdit-MM 报告][alpha]
- **证据范围。** 原题与 FIT 的 100% 属于来源拟合。T2L 只覆盖 31 条编辑，T1L 仅 2 个事件，新增 4 条编辑的 T2G 是构造改写。MDU150 在补评结果可见后定版，整个主队列已曝光。这里衡量冻结参考下的开发集表现，独立病例泛化、完整回答支持和临床适用性仍缺验证。[主结果登记][selection] · [队列与资格汇总][comparators]

### 3.3 专家参数与存储

按每个参数 4 字节计算，下面为编辑状态的纯 FP32 参数体积，单位 MB 和 GB 均为十进制。

| 写入方式 | 每条编辑参数量 | 相对 TT 参数量 | 150 条编辑的纯权重 |
|---|---:|---:|---:|
| MED／MDU 的 TT 专家 | 7,168 | 1 倍 | 4.30 MB |
| 项目中的 BalancEdit 完整末层 up-projection | 58,720,256 | 8,192 倍 | 35.23 GB |
| 同位置的普通秩 4 低秩矩阵（规格计算） | 73,728 | 10.29 倍 | 44.24 MB |

BalancEdit 每次编辑复制完整的末层 `up_proj` 作为 FP32 独立写入器，原执行审计的 150 个写入器文件为 **35.24 GB**。表中 MDU 4.30 MB 是由核参数量换算的纯权重体积，完整文件、路由和推理开销需另测。两种实测方法共享 R0 规则，但写入层、训练配方和精度均有差异，参数和结果可用于比较工程成本与开发集工作点，单独的 TT 因果效果仍需匹配控制。[BalancEdit 写入代码][routed-linear] · [存储实测审计][balanced-audit] · [精度与协议][comparators]

## 4. 历史结果、消融与扩展

| 实验 | 结果与解释 |
|---|---|
| 原 146 的 MED 与 M13P-160 | 单专家 T2G CORE 为 78.94% 对 78.08%，最终 146 库为 77.51% 对 76.66%。最终差值 +0.86 个百分点，95% 配对区间跨零，未达到预登记的 +3 个百分点。M13P-160 的结果使用冻结历史输出与补评。 [原146对照][original] |
| MED-RA V134 | 原 146 单专家 T2G CORE 为 63.47%，同期 MED 为 78.94%。科学筛选未通过，顺序阶段未执行。 [MED-RA 负结果][medra] |
| 同 MED 初态的普通 CE 消融（MCE） | 已完成的条件是 101 位续训专家＋49 位原 MED 回退。MCE 和 MDU 的 T2G CORE 均为 83.50%，T2L 均为 96.77%，T2G 逐事件互有 1 项修复和 1 项损失。完整 MCE150 缺同条件结果，当前无法单独归因 150 库提升有多少来自片段稳健项。 [101续训补评][mce] · [配对分析][attribution] · [MCE150 预登记][mce150] |
| T4 单列开发面板 | 最终 67 专家库的 T4G 修复率为 MDU 14.43%、BalancEdit 42.56%、GRACE 50.48%、LoRA 64.71%，支持 95 事件／32 编辑。四方法 T4L 均为 100%，支持 42 事件／35 编辑。 [T4 机器结果][t4] |
| T5 发布规范对齐 | 159 个 Base 既往回答确定错误的影像对，主指标为随访准确率变化 $\Delta\mathrm{ACC}$。最终库 MDU 的随访 CORE 为 97.48%，$\Delta\mathrm{ACC}$ 为 +89.94 至 +96.86 个百分点，BalancEdit 和 LoRA 同分。恒定回答 Yes 亦为 97.48%，且另有 154 个既往资格未知；结果无法确认模型利用了时序影像变化。 [口径修正][t5-protocol] · [T5 机器结果][t5] |
| PadChest 双图修正版 | 早期错误输入版本的能力结论已撤回。修正 16 位影像显示转换后，MDU 与 BalancEdit 的未见释义均为 8/8，三类变化探针均为 4/12，来源题与三类探针全对均为 0/4。MDU 的 12 次探针评价均输出所选专家的来源类别，其中 6 次评价跨病征选错专家。四个冻结专家各跨 10 个输入都复述自己的来源类别，交换影像时序后 9/10 的路由和输出不变。 [双图修正版][longitudinal] · [部署诊断][temporal] |
| 条件路由与保护投影 | 后续设计按病征及真实负病例校准半径，在归一化输入上投影去除保护方向，并计划以同条件 CE 对照 MDU。真实预检通过数据和训练侧准入，生成器的非零残差断言与保护投影的预期零残差冲突。公开结果登记为工程未完成，完成的训练对照为零。 [预检与工程记录][conditional] · [状态结果][conditional-json] |

## 5. 机制复核与后续验证

### 5.1 已有结果能把问题定位到哪里

- **MDU 的动态权重是待检验的机制。** 如果同段两视图损失接近，softmax 权重就接近各半，MDU 新增项逼近一个等视图、等片段加权的正文交叉熵。它与原整答 CE 仍可能因段长度和 EOS 权重而有不同梯度方向。MED 的 T0/FIT 生成全对，只能证明按 CORE 答对，无法推断教师强制下的 token 损失很小、两视图接近，或 Adam 的有效更新几乎相同。现有 101+49 同条件对照支持进一步区分普通续训、额外正文监督和动态加权这三种作用。[MDU 训练代码][mdu-code] · [101续训配对][mce] · [配对分析][attribution]
- **MED 原 146 的 T2G 主要缺口已见于单专家。** 最终库在 557 个合格事件中有 540 个选中目标专家。115 个单专家与全库都答错的事件中，112 个仍选中目标专家且评分载荷相同。另有 10 项由对变错、2 项由错变对，都发生在改选专家的组。这个配对把主要缺口定位到专家已激活的情形，写入目标、表示、生成和问题语义都仍可能影响结果。125 个全库判错事件里 54 个回答含不确定或不能判断的关键词，这只记录回答语言。早期前 20 个编辑曾发现能力或不确定性问句更难，但使用 M13 和 FISHER_MIN，完整 MDU150 的冻结分层结果见下节。[原146配对诊断][decomposition] · [既有判错回答审阅][error-review] · [前20例问法审计][early-subtypes]
- **T1G 对专家竞争更敏感。** 原 146 MED 单专家为 578/578，最终原 R0 库为 501/578。固定专家做精确问句切换回放后，T1G 为 99.32%，原 146 去重非来源题中 80 个 Base 与 R0 共同正确的查询新增损伤为 0，T2G 未提高。T1G 本身保留原问，精确匹配收益可能依赖任务构造。另一条模态分差规则虽将 T1G 提至 99.66%，却带来 4 项新增 CORE 损伤，T2L 由 98.39% 降至 91.94%。现有结果支持单独验证文本条件路由的收益与损伤。[精确问句回放][exact-switch] · [模态分差][modal-margin] · [原146单专家与全库][original]
- **跨病例迁移仍是开放问题。** T4G 测的是组合泛化，PadChest 则暴露了有序双图的小面板上按来源类别作答的问题。两者共同限制广义的知识迁移主张。PadChest 的强制专家诊断含超出专家自然适用范围的输入，结果只描述该面板上的行为。[上游任务定义][benchmark] · [T4结果][t4] · [双图部署诊断][temporal]

### 5.2 A1 冻结 T2G 按问法类型分层

使用 MDU150 主表的同一份冻结 CORE 判定与资格掩码，按查询 ID 逐项匹配三种方法。原 146 条编辑的 557 个合格事件均可一对一匹配 Core-9 谱系中原始探针 ID 的类型标记，覆盖已知类型的 557/557 个事件。新增 4 条编辑的 14 个合格构造题在冻结记录中有身份但缺少四类标签，统一列为“类型不明”。问法类型取自生成谱系，先匹配类型再读分数。仅对冻结 CORE 布尔值做离线汇总，三种方法各 571/571 项有有效评分，原主表得分及已公布的成对得失均复现。[上游四类定义][benchmark] · [探针谱系绑定代码][a1-lineage] · [新增探针来源][a1-extension] · [冻结主结果][main-json]

**CORE 编辑等权宏平均。** 括号为错误事件数。四个已标注类别均为每个支持编辑一个事件，所以这些行的宏平均等于事件正确率。合计按原方案对 150 条编辑等权，不能用各类型的事件数直接加权还原。

| 问法类型 | 事件 / 编辑 | MED CORE | MDU150 CORE | BalancEdit CORE |
|---|---:|---:|---:|---:|
| 同义改写 | 140 / 140 | 94.29%（错 8） | 95.00%（错 7） | 97.14%（错 4） |
| 否定／不确定 | 141 / 141 | 48.94%（错 72） | 62.41%（错 53） | 93.62%（错 9） |
| 数值扰动 | 131 / 131 | 70.23%（错 39） | 85.50%（错 19） | 91.60%（错 11） |
| 干扰语插入 | 145 / 145 | 95.86%（错 6） | 97.24%（错 4） | 95.17%（错 7） |
| 类型不明（新增构造题） | 14 / 4 | 100.00%（错 0） | 100.00%（错 0） | 100.00%（错 0） |
| 全体 | 571 / 150 | 78.11%（错 125） | 85.61%（错 83） | 94.83%（错 31） |

**配对得失。** 每格为“由错变对 / 由对变错”，括号是后一个方法减前一个方法的 CORE 宏平均差，单位为百分点。三种配对均无未知评分。

| 问法类型 | 事件 / 编辑 | MED → MDU150 | MED → BalancEdit | MDU150 → BalancEdit |
|---|---:|---:|---:|---:|
| 同义改写 | 140 / 140 | 2 / 1（+0.71 pp） | 7 / 3（+2.86 pp） | 6 / 3（+2.14 pp） |
| 否定／不确定 | 141 / 141 | 21 / 2（+13.48 pp） | 65 / 2（+44.68 pp） | 50 / 6（+31.21 pp） |
| 数值扰动 | 131 / 131 | 21 / 1（+15.27 pp） | 34 / 6（+21.37 pp） | 16 / 8（+6.11 pp） |
| 干扰语插入 | 145 / 145 | 2 / 0（+1.38 pp） | 2 / 3（−0.69 pp） | 0 / 3（−2.07 pp） |
| 类型不明（新增构造题） | 14 / 4 | 0 / 0（0.00 pp） | 0 / 0（0.00 pp） | 0 / 0（0.00 pp） |
| 全体 | 571 / 150 | 46 / 4（+7.50 pp） | 108 / 14（+16.72 pp） | 72 / 20（+9.22 pp） |

**错误确实集中在原标注的“否定／不确定”组，对 MED 和 MDU150 尤为明显。** 该组只占已标注事件的 141/557（25.3%），却占 MED 的 72/125（57.6%）和 MDU150 的 53/83（63.9%）错误。两者在此组的错误率分别为 51.06% 和 37.59%，均为各自四类中最高。BalancEdit 该组只占 9/31（29.0%）错误，且它的最低分落在数值扰动（91.60%）。因此集中现象主要见于 MED 系列，在这批结果里没有普遍出现在三种方法中。

MDU150 相对 MED 的 46 项修复里，有 42 项来自否定／不确定和数值扰动，各 21 项。MDU150 在干扰语插入上领先 BalancEdit 2.07 个百分点，但在否定／不确定组仍落后 31.21 个百分点。这一组是两者 T2G 差距最集中的位置。原标签合并了否定与不确定，数据只支持合并组的结论。分层沿用冻结开发队列的资格筛选，供定位问题，不能据此区分纯不确定句、证明语言形式造成失败或推断独立队列表现。

### 5.3 A2 冻结专家残差范数与片段权重

用 101 条共同编辑做 MED、MCE、MDU 三方配对，另对只有 MED、MDU 的 49 条编辑做两方配对。三方配对由早期实际续训的 25 位专家和随后实际续训的 76 位专家组成。逐一核验 MCE、MDU 的共同 MED 初态、原始训练视图和 token 锁，续训权重须与原完成回执一致。范数在 CPU 上用 FP64 对 TT 因子收缩，计算同层残差支路的等效矩阵 $\Delta W_e=B_eA_e$。该范数对应归一化输入上的线性算子，实际输入条件下的残差幅度仍取决于表示方向。

| 配对范围 | 编辑数 | MED $\lVert\Delta W\rVert_F$ | MCE $\lVert\Delta W\rVert_F$ | MDU $\lVert\Delta W\rVert_F$ |
|---|---:|---:|---:|---:|
| 三方配对 | 101 | 3.005 [2.813, 3.234] | 3.300 [3.073, 3.574] | 3.299 [3.078, 3.572] |
| 额外 MED／MDU 配对 | 49 | 2.852 [2.644, 3.079] | 无配对权重 | 3.101 [2.928, 3.409] |
| MED／MDU 全库 | 150 | 2.995 [2.760, 3.207] | 仅有上述 101 位 | 3.262 [3.011, 3.532] |

表中为中位数 [P25, P75]。101 组内，MCE/MED 与 MDU/MED 的**逐编辑范数比**中位数分别为 1.0944 和 1.0943，MCE/MDU 续训后的范数均高于对应 MED。MDU/MCE 范数比中位数为 **1.0006**，四分位范围 [0.9993, 1.0020]。两种终态算子的余弦相似度中位数为 **0.99845**，相对共同 MED 的两个算子变化量的余弦相似度中位数为 **0.99189**。两种续训得到近似的幅度和方向，仍存在非零差异。MDU 在全 150 组的范数均高于自己的 MED 初态，其逐编辑范数比中位数为 1.0945。[TT 实现][tt] · [同条件续训实现][mdu-code] · [主方法登记][selection]

**历史训练步的偏权上界。** MDU checkpoint 保存了每步各片段的 log-mean-exp 平均值 $R_t$。记单片段值为 $r_k=\tau\log((e^{\ell_{0,k}/\tau}+e^{\ell_{1,k}/\tau})/2)$。由于 token 负对数似然非负，$0\le r_k\le KR_t$，而较大权重为 $w_{\max}=1-e^{\ell_{\min}/\tau}/(2e^{r_k/\tau})$。因此每个片段满足

$$
|w-0.5|\le \tfrac12\left(1-e^{-KR_t/\tau}\right).
$$

在历史日志的 FP32 数值上，为计算上界对非负 $R_t$ 加 $10^{-6}$ 数值余量。150 位 MDU 专家的 **24,000/24,000** 个原训练步均可复核，每步最大偏离的上界中位数为 $1.86\times10^{-6}$，P95 为 $5.30\times10^{-5}$，P99 为 $2.89\times10^{-4}$，全程最大为 **0.003273**，出现在第一步。也就是说，原训练中每个片段的动态权重均落在约 **0.4967–0.5033** 内。这里是由聚合损失得到的上界，逐片段的实际值见冻结前向。[稳健目标实现][loss-code] · [历史日志保存代码][mdu-code]

**冻结状态下的实测偏离。** GPU 6/7 分片加载原模型，用 Native 和四个原 FIT 逐一配对。保持 FP16 多模态预处理及 FP32 解码，对原答案做教师强制前向。每个视图对和片段只记一个绝对偏离，两个视图互补。MCE 终态的数字来自在其冻结状态上套用 MDU 公式。原 MCE 训练只使用 CE。全部 MED 初态的 native＋FIT1 重算 CE 和 MDU 聚合稳健损失均与对应历史首步记录一致。完成 401 个专家状态 × 5 个视图，共 **2,005 次教师强制前向**。另保留加载兼容与显存恢复记录，已完成的首批 9 条编辑直接复用。

| 配对范围 | 冻结状态 | 视图对 × 片段 | 中位数 | P95 | P99 | 最大偏离 |
|---|---|---:|---:|---:|---:|---:|
| 101 三方配对 | MED 初态 | 800 | 1.74e-6 | 4.80e-5 | 1.54e-4 | 4.78e-4 |
| 101 三方配对 | MCE 终态 | 800 | 0 | 3.00e-7 | 9.86e-7 | 2.21e-6 |
| 101 三方配对 | MDU 终态 | 800 | 0 | 2.98e-7 | 9.27e-7 | 2.06e-6 |
| 额外 49 配对 | MED 初态 | 392 | 8.94e-7 | 1.43e-5 | 3.46e-5 | 1.10e-4 |
| 额外 49 配对 | MDU 终态 | 392 | 0 | 8.94e-8 | 1.49e-7 | 2.98e-7 |
| 全部 150，仅原训练首步使用的 FIT1 | MED 初态 | 298 | 1.46e-6 | 3.16e-5 | 1.04e-4 | 3.45e-4 |

在 101 条三方配对的 505 个训练视图上，整答教师强制 CE 中位数依次为 MED 5.81e-5、MCE 3.10e-7、MDU 3.05e-7。MED 初态的目标 token 损失本身已经很低。

**结论。** MED 初态在 1,192 个原题／FIT 片段对上的最大偏离为 **4.78e-4**，MDU 终态同分母最大为 **2.06e-6**。结合历史 24,000 步的上界，当前视图、温度和初始化给出的动态加权始终很弱。MCE 与 MDU 的算子状态相对 MED 均有明显变化，两个续训方向高度相似。分段额外损失即使等权也会产生梯度，它的独立收益仍未得到隔离。论文可以完整描述原目标，将动态加权的作用写为诊断所得的弱信号。诊断期间参数更新、答案生成和重新评分均为 0。[训练日志与前向契约][mdu-code] · [片段损失实现][a2-terms]

### 5.4 A1／A2 联合复核与续训算子分解

**BalancEdit 的 T2G 净优势集中在两类问法。** 以下为冻结 CORE 的正确事件数及组内事件正确率，主表仍按编辑等权。

| 合并组 | 事件 | MED | MDU150 | BalancEdit |
|---|---:|---:|---:|---:|
| 同义改写＋干扰语插入 | 285 | 271（95.09%） | 274（96.14%） | 274（96.14%） |
| 否定／不确定＋数值扰动 | 272 | 161（59.19%） | 200（73.53%） | 252（92.65%） |

前一组两种方法各有 6 项对方答错的事件，整体微平均打平。后一组 MDU150 净少 52 项，其中否定／不确定为 44 项（84.62%），数值扰动为 8 项。14 条未标注事件三方均正确。两类弱项的题干若改变极性、数值阈值或回答谓词，原参考可能不再适用。BalancEdit 输出复述编辑目标的解释同样待核查，整组得分和 T2L 的差异不足以确立这种行为。论文可报告同义改写＋干扰语插入组的描述性结果、TT 参数量和有限 T2L 支持上的保持率，并保留另外两类的得分及评估范围。[冻结主结果][main-json] · [上游四类定义][benchmark] · [BalancEdit 实现与协议][comparators]

在否定／不确定组，BalancEdit 答对且 MDU150 答错的事件有 50 项。假定其中 22 项变为正确且其余事件判定不变，以每项 $100/(150n_e)$ 个百分点累加，其中 $n_e$ 是该编辑的合格 T2G 事件数。46 项的 $n_e=4$，另 4 项的 $n_e=3$。因此追回任意 22 项对应 **+3.67 至 +3.89 个百分点**，T2G 反事实为 **89.28%–89.50%**。这个范围由固定分母与反事实条件决定，尚无相应干预结果。[冻结主结果][main-json]

**残差范数上升主要伴随算子方向变化。** 用 A2 已保存的逐编辑范数及两个算子的余弦，按同一 FP64 口径离线复算。设 $X=\Delta W_{\mathrm{MED}}$，$Y$ 为该编辑的续训终态。把 $Y$ 投影到 $X$ 的方向后分解为 $Y=sX+Y_{\perp}$。

$$
s=\frac{\langle Y,X\rangle_F}{\lVert X\rVert_F^2},
\qquad
\rho_{\perp}=\frac{\lVert Y_{\perp}\rVert_F}{\lVert Y\rVert_F},
\qquad
Y_{\perp}=Y-sX.
$$

| 配对 | 编辑数 | $s$ 中位数 [P25, P75] | $\rho_{\perp}$ 中位数 [P25, P75] | 正交范数 / 续训变化范数，中位数 |
|---|---:|---:|---:|---:|
| MED → MCE，同条件 | 101 | 0.993 [0.973, 1.008] | 42.47% [40.47%, 44.90%] | 99.93% |
| MED → MDU，同条件 | 101 | 0.991 [0.971, 1.008] | 42.77% [41.00%, 45.36%] | 99.93% |
| MED → MDU，完整库 | 150 | 0.988 [0.972, 1.006] | 43.02% [40.99%, 45.69%] | 99.91% |

最后一列是 $\lVert Y_{\perp}\rVert_F/\lVert Y-X\rVert_F$，与 $\rho_{\perp}$ 的分母不同。MDU 全库的范数比约为 1.0945，沿 MED 方向的系数中位数却低于 1，且相对 MED 的变化几乎完全体现为正交分量。这组算子证据反对「续训近似只做等比例放大」。强度扫描仍可检验固定方向上的行为敏感性，算子分解本身无法判断哪个方向有助于 T2G。[TT 算子实现][tt] · [同条件续训代码][mdu-code]

**A2 支持弱化动态加权的论文主张。** MED 初态在原训练视图的 CE 已很低，原 MDU 在固定 $\tau=1$ 下几乎始终等权，且 101 组内 MCE 和 MDU 的得失大致持平。现有证据无法把完整 150 库的 +7.50 个百分点完全归到 CE，也无法把 46 项修复里集中于弱项的 42 项直接归因于范数增大。原方法仍应按实际执行描述为 MED 后续训，目标含整答 CE 和分段 log-mean-exp。两视图的偏离等于 $\tfrac12\tanh(|\ell_0-\ell_1|/(2\tau))$，因此当损失差非零时，降低温度仍可放大权重差。当前结果只检验了冻结配方的温度和视图。优先级上，同一已饱和视图池的等权消融可以后移，保留此前记录的同条件结果。[原 MDU 目标][loss-code] · [同条件配对诊断][attribution]

### 5.5 A3 参考有效性与冻结输出行为

**题目盲审。** 用同一冻结队列中的 141 条否定／不确定与 131 条数值扰动，共 272 个事件，覆盖 145 条编辑。两组原题与变体的冻结参考逐字相同。审核时只看原问、变体和参考，隐藏方法、预测和对错。先锁定三类标签，再读取历史 CORE。该核查判断原参考是否能回答改写后的问题，由单人完成，医学事实仍沿用冻结参考。

- **否定／不确定。** 139 条保留、2 条证据不足、0 条能够明确确认参考失效。实际有 138/141 条以「能否判断」或指定的英语判断、识别类句型开头。参考没有纯二值的 Yes/No 或是/否条目。常规用法下，多数句型是要求给出内嵌问题的实际答案。2 条证据不足来自原参考的定位描述缺少明确的解剖参照。
- **数值扰动。** 123 条保留、8 条证据不足、0 条能够明确确认参考失效。73/131 条含预定义的数字或数量标记。明确给出连续数值的两条均沿用参考中的同一测量值。其余常见操作是加上与既有答案兼容的数量提示或普通重述。这批数据未观察到阈值翻转使原答案明确失效。8 条证据不足主要涉及原定位含糊、指定数量与参考中的近义或重叠条目、以及变体要求更具体的图像属性。

**用已冻结 CORE 复算。** 每格为编辑等权宏平均，括号为事件正确数。配对格为「由错变对 / 由对变错」。全库敏感性分析只去掉上述 10 条证据不足题，保留原来 571 条的主结果，其他类型未在本轮逐题重审。

| 范围 | 事件 / 编辑 | MED CORE | MDU150 CORE | BalancEdit CORE | MED → MDU150 | MDU150 → BalancEdit | MED → BalancEdit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 否定／不确定 | 139 / 139 | 49.64%（69） | 63.31%（88） | 93.53%（130） | 21 / 2 | 48 / 6 | 63 / 2 |
| 数值扰动 | 123 / 123 | 73.98%（91） | 86.99%（107） | 91.06%（112） | 17 / 1 | 13 / 8 | 27 / 6 |
| 全库敏感性分析 | 561 / 150 | 79.56%（445） | 86.22%（483） | 94.83%（530） | 42 / 4 | 67 / 20 | 99 / 14 |

排除的 10 条题在冻结 CORE 中由 MED 答对 1 条、MDU150 答对 5 条、BalancEdit 答对 10 条。排除后，否定／不确定组的 BalancEdit–MDU150 差距由原 **31.21** 变为 **30.22 个百分点**，数值扰动组由 **6.11** 变为 **4.07 个百分点**。否定／不确定问法仍是最主要的 T2G 差距。[冻结主结果][main-json] · [探针谱系][a1-lineage]

**错答行为。** 三个方法合计检查原两组的 203 个冻结错答。将空答、明确表示无法判断且未给出具体回答、或只提供泛化背景的输出归为回避。只有在同一所问谓词上明确作出相反的有无、增强或阈值方向断言，才归为极性冲突。实体、部位、完整性和其余错误归在其他，带保留措辞但仍给出具体病例判断的也归在其他。阅览时不显示方法标签，归类用于描述输出，CORE 判定保持原值。

| 问法类型 | 方法 | 原错答 | 回避 | 极性冲突 | 其他 | 保留子集 回避 / 错答 |
|---|---|---:|---:|---:|---:|---:|
| 否定／不确定 | MED | 72 | 35 | 0 | 37 | 33 / 70 |
| 否定／不确定 | MDU150 | 53 | 26 | 1 | 26 | 24 / 51 |
| 否定／不确定 | BalancEdit | 9 | 5 | 0 | 4 | 5 / 9 |
| 数值扰动 | MED | 39 | 0 | 1 | 38 | 0 / 32 |
| 数值扰动 | MDU150 | 19 | 0 | 1 | 18 | 0 / 16 |
| 数值扰动 | BalancEdit | 11 | 0 | 0 | 11 | 0 / 11 |

在相同的 141 条否定／不确定事件上，MED、MDU150、BalancEdit 分别有 **35、26、5** 条回避。该组仅观察到 1 条明确极性冲突。数值扰动组的失败主要表现为未给全所求条目、实体或属性错误。

**编辑目标的规范化精确一致。** 对冻结 T2G 和 T2L 输出复原原评分载荷。所有 623 个不同查询 × 3 种方法，共 1,869 个输出，均验证了对应的评分键。总计 625 个评测事件，T2L 的 54 个事件涉及 52 个不同查询，按原编辑—查询口径保留。三种方法在这 625 个事件上均激活且选中同一专家。对答案和编辑目标使用 NFKC、大小写折叠、空白和首尾标点清理，并移除单个开头的 Answer/答案标签及统一列表分隔符周围的空格。随后比较整个答案文本，保留词序、数值、极性及其余正文。

| T2G 类型 | 事件 | MED 所选 / 所属 | MDU150 所选 / 所属 | BalancEdit 所选 / 所属 |
|---|---:|---:|---:|---:|
| 同义改写 | 140 | 126 / 126 | 128 / 128 | 134 / 134 |
| 否定／不确定 | 141 | 49 / 49 | 68 / 68 | 132 / 131 |
| 数值扰动 | 131 | 64 / 64 | 85 / 85 | 117 / 117 |
| 干扰语插入 | 145 | 138 / 138 | 141 / 140 | 140 / 138 |
| 类型不明 | 14 | 14 / 14 | 14 / 14 | 14 / 14 |
| **全体** | **571** | **391 / 391** | **436 / 435** | **537 / 534** |

每格依次为「与实际选中专家的目标完全一致 / 与评测所属编辑的目标完全一致」的事件数，比例用事件分母计算。否定／不确定组的实际目标一致率为 MED **34.75%**、MDU150 **48.23%**、BalancEdit **93.62%**。但全体 T2G 中有 **566/571** 个事件的冻结参考本身就与实际选中专家的目标规范化相同，因而 T2G 高一致率同时符合预期的正确泛化和复述解释。

| T2L 范围 | MED | MDU150 | BalancEdit |
|---|---:|---:|---:|
| 全部事件，与所选目标一致 | 1/54（1.85%） | 1/54（1.85%） | 38/54（70.37%） |
| CORE 错答，与所选目标一致 | 0/1 | 0/2 | 22/23（95.65%） |

以上两行中，所属目标一致的计数与所选目标相同。T2L 的冻结参考均与所选目标规范化文本不同。在已判错的 T2L 事件上，BalancEdit 的错答几乎都字面复现了选中专家的目标，这为局部性损伤提供了比 T2G 更直接的行为证据。文本不同的两个答案仍可能语义等价，且固定样本较少，现有对照不足以断言这种模式解释 BalancEdit 的全部 T2G 优势。[冻结主结果][main-json] · [BalancEdit 执行与范围][comparators]

本轮只使用冻结材料，新增训练、GPU 推理、回答生成和 CORE 评分均为 0。盲审与行为标签留在私有审计记录。报告只写汇总。

### 5.6 后续验证顺序

1. **分别验证两个弱项的错答模式。** 否定／不确定组优先检查更换疑问句式后，回避率是否下降。数值扰动组优先检查数量约束和答案覆盖。后续若对同图非目标问题加保护，使用通过来源与语义核查且 Base 已知正确的参考，并保留复述率和 CORE 两种读数。
2. **再区分残差强度与写入方向。** 强度曲线需要新生成和评分。可在固定路由下比较 $\alpha=0$、原强度回放及预先固定的强度点，同时联报分类型 T2G、T2L 和 T0。BalancEdit 的 $\alpha>1$ 属于外推。若加入 MED 的放大点，用它检验强度对行为的影响。匹配 TT 表示的秩 4 对照还应固定层位、路由、精度、监督和预算。[历史深层 LoRA 诊断][depth] · [参考目标][handoff] · [历史同层秩4规格][historical-rank4]
3. **机制改进与独立评估。** 若题目核查支持弱项确属语义泛化，再用相同 MED 初态比较原视图 MCE、加入获准新问法的 MCE、加入新问法与非目标 KL 的 MCE。答案翻转的样本需单独绑定新参考，避免混入同答案 FIT。固定方法与强度后再用获准的新队列做独立确认，继续报告 CORE、范围外激活和有害激活。主权重、历史判定和 UNKNOWN 保留可追溯。[主结果与范围][selection] · [评分审计][audit]

## 6. GitHub 和复现入口

| 用途 | 仓库或版本 |
|---|---|
| 公开研究仓库 | [DLwbm123/Knowledge_editing-public][repo]，默认分支为 `main` |
| MDU150 代码、协议、结果 | [实验分支 experiment/med150-20261027][branch]。本次核对的公开快照为 [997168ae][snapshot]，主 T2G 评分结果固定在 [7c1ac215][result-commit] |
| 历史源码备份 | [DLwbm123/Knowledge_editing][backup-repo]，GitHub 当前也标为公开。MDU150 对照和复现入口以上面的研究仓库实验分支为准 |
| 上游基准 | [M3Bench 任务定义（固定提交）][benchmark]，用于核对各任务的输入关系、资格和计分 |

复现时先阅读[定版说明][selection]和[全库预登记][mdu-prereg]，再从[实验入口目录][entry]定位 `train_worker.py`、`matched_train.py`、`eval_worker.py`、`score_driver.py` 与 `summary.py`。可直接查阅[主结果 JSON][main-json]和[执行审计][audit]。仓库包括源码、配置、测试及选定的脱敏实验报告。完整重跑还需要按原实验绑定取得影像与问答、基础模型及 MED 初始化状态、冻结 R0、运行环境和评分协议。[公开仓库 README][readme]

本总结依据已归档的报告、机器结果和实现代码整理，并核对了 GitHub 仓库与分支。本文中的实验数字来自既有执行记录。本次复核只更新文档，新增训练、推理和评分均为零。

[repo]: https://github.com/DLwbm123/Knowledge_editing-public
[branch]: https://github.com/DLwbm123/Knowledge_editing-public/tree/experiment/med150-20261027
[snapshot]: https://github.com/DLwbm123/Knowledge_editing-public/commit/997168aecbf6d44e3faff8e27fbc529ff9acf955
[result-commit]: https://github.com/DLwbm123/Knowledge_editing-public/commit/7c1ac2150754ac75a7745389d6d237a71d6520ea
[backup-repo]: https://github.com/DLwbm123/Knowledge_editing
[readme]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/README.md
[selection]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_mdu150_t2g_supplement_v1_20261106/METHOD_SELECTION_ZH.md
[final]: https://github.com/DLwbm123/Knowledge_editing-public/blob/7c1ac2150754ac75a7745389d6d237a71d6520ea/reports/med_mdu150_t2g_supplement_v1_20261106/REPORT_ZH.md
[main-json]: https://github.com/DLwbm123/Knowledge_editing-public/blob/7c1ac2150754ac75a7745389d6d237a71d6520ea/reports/med_mdu150_t2g_supplement_v1_20261106/RESULTS.json
[audit]: https://github.com/DLwbm123/Knowledge_editing-public/blob/7c1ac2150754ac75a7745389d6d237a71d6520ea/reports/med_mdu150_t2g_supplement_v1_20261106/EXECUTION_AUDIT.json
[mdu-prereg]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_mdu150_v1_20261105/PREREG_ZH.md
[bank]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_mdu150_v1_20261105/REPORT_ZH.md
[comparators]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_results_summary_20261029/REPORT_ZH.md
[grace]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_grace_correctness_v1_20261107/REPORT_ZH.md
[original]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med146_single_sequential_v133_20261027/COMPARISON_RECOVERY1_ZH.md
[medra]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/medra_v134_20261027/COMPARISON_ZH.md
[mce]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_medinit_101_supplement_v1_20261104/REPORT_ZH.md
[attribution]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_mdu_analysis_v1_20261105/REPORT_ZH.md
[mce150]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_mce150_matched_v1_20261106/PREREG_ZH.md
[t4]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_t45_comparators_v1_20261108/RESULTS.json
[t5-protocol]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_t5_release_alignment_v2_20261108/PROTOCOL_DIFF_ZH.md
[t5]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_t5_release_alignment_v2_20261108/RESULTS.json
[longitudinal]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/padchest_longitudinal_v2_20261109/REPORT_ZH.md
[temporal]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/padchest_temporal_mechanism_v1_20261109/REPORT_ZH.md
[conditional]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/padchest_conditional_mdu_v2_20261113/REPORT_ZH.md
[conditional-json]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/padchest_conditional_mdu_v2_20261113/RESULTS.json
[med-prereg]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med146_single_sequential_v133_20261027/PREREG_ZH.md
[entry]: https://github.com/DLwbm123/Knowledge_editing-public/tree/997168ae/experiments/med_mdu150_v1
[tt]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/medtrace_tt_only_budget_h_v1_20261005/structures.py
[med-code]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/med150_v1/train.py
[med-curvature]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/medical_proposition_v133/med133_curvature.py
[med-math]: https://github.com/DLwbm123/Knowledge_editing-public/tree/997168ae/experiments/medtrace_m13_146_v1
[mdu-code]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/med_mdu150_v1/matched_train.py
[loss-code]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/med_proposition_generalization_v1/robust_objective.py
[anchors]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/m3bench_editor_paperspec/m3bench_repro/editors/methods.py
[router]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/m3bench_editor_paperspec/m3bench_repro/editors/routing.py
[benchmark]: https://github.com/BioMed-AI-Lab-U-Michgan/M3Bench/blob/03c6fda3813301dab3be5831fdc94b493c10afc9/TASK_DEFINITIONS.md

[bank-binding]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/med_mdu150_v1/prepare_bank.py
[pooling]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/m3bench_editor_paperspec/m3bench_repro/editors/llava_runtime.py
[routed-linear]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/m3bench_editor_paperspec/m3bench_repro/editors/routed_layers.py
[balanced-audit]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_balancedit_v1_20261028/EXECUTION_AUDIT.json
[alpha]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_alphaedit_v1_20261028/REPORT_ZH.md
[base]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_base_reaggregation_v1_20261028/REPORT_ZH.md
[decomposition]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med146_t2g_decomposition_v1_20261029/REPORT_ZH.md
[error-review]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med146_t2g_decomposition_v1_20261029/ERROR_REVIEW_ZH.md
[early-subtypes]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/medtrace_native146_interim20_v1_20261015/T2G_INPUT_SEMANTIC_AUDIT_ZH.md
[exact-switch]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_exact_switch_v1_20261028/REPORT_ZH.md
[modal-margin]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_modal_margin_v1_20261028/REPORT_ZH.md
[depth]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_tt_depth_summary_20261101/REPORT_ZH.md
[handoff]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med_mdu_pro_handoff_20261105/REPORT_ZH.md
[historical-rank4]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/medtrace_full_method_comparison_20261003/EXECUTION_PLAN_ZH.md

[a1-extension]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/reports/med150_v1_20261027/REPORT_150_ZH.md

[a2-terms]: https://github.com/DLwbm123/Knowledge_editing-public/blob/997168ae/experiments/med_proposition_generalization_v1/span_train.py

[a1-lineage]: https://github.com/DLwbm123/Knowledge_editing-public/blob/2c5bfc1537c553c2623a49eea023a49749f23eaf/scripts/m3bench_core9_public_query_inventory.py
