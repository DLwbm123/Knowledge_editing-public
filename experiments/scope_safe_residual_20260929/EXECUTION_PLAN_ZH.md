# MedTRACE 下一阶段：作用域选择与非目标残差保护

## 可直接交给本地 Codex 的执行任务

目标不是整理汇报，而是改进方法：在不牺牲原始编辑修复和跨输入泛化的前提下，提高真实专家库的非目标保持。立即执行可执行部分，不等待 E3/E4 的科学结论；但不得抢占其资源、改变其协议或把运行中结果用于选配置。本文是新增阶段的执行方案，不是训练已启动的声明。

### 0. 起点、隔离、资源

1. 先读取本地实际仓库、运行目录、模块绑定、GPU UUID/进程/租约、E3/E4 handoff 和资源账本。参考公开审阅提交 c411d636f96b39d745fd0c9e89db7b8ba492ca85，但不回滚或覆盖本地更新。公开结果参考 78e273e5176da6b9544a86c547e0f658fbec804b。
2. 新建独立 worktree/branch 和 RUN_ROOT。建议名称 scope_safe_residual_20260929；不得改运行中进程会动态导入的源码、source_patch、symlink、配置、队列或 Judge 缓存。科学协议、初始化、样本和输出均采用独立命名空间。
3. 物理 GPU 5/6/7 已用于旧实验；显存空闲不代表 E4 已放弃预约。只有 E3/E4 控制器明确释放且新任务取得租约后，才可使用。其他 GPU 必须已有明确使用授权和独占租约，不得自行假设“第四张卡”可用。默认一张卡一个 resident backbone worker，顺序执行专家分支。
4. 现在立即执行 CPU 诊断、数据清单/碰撞审计、利用已缓存特征做校准、实现和 CPU 单测。缺特征的生成任务排入新队列；若没有可用 GPU，生成 READY_WAITING_FOR_LEASE 状态并建立不争抢旧任务的接续机制，不能伪称 GPU 已开跑。
5. Judge 沿用现有固定模型、reasoning 配置和评分协议。读取真实累计用量与所有旧任务预留，不能挪用 E3/E4 额度；公开 PR 中 9539 是旧阶段内部尝试项上限，不是新阶段预算。新增额度未明确授权时不得自动提高。不要按未知去重率预支。
6. 每个完整配对块在训练/大规模生成准入前计算保守评分需求，扣除已经确认可复用的成功 payload 后再核实覆盖；额度不足则不准入该 GPU 块，但继续独立的 CPU 分析和单测。不得挑其中成绩较好的分支单独完成。报告资源阻塞，不循环空转。
7. 不给成功键重判，不把失败键改名重试，不重置旧计数。权重改变后如果新输出的完整 Judge payload 确实不同，可以成为新键；不能为了制造新键而改变 prompt 或答案。

### 1. 核对继承实现，避免重复发明

至少核对并记录实际加载路径和 hash：
- experiments/decomp_24h_20260927/structures.py
- experiments/decomp_appendix_20260929/training.py
- experiments/decomp_24h_20260927/evaluation.py
- experiments/decomp_appendix_20260929/report.py
- experiments/textjoint_r3_20260926/router_r3.py
- experiments/textjoint_r3_20260926/calibration_data.py
- experiments/textjoint_r4_20260926/README.md

已有 R1 是半径缩放 kappa + 最近邻间隔 mu；它是对照，不是新方法。旧 R4 因跨图像 CAL_PLUS 正例不足而冻结 R0；不要误写成“R1 已经被证明无效”。现有 U-KL 已采用 teacher-forced/forced-on 保护路径；不要把 forced-on 当成新增算法贡献。新变化仅是负例覆盖、困难采样、有效残差约束和正负原型作用域门控。

### 2. P0：立即完成的故障分解

只读取已完成、不可变的父实验和 E1/E2 快照，不读取 E3/E4 中途成绩。匹配键至少包括 seed、panel、method、edit、task、query/input hash、模型/生成绑定。

将 single -> bank12/24 转移拆为：
- OFF->OFF；OFF->ON；ON->OFF；ON->ON 且同专家；ON->ON 且不同专家。
- 在每个互斥桶中，分别统计 single 正确->bank错误、错误->正确、均正确、均错误、缺评分。
- 按 scope-positive / scope-negative、正式任务类型、Base 是否正确分别统计；不要把 task 的正负角色、Base 对错、route 激活混为一类。
- 同时输出 consumer 数、unique input 数、edit 数、source/patient cluster 数及 eligible 分母。
- 原 negative_activation_damage 是 bank 激活且 Base 正确但 bank 错误的计数，并不必然是从 single 新增的错误。更名输出或附明确字段解释，保留历史原件。

检查同一个输入在两种运行中若专家、模型、mask、生成设置都不变，输出是否一致；不一致应先定位数值/缓存/绑定问题。oracle-reject 仅诊断：保持正例原候选，对真正非目标输入强制 Base；不得把 oracle 当可部署性能或总体理论上界。

产物：FAILURE_DECOMPOSITION.json、IDENTICAL_ROUTE_PARITY.json、P0_FINDINGS.md。

### 3. 数据与作用域

创建 SCOPE_ROLE_MANIFEST.json、DATA_ACCESS_POLICY.json 和 DATA_AUDIT.json。fit、calibration、check、regression、holdout 的用途必须是程序可检查的。

- 旧正式 T0/T1G/T2G/T1L/T2L、旧压力 validation/test、其逐题成绩不得进入训练、困难负例检索池或门控阈值拟合。已暴露 REG/DEV 可用于本阶段探索评测，但不称独立测试。
- 编辑正例沿用合法 native/S_fit（以及该分支原本允许的 G_fit）；不能根据正式问题改写新训练问句。
- U_bg：从允许训练的辅助池中确定性建立，记录 source、image hash、question hash、非目标作用域依据。优先包含问题相近但事实不应被本编辑改变、同图不同且不相关事实、跨图像相近问题等类别。不能仅因图像不同、答案不同或 embedding 近就标为负例。
- 对涉及编辑事实传播、同义知识、同一病灶/实体等不确定样本，标记 scope_unknown 并排除负例训练。不能把其他编辑的合法正例笼统压回 Base。旧编辑已改变的知识也不能作为“应恢复 Base”的负例。
- U_bg 的主实现不依赖尚未到来的编辑支持，避免 sequential 的未来数据泄漏。每个新专家只用自身合法支持、预先冻结的背景池和当时已合法可用的信息。
- Gate 的正原型仅来自该编辑的合法 fit 支持。负原型仅来自 U_bg；不能使用正式评测 query_id 或 associated edit id 作为推理特征。
- CAL_SCOPE 正例必须包括同图改写和经作用域核验的跨图像正例。目标每个开发编辑至少 2 个跨图像校准正例、至少 4 个同图校准改写；这是准入目标，不得虚构填满。若不足，Rneg 标记 UNSUPPORTED_CAL_SCOPE，专家训练主线仍可继续，不用缩小到有利子集来宣称路由成功。
- 阈值仅使用独立于 fit 的 CAL_SCOPE。全局阈值做 edit/source-group 分组交叉拟合检查，禁止同一来源落入两个折。每个被编辑案例的合法支持可构建自己的原型，但不能用其正式测试问题选阈值。
- 新增 LOCALITY_STRESS_HOLDOUT 从未用于训练/校准的来源构建；目标至少 200 个 Base 原本正确的非目标输入、至少 50 个独立来源。先冻结候选清单，再做统一 Base mask；同时报告完整候选数和 Base accuracy，不只报告条件保持。不足时报告真实规模，不复制 QA 凑独立样本数。
- 若没有新来源，不将 DEV/REG 重命名为 CONFIRM。新独立编辑确认集需按患者/来源/编辑事实族与既往开发隔离；同一事实换句话不是新病例。

### 4. 共同初始化与训练分支

主线固定 CP -> FREE rank4、当前 layer 和归一化残差实现；不同时改 rank、层、路由特征编码器、G/D 配方或 precision。

优先导入 E1/E2 对应已完成 M1 的 post-continuation-80 最终 adapter。完成权重/数据/代码/模型/输入归一化绑定审计，将其标记 INIT_POST80，而不是 W0。对每个编辑克隆同一个 INIT_POST80，所有新分支使用相同新优化器起点、相同 seed、相同正例采样和额外 80 步。保留原 M1 不追加训练的 E_orig 作为历史起点。

没有可兼容共同初态时，按同一协议重建一份后分叉；不得从不同历史实验各拿一个“差不多”的权重作配对。

五个分支：
A0 EXTRA80_CONTROL：原 M1 loss + 原 U 采样，额外80步。
AK STRONG_KL：与 A0 完全相同，仅将原 U-KL 总权重乘4。
AU EXPANDED_UNIFORM：保留原 U_old 保护项；原 U_new 槽位按 50% 原 U_new + 50% U_bg 均匀采样。KL 权重、槽位数及前向次数与 A0 相同。
AH HARD_NEGATIVE：与 AU 使用同一个 U_bg 和相同的50/50混合比例，只把 U_bg 槽位改为 H_i 的均匀采样。H_i 是利用冻结 Base 路由特征、该专家原始 key/radius，按最近/最易激活排序的前16个经作用域确认的背景输入；按来源设上限并确定性打破并列。选择不得依赖正式评分、不得使用新增训练分支输出挖样本。H_i 不足16时如实报告。
AHS HARD_PLUS_RESIDUAL：与 AH 相同，再加入下节的有效残差惩罚。

AU/AH 必须共享背景全集、预算与类别配额。其差异是采样分布，不得让 AH 额外获得 AU 不可见的数据。若 U_bg 不足，AU/AH 标记受限；不能静默更换实验定义。

本阶段不新增 G/D。后续迁移到 M3 时，所有配对分支完整继承相同 S+G+D、教师、质量mask和采样，不能混在主线比较里。

### 5. 有效残差约束

以列向量记：
  hbar = h / (RMS(h) + epsilon)
  r_i(h) = B_i A_i hbar

在 AH 正在使用的负例保护 batch 上，沿与现有 U-KL 完全一致的 teacher-forced predictor positions 和 hook 应用位置，计算：
  L_res,i = mean_{x,t} ||r_i(h_{x,t})||_2^2 / c_i

固定尺度：
  c_i = max(mean_Q ||r_i,start(h)||_2^2,
            1e-6 * mean_Q ||o_Base(h)||_2^2,
            1e-12)

Q 是训练前冻结的小型负例参考集合，按 AH 的原U_new/困难负例混合分布构造；c_i 从共同 INIT_POST80/关闭专家的 Base 参考中一次性计算、detach、保存，不能在训练中随残差缩小重算。o_Base 是相同层 Base 模块输出，不得混用最终词表 logits 的维度。

AHS loss = AH loss + lambda_res * L_res,i。
只在 pilot 比较 lambda_res=0.01 与0.1，使用训练隔离的 CHECK/CAL 选择一次；并列选0.01。全量开始后固定，不能逐编辑选lambda。正例损失不变。所有负例保护均强制该专家ON，否则路由关闭会让惩罚虚假为零。

必须测试：
- 约束针对完整有效残差 BA，不是只约束本轮更新 BA-BA_start。
- 不用 ||A||^2+||B||^2 代替；在可逆 gauge 变换 A'=Q A、B'=B Q^{-1} 下惩罚数值应一致（测试容差明确）。
- 负例残差非零时，A/B 有有限、正确梯度；hook、mask、teacher logits 不错位。
- Base 参数不产生梯度、不变化；o_Base 和 c_i 不参与梯度。
- zero/near-zero scale、空mask、无合格负例必须明确定义：无数据不能标记训练PASS，空mask报错/跳块。
- 不物化4096x14336稠密残差或14336x14336稠密协方差；使用(hbar @ A.T) @ B.T。
- 训练中复用已做的KL前向与激活，不额外展开全库前向。单独计数所有诊断、参考、CE、KL调用。

### 6. 路由分支：先不改变检索专家

R0：现有冻结路由。
R1：现有 kappa/mu 拒绝门控，在本阶段合法 CAL_SCOPE 上按原规则校准，作为简单对照，不宣称新贡献。
Rneg：仍由 R0 选择原候选 j，不重排候选，不增加新大模型，仅对它做作用域拒绝。

用与当前路由完全相同的冻结 Base 特征 z 和距离约定，定义：
  d_plus(x,j) = min_{p in P_j} distance(z(x),p)
  d_minus(x,j) = min_{n in N_j} distance(z(x),n)
  q(x,j) = (d_minus-d_plus)/(d_minus+d_plus+epsilon)
  active = R0.active AND q(x,j) >= tau

P_j 最多8个合法正原型；N_j 最多16个困难非目标原型。原型裁剪按固定规则，不能看测试表现。无候选/无负原型、零距离、重合正负原型、NaN、并列需有单测与明确定义；重合正负先视为作用域冲突，不能强行拟合。

tau 仅在[-0.2,0,0.1,0.2,0.3]上校准。校准约束：native correct-scope hit 不下降，同图改写和跨图像正例分别不低于R0超过1个百分点；满足约束时最小化各已声明bank前缀等权的非目标激活。无可行收益就保留R0，并记录NO_ROUTING_GAIN；不能靠全拒绝获取好看的保持率。

多编辑具有兼容作用域时，以事先冻结的合法专家集合计算correct-scope hit，而不是用唯一ID制造假错误。推理函数只能接收真实输入特征和可部署存储，不接收任务标签、参考答案或associated edit id。

Rneg 是绝对局部正负比较，不对整个bank做softmax。它不保证库变大后FPR不增长，必须实测。

### 7. 执行顺序与有限矩阵

P0 现在CPU执行；准备所有代码/数据/校准，不占E3/E4预约。

P1 PILOT_DEV8：按结果无关的source/数据类型分层与固定hash，预先选8个DEV编辑（若source grouping不足，按实际组数报告）。执行A0、AK、AU、AH、AHS(0.01)、AHS(0.1)，共48条额外80步continuation。不得因某编辑失败而换成另一个。失败按同seed同W的整对/整块处理。pilot正式QA只作开发诊断；lambda和route参数仅在CHECK/CAL上定。

P2 FULL_DEV24：固定lambda后跑A0/AK/AU/AH/AHS五个专家分支，均使用R0评测；共120条continuation，pilot中绑定完全一致的40条可复用。重点full single、bank24；bank12作中间诊断，禁止把不同前缀不同病例的均值差直接解释为库规模效应。完整记录bank4/8/12/24的路由曲线。

P3 COMBINATION：冻结最有希望的一个专家改进和A0，做2x2：A0+R0、A0+Rneg、改进专家+R0、改进专家+Rneg；另加改进专家+R1对照。若Rneg不满足CAL准入，明确跳过其性能主张，不能阻塞已支持的专家结果。

P4 FROZEN_REG24：DEV结束后冻结规则。REG只复核A0、最佳简单对照和最终候选，最多三种专家路径，不再在REG反复调参。若两者重复则合并，不造重复实验。它仍是已暴露回归面板，不是CONFIRM。

P5 仅在出现可复现增益后准入：
- 在另外两个种子上，复核A0与最终候选。INIT_POST80的来源必须匹配同一生成训练配方；否则重建后配对，不能混淆seed效应与版本效应。
- 在M3起点上做继承其S+G+D的普通续训vs相同新保护方案配对，检查改进可迁移性；不先把五分支全部乘上M3/Tucker。
- 作用域/病例真正独立的新编辑集，数量按可用数据审计，不伪称已有N24。
- 对实际存在的唯一编辑测试bank48/96/146。不能复制同24个专家凑更大bank；每个prefix报告当时所有编辑及固定早期编辑子集。
- 三种固定插入顺序只用于顺序敏感性，不计作独立病例；若方法没有训练/动态阈值的顺序依赖，核验最终bank置换不变性即可，不重复训练相同结果。

E3/E4收口后仅作为后续独立模块信息：若gauge/rank显示收益，在最终候选上追加一个小型匹配对照再整合，不追着中途结果改变本阶段主矩阵。

### 8. 指标与决策

必须同时报告：
- T0 repair、T1G/T2G、正式T1L/T2L、压力保持；edit-macro与micro分开；new stress的无条件accuracy与Base-correct条件保持分开。
- scope-positive correct-expert hit、scope-negative activation、被激活非目标输入的条件损伤率以及无条件损伤率，包含分母和missing。
- all-bank前缀指标与固定早期编辑子集的成长曲线。不同N的原始平均分不能直接相减归因。
- 路由相同但权重不同的配对结果、仅gate改变的配对结果；在预先固定的同一组非目标输入和同一批指定专家上做forced-on诊断，避免把拒绝后样本组成变化误当成专家本身损伤降低。新旧模块的2x2交互：Y_new,new-Y_new,old-Y_old,new+Y_old,old。
- correct-positive coverage匹配下的risk/coverage曲线；不能拿更低coverage的拒绝器直接宣布更好。
- 按编辑和source/patient cluster的配对bootstrap区间；多个seed先编辑内聚合，不把seed、QA、prefix当新增病例。
- 缺评分保留null，同时给micro及有每编辑分母时的macro最坏/最好边界；边界不是置信区间。不能从成功子集算完整均值。
- GPU实际驻留、CE/KL/额外前向次数、训练步数、adapter/原型大小、prefill/生成延迟。Rneg特征若不能从现有预处理复用，新增成本必须计入。

开发筛选目标（不是成功承诺，也不是统计非劣证明）：T0不新增未修复native，T1G和T2G各不降超过1个百分点，bank24压力保持相对A0争取提高至少5个百分点。必须同时比较不再训练的E_orig：若候选只修复A0因额外80步造成的退化、但没有超过E_orig，不宣称相对现有方法取得净改进。最终候选也应满足相对E_orig的编辑与泛化约束。同时审查single保持和各来源分层。点估计达标但区间不支持时标候选而非确认。更复杂方案如果不超过AK/AU/AH等简单对照，优先保留简单方案，不为“方法复杂度”强保新模块。

### 9. 缓存、产物与存储

先用P0保存的Base/专家输出做R1/Rneg的反事实筛查。输出复用必须核对完整输入、模型、tokenizer、精度、生成协议、实际执行专家与mask；新路由分支创建新的provenance，不能冒充旧execution_binding。OFF分支仅在Base绑定完全匹配时复用Base。输出可能等价不等于缓存文件的执行绑定相同，必要时写独立counterfactual记录并用GPU canary验证。不同专家/新权重没有可证明的相同执行路径时必须实际生成。

每活跃GPU只保留latest/previous滚动优化器状态；新阶段共同初态只保留有未消费分支的部分。最终保留必要的基线/简单对照/候选adapter、原型、角色清单、配置、hash、成功评分和脱敏汇总。pilot淘汰权重仅在评测和审计全部完成且无消费者后回收；不得删除父实验、E3/E4或非本阶段文件。

不保存每步checkpoint、全量词表logits、全数据激活或稠密协方差。教师张量采用受限内存工作集；原始医学数据和逐题细节留私有。若使用负原型/缓存激活，明确其属于来自历史/辅助数据的信息，不能称严格no-memory/no-replay。

最低交付：
SCIENCE_LOCK.json
INIT_BINDINGS.json
SCOPE_ROLE_MANIFEST.json
DATA_ACCESS_POLICY.json
DATA_AUDIT.json
FAILURE_DECOMPOSITION.json
TRAINING_ABLATIONS.json
ROUTER_CALIBRATION.json
CORE_MATRIX_RESULTS.json
GATE_EXPERT_FACTORIAL.json
MISSING_SCORE_BOUNDS.json
COST_AND_STORAGE.json
NEXT_DECISION.md

NEXT_DECISION.md只回答：哪些改动真正改善，哪些只是改变coverage或增加训练，哪些应删除，下一块为何值得运行。不要花时间生成导师汇报或宣传式总结。公开发布仅在已有发布授权范围内，使用独立PR；不合并PR5/PR6，不推送私有数据、权重、tokens或凭证。
