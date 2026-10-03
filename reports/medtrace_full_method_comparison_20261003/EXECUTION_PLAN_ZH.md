# 完整 MedTRACE 与 BalancEdit、BELoRA：原 146-edit 配置比较计划

日期：2026-10-03。状态：`PLAN_ONLY_BLOCKED_H_SUPPORT`。

本文件响应用户“这个差别可以接受，H 监督是我们的方法特性，设计一下实验计划”。研究问题是：在原 Stage17 相同编辑请求、评测输入和生成设置下，完整 MedTRACE 的整体效果与两个已锁定基线相比如何？H 监督保留为 MedTRACE 的方法特性，单独披露额外监督量与成本。主表中的 `C_NO_H` 改为真正执行 H 监督的 `C_FACT`；历史 NO_H 结果保留，作为消融背景。

**这是待执行计划，不是启动授权或运行回执。用户此前暂停指令继续有效：不启动 GPU、Judge 或小时监测，不修改旧轮配置、结果和账本。** 方法与评价方案在这里确定；私有数据清单、代码绑定和资产可用性尚需准入后冻结，不能把当前文件称为已具备完整数据的可执行冻结包。

## 1. 实验矩阵与方法身份

| 主表方法 | 固定训练配方 | 部署与连续编辑 |
|---|---|---|
| MedTRACE C_FACT（完整 H 监督） | 自有 native 初始化 CP → A2 80 → CP-W0 320 → free-rank-4；继续训练 320 updates；保留 native/fit、H CE 和 U 上 Base KL | 原 R0：全局最近专家，再检查该专家原半径；无第二最近回退。独立专家按原顺序入库并真实生成全库结果 |
| BalancEdit | 原锁定 LLaVA-Med adaptation；layer31 up_proj；Adam lr=0.01，50 steps，alpha=0.2 | 原 pooled-key Euclidean 路由、半径和 insertion/replay 语义 |
| BELoRA V2 | 原锁定 effect-repaired independent implementation；layer31 gate/up/down，rank=16，alpha=16；AdamW lr=5e-5，50 epochs/edit，dropout=0 | 原实际连续状态更新与路由语义；不得把独立 single checkpoint 拼成 sequential |

完整 MedTRACE 在本计划中特指 Stage18 `FULL_METHOD_LOCK` 的 C_FACT writer 加 Stage17 原 R0 部署。不能把 H 缺失的 NO_H 改名为完整方法，也不把近期 TXT、PLOO、RC 或新学到的 scope router 混入这一主比较。若以后要评价这些组合，应另列方法和独立计划。

MedTRACE 固定 layer21 down_proj、rank4、每专家 73,728 个 FP32 参数。损失为：`0.5 native CE + 0.5 native-only fit CE + 1.0 H CE + 0.01 KL(Base||student)`。沿用 A_lr=1e-4、B_lr=1e-3、Adam betas=(0.9,0.999)、eps=1e-8、weight decay=0、clip=1，以及原 token/source-group averaging、320-slot 调度、Base-only full-vocabulary teacher。保留原 128-token teacher 上限的“超限报错、不截断”规则；不静默换成后续其他阶段的 teacher 协议。

BalancEdit 的训练 positive 使用已批准 native-only fit，不使用 official evaluation rephrase。两个基线维持标准锁定配方，不强行加入 H，不按结果重新搜学习率或训练步数。BELoRA 的 50 epochs 与原论文 5 epochs 的差异必须写入表注；两者是已披露的适配/重实现，不称作者原实现或 paper-exact。

## 2. 共同数据与生成条件

主队列为原 Stage17 **146 个 native 编辑请求**，保持全部成员、原顺序、原 probe 集和 prefix 角色映射。原 cohort freeze ID：`fea41e32c8b7780f45b060192343f3bb11c2e43b2e046fd709db19df6c0eef92`。不把全部 594 个 task-specific natives 合并成一个队列，不用既有 16-edit pilot 代替 146。

共同条件：LLaVA-Med Mistral 7B；official-native runtime_b/current_stack_v4；原模型、视觉模型和 tokenizer 绑定；FP16 backbone/generation；batch1、greedy、beam1、max_new_tokens=1024、use_cache=true、原 EOS；推理输入只有图像和问题，无答案、无 gold-length 截断、无 teacher-forced 评价。方法内部原有 FP32 edited state 保留并披露。

任务仅为原 146 队列附带的合法面板，主看 T0、T1G、T2G、T1L、T2L。其他任务仅按原合法支持单独报告，零支持为 NA。不扩张 T5、sealed/held-out 权限，不声称全十任务 Overall。所有 prefix 上 active-target 冲突和 locality 角色转换依原预先登记的 metadata map 执行，不按回答好坏删除样本。

## 3. 先解决 H 数据准入

当前冻结清单是 `E_U=146, E_H=0, E_HG=0, H_eval_supported=0`。这是完整 146-edit C_FACT 尚不能启动的实际原因。现有 UNKNOWN scope 提案不等于可用的 H 监督。

每个原编辑至少需要 **1 个合法、已验证的 H_fit 关系**，并按原 source-balanced schedule 在 320 个 H slots 中循环使用；报告独立 QA、图像、来源组和重复次数，不能把 320 次使用写成 320 个独立标注。若有多个合法 H，先按来源均衡和稳定 ID 确定全部选择/顺序，再冻结，不按模型表现挑选。

H_fit 的证据必须同时支持：同一待编辑命题、不同图像、原始来源答案与编辑 target 冲突，以及该编辑不应迁移到此输入的理由。保存原 QA/图像出处、命题对应、答案极性、关系证据、验证者身份/方式和版本。只有原问句、不同 image hash、答案不同或相似度高，都不足以自动确立 H。医学判断不能由程序伪签为临床审核。

H_fit 和 U_fit 必须排除**整个** formal native/probe/evaluation/protected 集，包括还未插入的未来编辑。按完整输入、图像、QA、源组与可获得的患者/病例证据检查；没有患者身份资料就记 UNKNOWN，不能据不同文件名声称患者独立。原 native-only fit 合法性重新绑定，禁止用 official T2G 评测改写当训练支持。

数据工作优先审阅此前已合法取得的 M3Bench/source 候选和真实验证材料。不足时登记具体缺口与可合法获取的原始来源，不能无限重扫已耗尽来源、自动造标签或解封保护数据。H_fit 选择人员不得查看本轮模型输出。

准入分支在模型输出前固定：

- **146/146 有 H_fit 且 U/角色审计通过**：执行本计划完整 146 比较。
- **部分有支持**：保留缺失，完整比较标为 BLOCKED。可以另写“共同 H-supported subset”计划，让所有方法按该子队列重新执行；不能切片历史 146 sequential，也不能标为完整 146 结果。本计划不自动授权该分支。
- **无真实 H**：停在数据准入，不占 GPU。

H_fit 是负向保护监督，与新 scope router 所需的 scope-positive CAL 是两件事。原 C_FACT/R0 比较不额外套用“每 pilot 4 同图+2 跨图 positive”的路由 CAL 门槛。独立 H_eval 如果能够取得，则单独冻结、与 H_fit 源隔离，所有方法用同一面板；否则 PairCorrect/H preservation 为 NA，主比较仍用原 formal locality。不能在 H_fit 上算 H_eval 或把新增 H_eval 称为原 146 配置的一部分。

## 4. 执行顺序与状态语义

第一轮用原训练 seed namespace `20260912`。MedTRACE 沿用 canonical_id 派生初始化/continuation seed；基线沿用原 seed 调度。确定性生成固定不变。

1. **P0：CPU 数据与资产准入。** 核对 H/U 支持、全局角色、原队列、代码/runtime、原始输出及必要初始化资产。列明可复用项、必须重算项、完整任务数和最后 checkpoint 消费者。冻结私有 manifest 与本轮绑定；未通过不进 GPU。
2. **P1：机械 smoke。** 按稳定队列位置取前 2 个已经通过支持审计的编辑，检查实际 H/U loss、FP32 梯度/更新、Base teacher OFF、RNG/hook/专家状态隔离、R0 选择、save/load 生成一致性及三方法真实完整 argv/GPU UUID。只依据机械正确性准入；不依据准确率筛选样本或取消方法。测试成本计入本轮；完全绑定一致的有效任务可纳入后续结果，失败证据保留。
3. **P2：single，全部 146。** 每次从规定 Base/初始化执行单次编辑及全部对应 probe。C_FACT 必须实际做 H continuation 并生成；不得借用 NO_H 的回答。按方法衔接其连续编辑依赖，避免先堆积所有方法权重。
4. **P3：sequential，同一原顺序。** 原生处理全部 146 编辑；记录每次插入时 native 成功率。在 prefix **1、50、100、146** 上评测此前所有编辑及合法 probes。C_FACT/BalancEdit 只在严格绑定通过时复用各自独立专家，再真实生成全库答案；BELoRA 保留原连续训练状态。不能以单次成功率或最终单次 checkpoint 代替连续编辑。
5. **P4：统一评分、完整汇总与交付。** 三方法全部完成或清楚记录 execution failure 后，一并报告结果、missing、成本、配对差、限制；没有按分数取消某个基线的分支。

主计划只有三个新比较方法。历史 C_NO_H 留在补充表；其初始化/支持若与新 C_FACT 不完全一致，不据它声称“只改变 H”的因果增益。严格配对 NO_H continuation 消融需要另列同 W0、同 RNG/slot 等绑定与预算，再执行；本轮不偷偷增添第四个训练方法。

## 5. 输出复用与 Judge

资产审计先于新模型输出：相同 cohort/order/runtime/model/precision/generation/路由、完整 ancestry 和 token/reference 绑定的**旧基线原始输出**可复用，避免无必要的重训。旧权重已删不代表原始输出无效；缺失绑定或协议变化时重跑受影响的完整阶段，不用 method 名或答案字符串判断复用。

本比较计划沿用原 Stage17 **gpt-6-astra / high、原盲评提示及严格 JSON 协议**，与此前 scope 研究的其他 Judge namespace 分开。服务没有 immutable snapshot 的事实必须披露，不声称模型跨日期完全相同。

为避免“新 C_FACT 与旧基线评分时代不同”成为主要混杂，主结果建议建立一次统一 comparison scoring epoch：Base 与三个方法的合法 raw outputs 都进入同一预先冻结的盲评队列；优先在学生生成之前处理/冻结 Base 审计。方法/训练/H 信息不交给 Judge。每个新 namespace 的完整 input/reference/raw-output/protocol key 只尝试一次；不得为翻转 verdict 重评。旧阶段永久 missing 通过 shared identity 继承，不借新 namespace 重判。执行前需冻结此一次评分复核的 manifest 与授权绑定，而不是把它称为旧请求的自动重试。

主历史复现面板固定原 146 成员、原 Base eligibility masks 和原 prefix role map，报告为“原冻结 Base-mask 条件下的效果”。新的统一 Base 评分只作一致性审计及另外标明的同 Judge mask 敏感性表，不能据新 verdict 改选 146 natives 或改变主分母。必须同时披露新旧 Base 判定分歧与其对结论的影响。

旧 Astra 评分表保留为历史参照；不得与新的统一 scoring epoch 拼成一个主表。若后来要改用 Sol，须在输出前改成一个完整、统一的新 scoring contract，连同 Base mask 处理一起登记；不可只给 C_FACT 改 Judge。连续 3 次新 transport failure 停新请求，保留已接受 verdict/partial/missing，不盲重试。Judge budget 计“首次尝试的 payload key”，另报 HTTP/API batch 次数、token 与费用，不能把一个 batch 当一个问题。

## 6. 指标、分母与主要对比

| 指标 | 用途与报告方式 |
|---|---|
| T0 PostAcc ↑ | native 编辑成功；single、插入时与各 prefix/final 分开 |
| T1G/T2G Fix ↑ | 原 Base-wrong probes 的修复率；同时报告全部合法 probes 的 PostAcc |
| T1L/T2L Retention ↑、c2w ↓ | 原 Base-correct probes 的保持；稀少支持单独标注 |
| Insertion→final retention ↑ | 插入时成功是否在最后 bank 保留，失效轨迹和 counts |
| PairCorrect / H preservation ↑ | 仅有真正独立 H_eval 时，按共同支持子集报告；native 失败不丢出分母 |
| 成本 ↓ | 总训练与生成 GPUh、失败成本、峰值 VRAM、存储峰值、支持量、Judge keys/tokens/费用及复用量 |

主比较为 C_FACT−BalancEdit、C_FACT−BELoRA，在 final prefix146 上报告；single 和中间 prefix 为预先登记的辅助结果。沿用 **edit-macro 为主，probe-micro 与分子/分母为辅**。不只给百分比，不以 T1L 的极小支持声称广泛 locality，不构造任意加权 Overall。

配对 edit bootstrap 10,000 次，seed=20260912，报告 95% descriptive intervals 及 source-group cluster sensitivity；患者独立性未知时不作患者级独立推断。多个任务/prefix 的区间是探索性描述，不据单个显著数字宣称全域胜出。

评分缺失保持完整分母：报告 known counts、coverage 和 shared-key missing 的配对最坏/最好差界，不仅算共同有评分的成功子集。执行失败和语义错误分开，附 intention-to-edit 保守界。无支持为 NA，不填零；部分完成不称完整比较。

## 7. 有限后续验证条件

先完成三方法全部主表，不在运行中改门槛。以下是**新计划的后续决策规则**，不是重定义历史 PASS：

- 若相对两个基线，final T0/T1G/T2G 的 edit-macro 配对差下界均不低于 **−1 pp**，且 final T2L Retention 的差下界均至少 **+5 pp**（此处下界指 shared-missing 最坏情形点估计，不等同统计置信下限），同时没有未解决运行/绑定问题，则进入“值得验证”状态。CI 与 small support 仍决定允许的结论强度，不能把此规则当成已证明统计优越。
- “值得验证”时，建议另开真实训练多 seed 验证：基础 seed20260912 加固定 seed20261003、20261004，共三种子，三方法全部同组 seed、同队列、同支持和同门槛；完整报告各 seed 与 mean/std。先冻结 seed 对 RNG 的实际作用，不能只重复 deterministic generation。两个额外 seed 合计拟议上限 64 GPUh、64,000 新评分 key；执行前独立准入和授权，不由本计划自动启动。
- 若只胜一个基线、牺牲 generality、发生 H preservation 损伤，或 missing 使结论无法判断，保留完整负结果并审阅错误类型/路由/支持关系；只在有明确机制假设时另开一个最小变更阶段。没有预登记的大网格、无限 seed 或半径搜索。

该规则较严格，未达到只说明未达到预定后续验证条件，不等于所有应用无效；仍交付实际完整数值。原面板已经暴露，追加 seed 只能检验训练随机性，不能冒充新数据独立确认或编辑顺序鲁棒性。

## 8. 预算、GPU 与停止条件

以下为**第一轮完整比较的拟议硬上限**，未启动、未消耗，也不代表用户已批准支出。它不同于旧 scope 小阶段的 4GPUh/500-key 默认额度；启动前应把明确采用的有限额度写入独立 RUN_MANIFEST，不能继承旧额度后暗中超限。

| 资源 | 拟议第一轮上限 |
|---|---|
| GPU | **32 累计 GPUh**，含机械测试、初始化、训练、生成、失败及修复成本 |
| 首次执行墙钟 | **72 h**，从本轮 P0 执行首次启动计时；暂停/修复不重置；不得从本文件写作时间计时 |
| 新 Judge payload-key 尝试 | **32,000**；包含共同 Base 审计、统一三方法主评分及必要失败；重复 input/output 正确去重 |
| 本轮生成 checkpoint/optimizer 临时峰值 | **64 GiB**，另外必须预留完整输出/teacher cache 估算量和至少 8GiB 可用余量 |

历史 BELoRA 单/连续完整 raw row 出现量分别为2524/3359；三方法加历史全 Base2465 的未去重参考规模约20,114个 raw row。实际多参考、角色转换、去重、missing 和审计决定新 key 数，必须在 P0 算清任务清单再准入，不能把此估计当最终评分分母。32,000 是停止上限而非必须使用量。历史硬件时间不能当 pro5000 的 ETA；若资产/吞吐估计显示上限不足，在输出前调整完整计划，运行中到限则保留 partial，不能续开新 ledger 绕过。

未来执行只用 **pro5000 物理 GPU6/7**，不得用 GPU5 或沿用历史 GPU2/3。每卡本轮最多一个 resident model；启动前检查峰值加余量即可共享，不干预其他进程。可按方法依赖分配两卡，但不把 BELoRA 原生 sequential 任意拆断以并行。主/子入口和 argv 中性，RUN_ROOT 通过环境传递；启动后核实完整 argv、PID/start identity、RUN_ROOT、GPU UUID。

大文件只在 `/data/bmw/Knowledge_editing/outputs/medtrace-full-method-comparison-20261003/` 及既有 `/data/bmw` 缓存。创建前核实实际挂载、空间和一次读写探针。保留 Base/原始数据/历史权重，原子保存与活跃恢复状态算入峰值。

checkpoint 按既有 CHECKPOINT_LIFECYCLE：single → 所有登记 sequential/prefix 消费者 → save/load → 完整 raw/token/Judge bindings 落盘后，才删除确认无消费者的**本轮生成文件**；不得删除尚需入库的早期专家。写具体路径清单和删除回执；不默认迁移无限权重归档，不等待评分好坏才决定清理。本次规划不删除任何文件。

硬停止：用户停止；H/U或全局角色不合格；科学配置/生成/runtime/身份绑定漂移；nonfinite/OOM；存储不满足余量；预算达到上限；连续3次新 Judge transport failure。运行故障修复前保存证据、成本与可恢复状态，最小修复和必要机械检查后才恢复；科学结果差不是运行故障。已完成有效产物不重复计算，时钟与失败成本不清零。

## 9. 交付与启动前剩余项

交付包包含主/辅助指标 CSV、macro/micro/counts/missing bounds、配对比较与 CI、prefix 曲线、insertion→final 表、H/U支持与角色覆盖聚合、资源/失败/复用/清理账本，以及完整中文报告。源代码/config/必要检查/匿名汇总通过本机127.0.0.1:7897代理发布到对应公开仓库的新 review 分支及 draft PR，核实 remote SHA/匿名访问并 attach。不得发布医疗 QA/图像、患者信息、逐题 verdict/key、原响应/tokens、权重、密钥；旧 PR 与旧结果只读。公开交付受阻时明确未交付内容，不重跑科学计算。

真正启动还需：用户明确恢复；146/146 H_fit及U角色准入；私有 cohort/support/teacher/eval/scoring manifests；实际代码/runtime与资产复用审计；明确采用的预算与初始账本；GPU6/7/存储检查。现有小时监测保持 PAUSED。此计划已可审阅，但当前不声称数据到位、实验已冻结启动或比较结果已获得。

## 依据文件

- Stage17 `METHOD_LOCK.json`、`EVALUATION_CONTRACT.json`、`formal/GPUHOME_COHORT_SUMMARY.json`、`campaign_closeout/CAMPAIGN_RESULTS.json`。
- Stage18 `FULL_METHOD_LOCK.json`；此配置可在 scope-m3bench-candidate-audit checkout 的对应 reports 中读取。
- Stage17 `formal/BELORA_SINGLE_COMPLETION_20260915.md`、`BELORA_SEQUENTIAL_COMPLETION_20260915.md`、`CHECKPOINT_LIFECYCLE.md`。
- `.local_run/FULL_METHOD_MATCHED_COMPARISON_PREFLIGHT_20261003.json`、`ACTIVE_STAGE.json`、`USER_PAUSE.json` 位于当前 scope-m3bench-candidate-audit checkout，提供 H=0 与暂停状态依据。
