# MedTRACE Stage17：C_NO_H 实验报告（供 Pro 审阅）

日期：2026-09-15。证据截止：C_NO_H 于北京时间 21:07 完成统一 Astra 评分，随后报告公开发布。本文为已完成结果的整理与描述性比较，不新增训练、评分、超参数选择或样本筛选。

## 1. 审阅摘要

本次完成的是 MedTRACE 的 **C_NO_H 分支**：146 条合法编辑的 single，以及固定顺序、前缀 1/50/100/146 的独立专家插入与全库重新生成。完整 **C_FACT 未获本队列验证**，因为主队列没有合法 H 支持；不能把 C_NO_H 的结果写成完整方法或 H 机制已经有效。

主要观察：

- **原始请求写入与最终命中成功**：single 与最终 sequential 的 T0 均为 146/146；146 个请求在插入时正确、最终也正确。
- **连续编辑后的泛化下降**：T1G 的 edit-macro 从 100.00% 降至 86.47%，T2G 从 91.95% 降至 90.58%。
- **文本局部性仍有明显损伤**：single 与最终 sequential 的 T2L 都仅保留 25/54 个 Base 原正确探针；probe-micro Retention=46.30%，edit-macro=48.39%。29/54 个原正确观测变错。
- **没有显示出对已完成 BalancEdit single 的优势**：C_NO_H 的 T2G、T2L edit-macro 分别低 3.94、8.60 个百分点；这是点估计差异，不是显著性检验结果。
- **局部性和小子集结论不能夸大**：T1L 的 100% 只有 2 个 Base-correct 探针、1 个编辑；T4G 附属子集虽无 Fix 分母，但其 PostAcc 从 single 6/6 变成最终 sequential 0/6，不能省略。

当前证据支持“该分支能在本冻结队列中写入并保留原始请求”，尚不支持“完整 MedTRACE 优于基线”“U-KL 改善局部性已被因果证明”或“解决了多模态编辑干扰”。

## 2. 实验范围、方法与协议

### 2.1 队列与支持

|项目|本次实际范围|
|---|---|
|主队列|N=146，均为统一 Base Judge 原判错误的 T0 编辑|
|U 支持|E_U=146；缺 U 不允许静默去掉 KL|
|H/G 配对支持|主队列 E_H=E_HG=E_HG_eval=0，无法验证 C_FACT/H 对照|
|来源组|146 个 native 对应 125 个来源组；不等于 125 个独立患者|
|任务|主 T0 及其合法挂接的 T1G/T1L/T2G/T2L/T4G 探针|
|未覆盖范围|完整十任务 benchmark、额外 task-specific 队列和 T5；不构造 Overall|
|独立性声明|并非声明未见的独立确认集；无人工临床 signoff|

其他任务的合法候选 native 曾汇总为 594 条，但它们不是一个共同 N=594 的实验，不能混入本报告的 N=146。辅助来源隔离、native-only fit 例外和冻结 Base 掩码沿用原计划；不根据本次分数重新选样本。

### 2.2 C_NO_H 配方

- Base：冻结的 LLaVA-Med v1.5 Mistral-7B runtime；主干 FP16，writer 为 FP32 参数。
- Writer：`model.layers.21.mlp.down_proj`，free rank 4，73,728 个 FP32 参数。
- 初始化：本编辑 native CP → A2 80 步 → CP-W0 320 步 → 函数保持的 freeR4 转换；随后 continuation 320 步。
- 损失：`0.5 native CE + 0.5 native-only fit CE + 0.01 full-vocab KL(Base || student) on U`。
- Adam：输入/输出因子学习率分别为 1e-4/1e-3，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，clip=1；保留既有因子归一化。
- 生成：batch size 1、greedy、beam 1、max_new_tokens=1024，保持冻结 tokenizer、prompt、图像处理与 EOS 规则。

C_FACT 是在相同骨架上增加 H source CE 的完整分支；本报告没有相应 H 机制实验。C_NO_H 保留 U-KL，但没有“去 U-KL”的匹配消融，因此不能从本结果单独识别 U-KL 的因果收益。

### 2.3 部署与路由

Single 每次独立编辑冻结 Base。Sequential 复用这些独立专家，按冻结顺序插入专家库，并在每个规定前缀对已有请求和探针重新路由、重新生成；它不是把 single 分数拼成 sequential，也不是一个共享模型的在线累积重训。

路由在关闭专家的 Base 特征上选全前缀库最近专家，再检查该专家的 radius；不传 gold edit ID，不查未来专家，不在拒绝后寻找第二近专家。C_NO_H 与 BE 使用匹配的 Base 路由锚点，但 writer 不共享。R0 是预先指定的主路由；RC 以 κ=0.7696741135364367 缩紧 radius；FORCED_ON 仅为 single 诊断。

## 3. 评分和统计口径

新评分共 **5,732 个完整绑定的 student 判断，115 批**，已全部通过格式、ID 和覆盖检查。这不是 5,732 个独立编辑或独立患者，也不是表中所有探针观测次数之和。相同 Base OFF 输出复用本阶段已接受的 Base 评分；不同训练/前缀/参考绑定不因为答案字符串相同就随意合并。

Judge 固定 `gpt-6-astra/high`，服务端不可变 snapshot 不可得，记录为 null。每批新建隔离上下文；可见内容只有匿名 ID、问题、验证参考和候选答案，不看图像、方法名、既往评分、项目私有上下文，也不调用工具。它评估文本参考一致性，不能替代医学影像真值核验或人工临床审查。

评分先在第 26 批发生网络流中断，随后在第 90 批再次中断；分别保留 25 批、89 批接受前缀后恢复。失败批没有最终响应，沿用相同输入与配置；没有因为分数不理想重评，没有覆盖原失败证据。最终 115/115 批、5,732/5,732 条完整合并。

指标定义：

- **Fix**：只在 Base 原错误的合法探针上计算编辑后正确率。
- **Retention**：只在 Base 原正确的合法 locality 探针上计算保持率；`c2w=1−Retention`。
- **PostAcc**：在当前合法面板的全部探针上计算，作为补充，不替换 Fix/Retention。
- **edit-macro 为主口径**：先在编辑内部平均，再对有有效支持的编辑平均；probe-micro 给出总正确数/总有效探针数。二者不能混读。
- 零分母记 NA；同一 query 可能在不同编辑/部署上下文中形成多次观测，不能按全局独立样本解释。
- 95% 区间使用 10,000 次 bootstrap、seed=20260912；另报 native 来源组 cluster bootstrap 敏感性。它不是完整患者/数据连接图聚类，也不覆盖编辑顺序不确定性。

## 4. C_NO_H 主结果：R0

表 1：single 与最终 146 专家库。所有百分比越高越好；T0/T1G/T2G/T4G 为 Fix，T1L/T2L 为 Retention。

|任务|Single 正确/有效探针|Single micro|Single macro（主）|Sequential-146 正确/有效探针|Seq micro|Seq macro（主）|
|---|---:|---:|---:|---:|---:|---:|
|T0|146/146|100.00%|100.00%|146/146|100.00%|100.00%|
|T1G|578/578|100.00%|100.00%|500/578|86.51%|86.47%|
|T2G|512/557|91.92%|91.95%|504/557|90.48%|90.58%|
|T1L|2/2|100.00%|100.00%|2/2|100.00%|100.00%|
|T2L|25/54|46.30%|48.39%|25/54|46.30%|48.39%|
|T4G|0/0|NA|NA|0/0|NA|NA|

T0/T1G/T2G 均覆盖 146 个有效编辑；T1L 仅 1 个有效编辑；T2L 为 31 个有效编辑、29 个 native 来源组。T4G 没有 Base 原错误的有效探针，所以 0/0 是 NA，不是零分或 100% 修复。

表 2：主要非退化指标的 edit-macro 95% 区间，最后一列为最终前缀的来源组敏感性区间。

|任务|Single edit CI|Seq-146 edit CI|Seq-146 source-cluster CI|
|---|---:|---:|---:|
|T1G|100.00–100.00%|82.88–89.73%|82.89–89.86%|
|T2G|88.98–94.58%|87.27–93.49%|87.22–93.75%|
|T2L|33.33–63.98%|32.80–64.52%|31.67–65.05%|

全部观测相同的经验 bootstrap 会退化成 100–100%，不表示总体没有不确定性。尤其不能据 T1L 的 2/2 宣称稳健的图像局部性。

### 4.1 连续插入前缀

表 3：每个单元格为“正确/有效探针；edit-macro”。分母随前缀变化，前缀间不是完全同一探针集合的配对因果比较。

|前缀|T0 Fix|T1G Fix|T2G Fix|T2L Retention|
|---:|---:|---:|---:|---:|
|1|1/1；100.00%|4/4；100.00%|4/4；100.00%|0/0；NA|
|50|50/50；100.00%|178/196；91.00%|180/195；92.17%|11/17；62.96%|
|100|100/100；100.00%|355/394；90.00%|354/386；91.75%|17/34；50.00%|
|146|146/146；100.00%|500/578；86.47%|504/557；90.58%|25/54；48.39%|

从 single 全队列到最终 sequential 全队列，T1G 净少修复 78 个探针，macro 下降 **13.53 个百分点**；T2G 正确数净少 8 个，macro 下降 **1.37 个百分点**。T0 的插入时→最终轨迹全部为 `correct→correct`（146/146），但不能将其外推为所有泛化/局部性都不遗忘。T2L 两种部署的聚合计数相同也不证明逐条结果完全相同。

### 4.2 不能遗漏的 PostAcc 与角色变化

表 4：辅助 PostAcc 原计数。Base 与 single 使用原始挂接面板；sequential locality 排除已成为当前编辑目标的探针，T1L 分母因此不同。

|任务|原面板 Base 正确/全部|Single R0 正确/全部|Seq-146 R0 正确/当前面板|
|---|---:|---:|---:|
|T0|0/146|146/146|146/146|
|T1G|6/584|584/584|506/584|
|T2G|27/584|535/584|527/584|
|T1L|2/960|487/960|262/524|
|T2L|54/98|45/98|45/98|
|T4G|6/6|6/6|0/6|

最终 sequential 的 T1L 按预先冻结的 active-target 规则排除 436 次观测，剩 524 次；因此不能把 single 的 487/960 与 sequential 的 262/524 直接解释为同一集合上改善或退化。排除依据是运行前元数据角色，不能依据输出对错临时删除。保留下来的 Base-correct Retention 仍只有 2/2。

T4G 的 6 次观测来自 2 个编辑，原本全部 Base-correct。Single R0/RC 的 PostAcc 是 6/6，最终 sequential 为 0/6；单编辑 FORCED_ON 也是 0/6。这个附属小子集提示部署/专家选择相关干扰值得核查，但样本极小，不能推断完整 T4G 性能，更不能把其无 Fix 分母当作无需报告这些错误。

### 4.3 路由对照

在本报告所列主指标上，single 的 R0/RC/FORCED_ON 相同；最终 sequential 的 R0 与 RC 也相同。R0 与 RC 的相应 PostAcc 也相同。T4G single 的 FORCED_ON 与路由部署不同，已在上节明确列出。

这仅说明现有 RC 设置未在这些聚合语义指标上带来改善，不证明路由决策完全相同、阈值普遍无用，或错误一定来自 writer。没有在看到结果后搜索新的 κ。

## 5. 与已接受 BalancEdit single 的描述性比较

BE 使用同一 Stage17 冻结主队列、Base 掩码和 Astra/high 协议。下表只比较 **single 对 single**，不拿 C_NO_H sequential 与 BE single 冒充同一部署条件。BE 为披露过的 adaptation，并非这里重新执行的作者原实现。

表 5：每个方法单元格为“正确/有效探针；edit-macro”；差值为 C_NO_H−BE，单位为百分点。

|任务|C_NO_H single|BalancEdit single|Macro 差值（百分点）|
|---|---:|---:|---:|
|T0|146/146；100.00%|146/146；100.00%|+0.00|
|T1G|578/578；100.00%|578/578；100.00%|+0.00|
|T2G|512/557；91.95%|533/557；95.89%|-3.94|
|T1L|2/2；100.00%|2/2；100.00%|+0.00|
|T2L|25/54；48.39%|29/54；56.99%|-8.60|

T2G 中 C_NO_H 正确数比 BE 少 21；T2L 保持正确数少 4。可陈述本队列中的点估计未优于 BE，不能把这些差值写成统计显著劣于，也不能声称模型总体失败。当前没有完成跨方法的配对显著性检验；公开 C_NO_H 聚合中的 `paired=[]` 不包含此类检验。

这是不同原生配方的比较，并非等监督、等参数或等计算量的机制消融：C_NO_H 有多段初始化、U-KL 和 73,728 参数 writer；BE 是另一层的全线性更新、50 步配方。没有据此证明 U/H 机制或参数效率的因果效应。两次评分使用相同协议，但批次上下文和实际服务调用时间不同，snapshot 均未知。

其他基线及 BE sequential 的统一评分结果未纳入本报告；它们的生成完成状态不能替代已经验证的可比评分。本文不作全基线排名。

## 6. 工程证据与重要解释限制

1. **历史 Base 与实际运行环境的差异**：C_NO_H 在第 95 条初始化时曾触发 restoration guard。未编辑模型在同运行环境得到 52-token 输出，而冻结历史 Base 缓存为 46 tokens，第 41 个零基 token 处首次分歧；问题发生在编辑前。修复后使用同运行环境的编辑前输出验证恢复，同时单列历史缓存一致性；冻结 Base 答案、正确性掩码和队列未改。不能声称全程与历史 Base 逐 token 完全一致，也不能把输出差异自动认定为语义等价。该事实对共同 Base 可比性和因果解释的影响应由 Pro 优先审阅。
2. **Sequential 是独立专家回放插入**：记录的 3,744.58 秒（约 62.41 分钟）包含该阶段回放/生成，`replay=true`、`training_reused=true`；不能作为从头连续训练时间。该阶段 Torch 峰值 allocated 约 14.34 GiB、reserved 约 14.41 GiB，不是全项目或整卡峰值。本报告没有足够统一计时证据作完整效率优势比较。
3. **checkpoint 生命周期已结束的部分不可直接恢复**：C_NO_H 清理回执记录删除 144,604,338 字节生成状态，无权重备份，重建需要重训；已保留原始输出、tokens、输入/训练/评分绑定和配置，现有评分可继续复核。不能为新的分析假定所有权重仍存在。
4. **原始失效记录不会抹掉**：训练恢复和两次 Judge 网络恢复均有血缘记录；旧记录的 FAILED 表示历史尝试，不覆盖最终恢复成功。结果 JSON 的 `C_NO_H_REPORTED_NOT_YET_PUBLISHED` 是发布前快照；实际完成发布以公开提交及 publication receipt 为准。
5. **统计与语义范围有限**：仅一个冻结顺序；125 个 native 来源组不等于患者独立；T1G release 同图变体不等于独立临床图像泛化；T1L Base-correct 支持极弱；T4G 小样本；无 H 效果、完整十任务或临床安全结论。

## 7. 请 Pro 重点审阅的问题

请先审查现有证据和实验解释，再提出按优先级排序的最小补充分析。以下请求不是重新训练、重新评分、扩大数据访问或修改冻结协议的执行授权。

1. **结论是否与证据相称？** 当前合理表述是否应限于“C_NO_H 完成原始请求写入，但连续泛化、文本局部性和基线优势尚不成立”？哪些措辞仍过强？
2. **Base 漂移是否影响有效性？** 冻结 Base 标签与同运行环境未编辑输出不完全一致，哪些比较可保留，哪些只能降级解释？是否可先利用现有绑定与 guard 回执做不重评的敏感性核对？
3. **连续泛化下降的原因如何区分？** 能否优先利用现有路由记录、专家选择、距离/radius、输出绑定和已接受 verdict，区分专家竞争、拒绝、writer 泛化不足与评分/参考问题？现有聚合不足以定因，请不要直接下结论。
4. **T2L 与 T4G 应如何解读？** 29/54 的 c2w、T4G 6/6→0/6 是否提示需要限定论文主张？哪些能通过已有证据分析，哪些确需新增受控实验？T4G 不宜因 Fix=NA 被忽略。
5. **对 BE 的比较够不够公平？** 是否应先补现有输出上的逐编辑配对差值与来源组敏感性，而不是凭点估计声称显著优劣？非等预算与未知 Judge snapshot 应如何披露？
6. **完整方法的验证缺口是什么？** 在 E_H=0 下，C_FACT/H 因果机制仍未验证；如何明确区分 C_NO_H 的可报告结果、缺失支持和将来需单独授权的机制实验？

请输出：可支持的论文结论、必须修改或降级的结论、证据/实现风险、按优先级排序的后续建议。请将“只需已有材料的分析”和“需要新实验/新支持”的建议分开；不要为提高分数建议事后改 Judge、改样本、阈值搜索或混用旧阶段标签。

## 8. 可公开访问的证据

以下链接固定到已经包含评分结果的提交 `0ae6fa22a03fbc4a7d275a698225210165a89d60`；无需私有数据即可阅读聚合与代码。旧里程碑文字中的“尚未运行/待评分”是当时快照，当前 C_NO_H 状态以本次完成记录为准。

- [C_NO_H 原始聚合 JSON](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/cnoh_priority_closeout/CAMPAIGN_RESULTS.json)：主指标、全部前缀、R0/RC/FORCED_ON、PostAcc、区间、轨迹与成本。
- [C_NO_H 原始结果 CSV](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/cnoh_priority_closeout/RESULTS.csv)。
- [C_NO_H 已发布简报](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/cnoh_priority_closeout/GPT_PRO_REVIEW.md)。
- [BalancEdit single 聚合 JSON](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/single_be_closeout/SINGLE_BE_AGGREGATES.json)及[结果说明](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/single_be_closeout/GPT_PRO_REVIEW.md)。
- [冻结评价合同](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/EVALUATION_CONTRACT.json)、[方法锁](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/METHOD_LOCK.json)和[后续正式授权](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/formal/AUTHORIZATION.json)：早期合同的 pending 状态须结合后续授权与实际执行理解，不能改写历史。
- [第 95 条训练恢复及历史 Base 差异说明](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/formal/CNOH_RECOVERY_20260913.md)。
- [优先评分流程](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/formal/CNOH_PRIORITY_JUDGE_20260915.md)及[评分恢复说明](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/reports/medtrace_stage17_20260912/formal/CNOH_JUDGE_RECOVERY_20260915.md)。
- [聚合与优先评分代码](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/experiments/medtrace_stage17_20260912/scripts/medtrace/stage17_campaign_closeout.py)、[隔离 Judge 代码](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/experiments/medtrace_stage17_20260912/scripts/medtrace/stage17_judge.py)、[指标与区间代码](https://github.com/DLwbm123/Knowledge_editing-public/blob/0ae6fa22a03fbc4a7d275a698225210165a89d60/experiments/medtrace_stage17_20260912/scripts/medtrace/stage17_report.py)。

复算完整实验需要受权限控制的原始材料；公开聚合不声称足以从零重训。本文件不包含私有 QA、图像、回答、tokens、逐样本映射、服务器路径、凭据或权重。
