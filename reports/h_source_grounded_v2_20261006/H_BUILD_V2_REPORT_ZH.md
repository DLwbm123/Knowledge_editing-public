# MedTRACE H Source-Grounded V2 实际构建报告

日期：2026-10-06。基于 PR #28 的 V1，保留原 146 编辑成员、顺序、历史报告、冻结来源和 EVAL。本轮已完成规则修复、影响审计及真实模型试点；没有启动学生训练或 GPU Judge，没有修改 TT、路由或损失，也没有读取学生输出或评分。

## 本轮直接结论

- **修复后 V1 的 FIT verified 仍为 9/146**。关系数由 31 降至 **29**，不能把覆盖未降理解为原规则没有问题。
- 实际完成 **17 次 A 盲证据提取、29 次 B 候选构建、29 次 C 上下文隔离审核**，共 **75 次真实来源调用**，没有真实来源技术重试。另有 3 次完全虚构 fixture 模型调用、2 次本地故障注入；合计 78 次实际模型调用、80 次 runner 尝试，未达到 300 次上限。
- 固定选择的 **12 个当前 FIT 未覆盖编辑**中新增覆盖 **2 个**，均为 **SOURCE_GROUNDED、完成隔离审核**；新增 SOURCE_VERIFIED 为 **0**。
- 新覆盖 **2 个都属于非模态、非增强状态问题**。其中 1 个原始来源图内带答案文字，其未来训练输入状态为 **BLOCKED_ANSWER_TEXT_IN_ORIGINAL_FIGURE**。本轮证据构建接受不等于可自动用于训练。
- **严格近边界 verified H 为 0；本轮新增 grounded 近边界候选为 0，确认近边界 grounded 为 0**。历史未审核的解剖近边界、原图模态未核实候选仍为 FIT 4 / EVAL 4；它们不是本轮新增成功。一个新接受关系有同部位元数据，但通过的是明确否定指代，不能据此称近边界。
- 当前试点主要瓶颈是**当前指代与限定不完整、命题不能推出 target 不适用，以及需要来源明确支持的解剖关系**。接口已经接通。试点外大量候选仍是 **MODEL_NOT_RUN / NOT_YET_ASSESSED**，不能把它们概括成已证实来源不足。

## CT、模态修复及 V1 影响

`rules.py` 用结构化状态 NONCONTRAST / CONTRAST_WITH_QUALIFIERS / MIXED / UNKNOWN；静脉、口服、消化道、鞘内和直肠途径分别保存，缺失途径不推定为否定。明确识别 Non-contrast target，口服单一协议不会被粗略压成所有增强途径的一个二值字段。

图注必须指向当前整图或有可核验标签映射的当前 panel。比较检查、否定的模态、历史检查、不确定描述，以及无 panel 绑定的混合协议保持 UNKNOWN。生成 CT 答案后再次检查与 target 兼容性；不能靠固定文字声称所有 panel 的状态。保留的多 panel CT 关系采用私有的人工图像标签核验及精确图注片段映射，没有将该来源特例写入公开代码。

共审核 **41 条相关自动图注关系**，覆盖 FIT 和 EVAL。**8 条原 SOURCE_VERIFIED 降为 UNKNOWN**：FIT 2、EVAL 6，原因均是 panel 绑定未解决。私有 `acquisition_impact_audit.jsonl` 保存来源、原始证据、修复前后判定和差异原因。EVAL 的这 6 条只是另存 V2 审计结论，**原冻结 EVAL 文件、角色和等级未改写**。FIT 仍有其他合格来源，因此 verified 编辑覆盖保留 9。

另审核 **6 条原始当前图像模态 QA**，未发现新的指代歧义。另有 9 条历史解剖框架证书保留在非图注修复分支；本次没有证据支持宣布所有历史数据失效。私有 `other_verified_branch_audit.jsonl` 保存这 15 条检查。

## 两组状态及问题类型漏斗

冻结 V1 的 2865 候选中，1206 条归入规则路径（包括 1129 条来源/角色阻塞），**1659 条没有实现适用构建分支**。V1 记作 ENGINE_NOT_IMPLEMENTED / NOT_YET_ASSESSED。接通 V2 通用模型 runner 后，这批尚未逐条运行的记录记作 **MODEL_NOT_RUN / NOT_YET_ASSESSED**，保留 `v1_construction_status`，不伪造已完成医学判断。

V2 合计 2894 条记录（2865 旧候选 + 29 试点关系）：

| execution 轴 | 关系数 |
|---|---:|
| RULE_PATH_IMPLEMENTED | 1206 |
| MODEL_NOT_RUN | 1659 |
| MODEL_REVIEW_COMPLETED | 29 |

| evidence 轴 | 关系数 |
|---|---:|
| ANSWER_SUPPORTED_TARGET_EXCLUDED | 68 |
| ANSWER_SUPPORTED_TARGET_NOT_EXCLUDED | 4 |
| REFERENT_OR_QUALIFIER_UNRESOLVED | 34 |
| SOURCE_OR_ROLE_BLOCKED | 1129 |
| NOT_YET_ASSESSED | 1659 |

ANSWER_EVIDENCE_MISSING、EXECUTION_FAILED 仍是有效状态；本快照未出现相应最终记录。上述是候选关系计数，包含 FIT/EVAL，不能当作编辑覆盖或主线可用数量。

`FUNNEL_BY_QUESTION_TYPE.csv` 按原问题类型输出：编辑数 → 合法来源 → 进入对齐构建 → 实际审核 → 答案支持 → target 不适用 → 来源组隔离 → 最终接受。补列语义对齐确认与模型审核完成，避免把进入对齐路径误写为对齐成立。合法来源只表示本地许可/角色初筛，不等于可外发或能排除 target。`EXECUTION_EVIDENCE_BREAKDOWN.csv` 按问题类型交叉分解两组状态；私有逐编辑表仍有全部 146 个成员及具体缺口。

## 官方 CLI、外发许可与执行证据

实际官方 CLI 为桌面应用内 **0.160.1**，认证状态为 **Logged in using ChatGPT**；现有 npm 包入口缺少 vendor binary，不可用，本轮没有安装或修复它。模型缓存能列出多个模型，但实际可调用性只对执行过的 **gpt-6.1-sol** 建立了证据。采用私有可信本机的官方非交互 `exec --ephemeral --ignore-user-config --ignore-rules`，medium effort；没有读取/复制认证文件内容，没有提取订阅凭证充当 API key，没有新增付费 API、购买额度、重置限额或向公开 CI 传递登录凭证。

开始/结束的共享账号周额度使用率为 **40% → 44%**，ordinaryAllowed 均为 true；平台返回的非美元 credit balance 为 62500，未变化。账号其他任务也可能使用额度，不能把差额全归本轮。真实来源 usage：input_tokens 696200，cached_input_tokens 77440（属于 input 的子集），output_tokens 42677，reasoning_output_tokens 10417（保留 CLI 原字段，不重复相加）。**API 美元账单不可获得**，不伪造 token 费用或订阅成本。

每次调用保存实际选择模型、阶段、接口、CLI 版本、输入 hash、提示词版本、原始输出、解析、起止时间、错误、usage、进程命令和会话标识。上游不可变模型 snapshot 未提供。75 个来源调用均完成、会话标识互不重复、没有工具事件；所有 C 都是新会话，没有 resume，也不能读取生成器日志/verdict 或此前会话目录。macOS 文件边界 preflight 实测自身输入可读、生成器目录及项目记忆不可读。

A 只收到绑定原始来源，不含 edit target；同来源复用提取。B 收到问题/target、原始来源与提取事实。C 只收到原编辑、候选 QA 与引文及原始来源，不接收生成器判定、解释、自信度、提取结论或成功配额。两角色均使用同一模型，标记 **SAME_MODEL_SEPARATE_CONTEXT**，不是独立模型或临床专家审核。沿用 V1 三阶段模板并追加精确引用、问题命题和上下文隔离约束。

完全虚构 fixture 覆盖调用、结构输出、日志、缓存、UNKNOWN 外发阻塞和失败最多重试一次。最初两次正确模型输出被运行时警告分类错误拦截；保留失败并修正“警告≠工具调用”分类，随后 fixture 通过。真实来源没有为 UNKNOWN/FAIL 重抽样。

旧 427 条来源外发状态逐条记录：ALLOWED 210、UNKNOWN 216、BLOCKED 1。实际使用的旧 VQA-RAD 原始 QA 已核实作者公开数据许可 CC0；216 条 SLAKE 未确认外发协议，仍 UNKNOWN，**没有发送**。新资料只使用访问、许可、角色和整图图注绑定合格的公开 CC-BY 来源。模型只收到公开去标识来源文本，未发送图像像素。没有核实账号的数据训练/留存设置，**不声称零留存或 opt-out 已完成**。

官方路径参考：[非交互执行](https://learn.chatgpt.com/docs/non-interactive-mode)、[官方认证](https://learn.chatgpt.com/docs/auth)。来源许可参考：[VQA-RAD 作者公开登记](https://api.osf.io/v2/nodes/89kps/?include=license)、[关联 CC0 许可](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96c/)。具体医学来源 URL、证据与许可回执只在私有台账中保存。

## 固定 12 编辑及来源获取

选择在真实来源调用前锁定：FIT 未覆盖、4 病变/发现 + 4 单一指代定位 + 4 侧别/数量；现有来源中的字面否定检索优先，然后按原序，不看学生输出/评分，不用模态或增强问题填数。为检索排序访问过已有冻结角色的来源文字，因此不是未开发暴露设计。匿名原序位置为 **3、5、16、18、38、61、66、69、82、88、99、121**。

实际审核 **29 条关系**：旧缓存 24、新原始资料 5；其中 11 条重新包含 V1 旧候选来源。按整图聚合原 QA，重复 QA 不成为独立来源。每编辑 2–4 个来源组，最大 4，均低于 10。29 条全部接受/拒绝/UNKNOWN 与其原始模型输出、具体缺口均保存，不仅导出通过样本。

定向新增检索检查 **17 篇新原始资料**，取得 **4 个新整图—证据单元 / 4 个来源组**；3 篇访问失败、6 篇本协议许可未确认、8 篇许可/角色初筛通过，最终只有 4 个合格绑定单元。未达到 40 篇/60 单元上限，也未为了填满预算无限追加。取原始 JATS 和原出版者图像，人工检查原图与 graphic/figure 图注绑定；同篇/病例/重复图像与派生 QA 统一分组。角色预先固定为 FIT，没有为了补编辑搬动旧 FIT 到 EVAL，没有新增 H_eval。排除既有保护/评价角色、test 论文 ID 和全部 594 个未来 native 图像；新图未命中这些 native 图像标识。

本次新增原始资料的下载与图注绑定由受控检索/人工核验完成；`acquire.py` 负责根据私有固定计划**登记和资格核验已取得文件**，不宣称它自身实现了自动检索下载。

## 试点结果与具体语义缺口

结果：**2 grounded-reviewed 接受、2 拒绝、25 UNKNOWN**；3 个生成器/审核器分歧全部保留 UNKNOWN，不抽样寻求 PASS。`PILOT_ANONYMOUS_RESULTS.csv` 给出 12 编辑的来源/审核/结果计数。

两个新覆盖分别来自原始 QA 的明确指代不存在，以及新原图证据对目标复合答案中必要部分的明确否定。它们均保持原问题命题，未把另一疾病存在推成目标疾病不存在。第二例只排除目标的必要部分，不补全来源未说明的其他部分；原图同时有答案文字，未来训练前必须另行注册合格图像输入。**2 个证据覆盖不等于 2 个无条件可训练样本**。

实际未通过情况包括：多 panel 只否定其中一个/一个时间点；特定期相的否定不能泛化全部协议；双侧与右侧答案可共存；另一病变存在不能排除目标病变；来源给出位置却未给原问题要求的邻近关系；导管位置需要额外明确解剖包含/排斥证据才能否定 target；列表不完整或缺乏当前单一指代。不依靠常识补齐这些关键关系，也不统一把缺口写成“需要医生”。私有 `specific_review_gap` 按关系给出需要的指代、限定、完整列表、明确位置关系或额外原始证据。

## 医学等级、FIT/EVAL 和交付

| FIT 集合 | 编辑数 | 关系数 | 来源组数 |
|---|---:|---:|---:|
| 修复后 verified | 9 | 29 | 15 |
| 历史 grounded，未完成本轮审核 | 5 | 13 | 8 |
| 本轮 grounded-reviewed | 2 | 2 | 2 |
| any 合计（编辑/来源按集合去重） | 11 | 44 | 20 |

“模型审核完成”仅是执行状态。SOURCE_GROUNDED 不因两个上下文均通过而升级 SOURCE_VERIFIED；MODEL_ONLY 不进入主线。`h_fit_grounded_reviewed.jsonl` 只供未来另行注册的探索性消融，本轮没有训练。

不调用学生或评价模型，由 manifest ID 计算，与原冻结 EVAL 的交集：**FIT_verified ∩ EVAL_verified = 3 编辑；FIT_verified ∩ EVAL_grounded = 3；FIT_any ∩ EVAL_any = 6**。V1 冻结、V2 规则修订及加入本轮 reviewed 三个版本均为这些数值。`FIT_EVAL_INTERSECTIONS_ANONYMOUS.csv` 逐交集编辑给出 FIT/EVAL 独立来源组数，group_overlap 为 0；完整 ID/来源组在私有表。现有 FIT/EVAL 图像和已知来源组隔离，**患者独立性 UNKNOWN，EVAL 有历史开发暴露限制**，不能叫全新、完全未暴露确认集。

公开交付：本目录报告、匿名 SUMMARY/规则审计/漏斗/状态交叉表/试点/交集 CSV，以及 `experiments/h_source_grounded_v2_20261006/` 代码、19 个回归检查和通用状态模板。V1 的 18 个检查保持通过；V2 的 19 个检查通过。公开仓库没有原始医学证据、病例信息、图像、逐来源模型回执或认证信息。

私有交付包含：修订及完整候选 manifest、逐编辑覆盖、规则差异证据、许可/外发/新增及旧缓存来源台账、全部三阶段原始输出/解析/usage/执行上下文、失败与旧版本、固定选择/队列、`h_fit_verified.jsonl`、`h_fit_grounded_reviewed.jsonl`、`h_fit_grounded_pending_review.jsonl`、`v2_pending_manifest.jsonl` 和 FIT/EVAL 交集表。原 V1 与 EVAL 继续在原存储位置保留，本轮只创建新的 V2 交付目录。

本报告衡量数据构建覆盖和证据资格，不报告 M3Bench 或 TT 性能，也不因数据关系数变化推断方法效果提高。
