# MedTRACE Stage17-A：材料衔接与接口检查

## 当前结论

已完成本次**元数据衔接、预执行协议登记、CPU 合同检查和 GPU3 开发样本机械检查**。这不是正式 single/sequential 实验的完成报告：Stage17 的正式队列、Base 正确性分层和 H/G/U 支持成员仍未冻结，不能把历史 179 条写成本轮已经合法确认的 N。尚未训练或评分任何 Stage17 正式编辑。

公开起点为 [Stage16 收口 d30536d](https://github.com/DLwbm123/Knowledge_editing-public/tree/d30536dc31ec3472a5edd143a2c7749f9ec7a727)。Stage15/16 的数据、模型状态、标签及报告均未修改。Stage16 的 133 条新增覆盖、266 个 writer-probe、326 个接受的 Astra tuple 不重做；全 200 条 R0 输出保持 148/200 对 137/200 保留在历史报告，旧 Qwen 与新增 Astra 语义不合并。

## 实际核实的队列

本次只读取当前 V4 的十份小型元数据/锁，以及第一个已暴露 DEV 样本用于机械检查；没有打开正式队列原始 QA、Base 答案文件、sealed 或 T5 内容。下表来自现场读取的 `COHORT_V4_REPORT.json`，是**旧 Judge 下的历史可用性**，不是新 Judge 下的 Stage17 分母。

| 任务 | 历史候选编辑 | 历史可用编辑 | 历史可用探针 |
|---|---:|---:|---:|
| T0 | 189 | 179 | 179 |
| T1L | 115 | 14 | 53 |
| T1G | 179 | 179 | 704 |
| T2L | 411 | 358 | 601 |
| T2G | 179 | 179 | 676 |
| T3L | 1 | 1 | 1 |
| T3G | 9 | 9 | 10 |
| T4L | 255 | 75 | 75 |
| T4G | 220 | 114 | 197 |

T0 的 189 个历史候选中，10 个因 canonical Base 原正确被移除，留下 179 个、159 张 native 图像。旧诊断 200、amended189 和 superseded 清单没有重新启用。T2L 等任务显然不是同一个 179 编辑流；以后必须分别登记任务队列及与 T0 流的映射。

DEV16、QUAL8、QUAL16、SEQ16 的冻结选择元数据已定位；DEV16 与 QUAL16 的事件 ID 不重合。但事件级选择不等于完整的来源暴露台账，SEQ16 被选中也不等于已执行。未核实的重复事实数、来源连接组数、开发/评价/未开发来源分层都记 `null`，不伪造 0，也不声称“未见确认”。

## 支持的具体缺口

`E_U / E_H / E_HG / E_HG_eval` 已定义并实现三态成员合同：合法、明确不合法、待核实。**当前成员和 S 均为待核实，不能当作空集。**

- U-fit：已有 train-only 支持可复用，但缺少与整个正式编辑—探针队列的来源组排除连接，以及各编辑自己的 teacher 绑定；不能删除 KL 后运行 NO_H。
- H-fit：须核验相同命题、不同图像、来源答案冲突及 out-of-scope 关系；当前缺全局来源隔离连接，不能只凭“换了图”判合法。
- G：可复用既有 Stage14 的训练来源候选，但需与 H 槽位配对、排除 U 和评价来源，并验证其为不同命题；不能把 G 当 G+。
- H-eval / U-eval：缺正式队列上的预先关系标注、与 fit 的来源隔离及相应访问授权；复用一个 U-fit 不是多个独立评价问题。

Stage13R 原有来源摘要已经记录完整 SLAKE 训练来源 9,835 QA/586 图像及合法来源修订，Stage14 有 7 个已匹配 G 的旧编辑支持。它们只是可复用来源线索，**不是 Stage17 的 7 个合法 H/G 编辑证明**。本次没有重新扫描旧 875/4,490 候选池，没有下载数据，没有增加医学事实或代签人工临床审查。sealed、质量排除、保留数据、T5/PadChest 保持原权限。

## 冻结与兼容性

C 分支保持 V1：layer21 down_proj，freeR4/73,728 FP32，自己的 native CP → A2(80) → CP-W0(320)，共享初始化，continuation320，原 Adam/归一化和 native/fit/U 调度；仅 H/G 槽位区分 FACT/EXTRA，缺 H 不退化冒名。

现场确认 LoRA-Perf-v1 为 r16/alpha16、全语言 MLP gate/up/down、lr=5e-4、固定 80 步。其历史 QUAL16 `QUAL_VALIDATION_FAIL` 和 readiness=false 保留，不重搜。GRACE/BELoRA 沿用冻结实现；BELoRA 实际为 **V2 effect-repaired 50 步**，原 paper-spec 是 5 步，明确作为偏离报告，不伪称作者实现或 paper-exact。

新研究面板预登记 C/BE 的 R0 主路由、旧 RC=0.7696741135364367 迁移对照、single FORCED_ON 机制诊断。不采用 Stage16 事后 κ 区间。BE 正锚点必须衔接合法 native-only fit，不能沿用旧官方评价改写作为训练锚点。

实际 canonical 生成是原生 Mistral 提示、greedy、1024-token 上限，已在本次真实模型检查中确认。早期 helper 的 128-token 常量及部分 teacher 路径不能代表 Stage15/16 全部生成。旧 raw 只有完整输入/runtime/解码绑定一致才能复用；旧判定不跨新 Judge 协议，旧 checkpoint 不按方法名直接接入新编辑。

评价版本固定为 [论文 v1 A.5/C.2](https://arxiv.org/html/2607.05310v1) 和 [release 03c6fda 的任务定义](https://github.com/BioMed-AI-Lab-U-Michgan/M3Bench/blob/03c6fda3813301dab3be5831fdc94b493c10afc9/TASK_DEFINITIONS.md)。新协议采用原正确 locality retention/c2w、原错误 generality Fix 主口径及全探针 PostAcc 辅口径；OutputAgreement 单列。论文与 release 的 T1G 图像构造等差异保留，不能称精确复现。零分母 NA；PairCorrect 不排除 native 失败；Base 掩码先于学生冻结。统计预登记 edit-macro、probe-micro、10,000 次 bootstrap/seed=20260912 和来源组敏感性。

## 工程检查与成本

CPU 检查结果见 `CPU_TEST_RESULT.json`。复用已有 core/selective-write/bank/editor 测试，新增分母、三态支持、完整缓存绑定及只读 inventory 检查；没有重写框架或添加性能资格线。

GPU3：第一个既有 DEV native，1 编辑，11 项检查全部 PASS。四次 canonical 生成和一次 1-token 生命周期探针，共 5 次；训练步数 0，Judge 调用 0。验证零 freeR4 效应、OFF/移除恢复、非零专家驻留后的 Base 特征隔离、save/load、predictor/EOS、无 gold 路由、self-route、hook 清理、Base 参数版本/存储指针不变及参数数目。非零扰动只用于接口检查，不是算法变体或训练结果。没有用生成对错挑选样本。

用时 157.83 秒；峰值 allocated 15,316,638,208 bytes（约 14.27 GiB），reserved 15,393,095,680 bytes（约 14.34 GiB）。这是一个开发样本的机械峰值，不是未来训练/大 bank 显存承诺。GPU 进程已结束；中性命令行与 GPU3 UUID 均已核实。挂载、容量与小型写读探针通过；没有回退系统盘大写入。

## 正式运行为何尚未启动

附件默认授权 Stage17-A，且明确不自动扩展 formal/sealed 访问、云端 Judge、完整训练或旧资格锁变更。用户本轮补充了 GPU3，并未另行批准这些扩展。旧 LoRA-first 约束不能静默取消。

需要明确批准：在保持 sealed/T5/人工临床审查边界的前提下，处理未封存 M3Bench T0–T4 候选/评价材料；用一次统一的 Stage17 Judge 配置覆盖 Base 与方法；以新的 Stage17 协议允许低准确率仍作为有效测量（旧锁不改写）；数据与机械合同满足后执行计划内 single/sequential。这不是要求放宽来源合法性或工程错误检查。

随后才能完成真正的来源角色/暴露连接、固定 Base 掩码与实际 N/S，并补齐 Stage17 formal adapters。`RUN_MANIFEST.json` 按分支列明：不依赖 H 的四基线、E_U 的 NO_H、E_H 的 FACT、E_HG 的 EXTRA/共同流，以及授权和数据缺口。正式入口尚未实现的命令为 null，不能拿旧 Stage3/15 coordinator 冒充新队列入口。缺 H 不阻塞合法基线；顺序流不能以全流切片冒充子流，也不允许将 single LoRA/GRACE 状态拼成连续系统。

## 可交付范围

公开：新增小型代码、测试、协议、聚合元数据和机械检查摘要。私有：来源路径、事件 ID、完整元数据 ledger、DEV QA、raw/tokens、模型及执行配置。公开发布回执记录对应内容提交及匿名访问检查；GitHub 发布不代表正式实验已完成。无新增轮询、定时监测、后续阶段或语义重评任务。
