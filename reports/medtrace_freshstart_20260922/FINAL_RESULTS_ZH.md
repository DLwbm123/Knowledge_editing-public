# MedTRACE FreshStart 实验结果（2026-09-23）

## 结论

本轮在新服务器上完成了 canary、正式训练和生成，并于 2026-09-23 01:46 UTC 写出最终回执。最终状态是 `CLOSED_WITH_DOCUMENTED_LIMITATIONS`，方法状态是 `NO_VALIDATED_CANDIDATE`。这是一轮有明确结果边界的小规模探索，不构成方法验证成功、历史结果复现或临床泛化证据。

实际冻结样本为 **16 条源标签编辑、22 张图像**，并非 DEV45；患者或研究层面的独立性未知。Base 输出、CP/A2/W0 和各臂编辑状态在本轮重新生成，未与历史结果混合。canary 与 baseline、primary、secondary、order 四个正式阶段的进程退出码均为 0；九个正式运行项均记录了 16/16 的训练和生成前缀。BalancEdit 是论文规格适配实现，并非作者代码。

## 评分覆盖

固定前缀共有 **304 个评价消费者**，其中 **122 个未评分**；评分任务完成 **27/46**。下表为最终前缀 16 的 source-answer agreement，格式是“已知正确 / 已评分 / 应评分”，所以未评分项没有被算作错误或正确。

| 运行项 | Native | H_eval | U_eval |
| --- | ---: | ---: | ---: |
| BalancEdit 论文规格适配 | 10/10/16 | 1/1/7 | 0/1/1 |
| Primary F0 | 10/10/16 | 1/1/7 | 0/1/1 |
| Primary FA | 10/10/16 | 1/1/7 | 0/0/1 |
| Primary FH | 10/10/16 | 1/1/7 | 0/0/1 |
| Primary FR | 10/10/16 | 1/1/7 | 0/0/1 |
| Secondary F0 | 10/10/16 | 1/1/7 | 0/1/1 |
| Secondary FH | 10/10/16 | 1/1/7 | 0/1/1 |
| Order F0 | 10/10/16 | 1/1/7 | 0/1/1 |
| Order FH | 10/10/16 | 1/1/7 | 0/0/1 |

这些分数只描述已评分子集。尤其 Native 尚有每臂 6 项未评分，H_eval 尚有每臂 6 项未评分；不能把 `10/10` 或 `1/1` 当作完整前缀准确率。各臂完整表现及差异无法由现有评分确定。严格无关样本资格未确认，相应保持率为缺失值；空分母也保留为空值。`METHOD_LOCK.json` 将 FH 记作参考臂，同时明确没有经验证的候选方法。

## 协议边界与资源

缺少独立改写、正向图像评价面板、严格 U 资格和独立确认队列，完整保护条件无法验收。当前 Judge 仅评估来源答案一致性，不是独立影像专家判定。评分桥最后保留了 `FAILED_NO_RETRY` 记录；本报告没有重试或补判未完成任务。

GPU 累计驻留约 **20,608 秒（5.72 小时）**，账本状态为 `GPU_SESSION_CLOSED`；实际云端费用未记录。实验状态及恢复材料目前仅在该服务器，没有异机备份。原始问题、参考答案、模型回答、tokens、图像、身份映射、逐样本记录、权重和私有 Judge 证据均未公开。

## 证据与源码

本目录的 `FINAL.json`、`METHOD_LOCK.json`、`RESOURCE_LEDGER.json`、`CLAIM_EVIDENCE.json` 和 `AGGREGATE_METRICS.json` 是脱敏聚合证据。后者从服务器 `public/METRICS.json` 提取各臂、前缀和角色的分子、分母及缺失状态，省略源组与逐样本明细。服务器最终报告为 `/root/rivermind-data/job-622/state/public/FINAL.md`，运行标识为 `fresh-20260922T103133Z`。

执行源码绑定提交 `7a9b1f22dc410dd8322b5813aaee935c93c38a79`；公开的核心源码位于 [`experiments/medtrace_freshstart_20260922`](../../experiments/medtrace_freshstart_20260922/README.md)。私有 Judge 传输桥及运行数据不在公开仓库中；仅凭公开材料不能复算未评分项或重新执行涉及私有数据的实验。
