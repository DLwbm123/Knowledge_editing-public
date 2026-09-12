# Astra 评分材料：操作说明（不要交给评分模型）

本包仅准备材料，未调用任何评分模型。只使用原本未评分的输入；已生成回答、
历史 Qwen 评分、checkpoint、训练配置和封存集保持不变。

## 文件分区

- `judge_only/`：可用于独立评分的提示词、固定输入和对应 JSON Schema。
- `operator/`：原始输入备份、私有映射、协议来源、校验器、执行记录和回收结果。
  **不要上传这个目录，也不要把整个材料包目录作为评分模型可访问的工作区。**
- `JUDGE_ONLY_PRIVATE.zip`（如提供）：只含 `judge_only/` 材料，但仍是私有问答，不得公开发布。

原始问题、参考答案和模型回答没有改写；`adjudication_pass` 仅由操作端保存和恢复，
评分端不接触方法映射、图像路径、历史成绩、token、协议来源或项目背景。

## 网页端或 Codex 桌面：每批一次

1. 确认这些固定评价问答允许提交到所选云端服务；不得上传患者身份信息、无权外发的
   内容或原始影像。用户目前授权的是材料准备，**本包不自行启动评分或上传**。
2. 为每个批次建立真正独立的上下文，选择 **GPT-6 Astra**，固定同一推理档位
   （建议 `high`，按界面实际名称记录；不要中途换模型或档位）。不要复制/分叉本项目
   长对话；关闭项目记忆、跨对话引用、无关个性化指令和连接器，避免自动加载研究仓库
   的 AGENTS、技能、记忆和结果。如果平台无法隔离这些内容，不能声明是独立盲评。
3. 每次只粘贴或附加一个 `batch_NNN.prompt.md` 的完整内容。它已包含本批全部输入和
   输出规则。不要只让模型搜索附件摘要，不要截断记录，也不要混入本操作说明。
   如需文件工作区，只复制该批材料到隔离目录，不给原项目或 `operator/` 的读取权限。
4. 保存最终 JSON 为 `operator/responses/batch_NNN.json`。不要保存推理/工具事件流，
   不把 Markdown 围栏、解释或多份候选答案拼接进去。每条判定须由目标模型实际产生；
   禁止用字符串相等、脚本规则、随机数、常量、人工补齐来冒充 Astra 判定。
5. 在材料包根目录运行（Python 3，仅标准库，不需 GPU）：

   ```sh
   python3 operator/judge_io.py validate --input judge_only/batch_001.input.json --response operator/responses/batch_001.json
   ```

   依次替换批次号。仅 `FORMAT_VALID` 才表示格式和本批覆盖通过，**不是医学正确性认证**。
   校验拒绝重复 JSON 键、额外字段、字符串/数字冒充布尔值、错批次、漏行、重复/陌生 ID、
   乱序、围栏及截断输出。不合格原始响应另行保留，不静默修标签、不将缺失记为 false。
   本版不自动重试；失败批次交由操作人处理并记录，不能根据评分高低择优重跑。

## 完成后汇总

复制 `operator/EXECUTION_RECORD.template.json` 为 `operator/EXECUTION_RECORD.json`，
按真实执行填写模型、界面、推理档位、完成时间、隔离/数据权限确认，以及每批任务 ID、
截图路径或导出记录的引用。平台未提供不可变模型版本时保留 null，**不要伪造 snapshot**。
这只是操作人证据记录，格式校验器不能独立证明云端确实使用了哪个模型。

```sh
python3 operator/judge_io.py merge --bundle . --execution operator/EXECUTION_RECORD.json
```

仅在所有批次及全局覆盖通过后，生成 `operator/VERDICTS_ASTRA.jsonl`；已有文件不会覆盖。
每行恢复 `opaque_query_id`、`adjudication_pass`、布尔判定和明确的 Astra 来源。
JSON Schema 只能限制结构，ID 顺序/重复/完整覆盖仍由本地校验器检查。

## 评价协议变更与后续接入

本轮改用 `medtrace-stage16-astra-semantic-v1`：沿用语义评分原则，改变评分模型、运行环境、
批次上下文和输出传输格式。原 Qwen 的 `temperature=0`、无 thinking、24-token 输出限制
不能冒称为 Astra 的同等配置。提示词和单次响应都不能保证跨运行判定完全一致。

**不要把 Astra 输出覆盖原 `private/judge/OUTPUT.jsonl`，也不要直接运行旧
`stage16_coverage.report()` 来接收它。** 旧汇总会组合历史 Qwen 语义分数；同一指标不能混用
两种 Judge。后续接入须单独标记新恢复子集的 Astra 结果，历史 Qwen 结果保持原样；
除非另获授权，不重评历史、不增加实验、不调整方法或阈值，也不替代明确要求的人工临床审查。

## Codex CLI 说明

如果使用可用的 Codex CLI，可用对应 `.schema.json` 作为 `--output-schema`，并用
`-o` 保存**最终响应**；`--json` 是工具事件流，不是本协议的判定输出。
CLI 登录、隔离配置和模型可用性须先核实，本包不安装 CLI、不复制认证文件、不发起调用。

参考：[Astra 提示指导](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)、
[Codex 结构化输出](https://learn.chatgpt.com/docs/non-interactive-mode)。
