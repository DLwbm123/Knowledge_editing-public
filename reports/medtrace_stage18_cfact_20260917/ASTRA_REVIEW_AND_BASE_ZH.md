# Astra 审阅与新 Base 筛选（2026-09-17）

本轮按用户“直接用 astra 审阅”完成了看图审阅，并在 GPU3 完成 13 条新 Base 生成及统一 Astra 判定。**当前严格合格的新增训练队列为 0；没有启动新的 student 训练。** 算力与训练权限已具备，阻塞来自具体数据资格。

## 实际完成

|检查|结果|边界|
|---|---:|---|
|Astra 关系审阅|10/10|独立 agent、无学生预测输入；记录为 AI，非人工签字|
|关系通过|2|均为 H_eval；属于同一个候选编辑|
|关系尚不确定|8|6 条最大器官范围含糊；2 条胸腹交界部位含糊|
|严格通过 H_fit|0|不能把 UNVERIFIED 自动改成通过|
|患者／研究独立性|10/10 UNKNOWN|不同文件、图像组或模型审阅不能证明不同患者|
|新 Base 生成|13/13，exit 0|GPU3、FP16、冻结生成配置，无 writer；约 68.3 秒|
|Astra Base 判定|13/13，exit 0|10 正确、3 错误；无语义重试、无工具事件|
|候选 native 资格|2 错误、1 正确|新增 native 已正确，不满足 Base-wrong|
|最终可训练新队列|0|目标 16 的合格缺口为 16；来源包数量 3 不等于合格数量|

最大器官的 6 条关系中，各自作者 Lung/Liver 标签与图像基本一致，可以带限定地作为作者 QA 的开发诊断；Astra 没有确认统一面积／体积／视野语义下的严格冲突关系。另两条 Chest 来源位于胸腹交界，保留实质范围歧义。没有为了启动训练修改题目、答案、审阅结论或 Base 筛选规则。

两条旧 native 的 Base 确实错误，但其 H_fit 和 H_eval 关系仍未严格通过。新增候选有两条通过的 H_eval，然而 H_fit 仍不确定，且 native Base 已正确。这是三个候选逐项交集后的零资格，不是只重复报告“缺 H_eval”。

## Smoke 与后续训练的区别

此前 2 条 smoke 的 C_NO_H、C_EXTRA、C_FACT 共 6 个 continuation 分支已完整训练，各 320 步；H/G 梯度、W0、公平预算及保存恢复均有真实 GPU 验证，见 [smoke 验收](SMOKE_ACCEPTANCE_ZH.md)。其机械验收仍成立，不能据此断言严格 H 关系或未见 H 泛化已验证。本轮保留全部 checkpoint，没有重训已有分支，也没有把 smoke 改标成正式 pilot。

用户授权使用 Astra 后，先前“必须等人工签字”的执行阻塞已移除；但 Astra 本身给出的 UNVERIFIED 和无法恢复的患者信息仍保留。本轮没有以 AI 代签临床医生或推断患者身份。

## 下一步的具体材料缺口

1. 两条 Base-wrong 旧 native：需要问题范围明确、标签冲突成立的合法 H_fit，以及与全部训练隔离的 H_eval；当前作者 QA 可保留开发诊断，不能自动升级严格资格。
2. 新增 native：即使补好 H_fit，也不能加入当前 Base-wrong 队列。不能改答案制造编辑需求。
3. 已授权有限来源池没有新的完整合格包。建设优先使用 [来源方案](DATA_SUPPORT_BUILD_PLAN.md) 中的全新 PMC OA 来源；下载授权仍未收到。方案上限 64 个 DEV 来源组，逐图核实 CC-BY/CC0 与第三方图权利；保留文章／病例／研究／派生图分组和 UNKNOWN，Astra 可完成此次已授权的 AI 审阅。不得把历史 eval 挪成 fit，也不混入 sealed 数据。
4. 只有来源、关系与新 Base 筛选交集非空后，才按冻结四方法 single／短 sequence pilot 在 GPU3 训练。实际 N 可小于 16，不能靠重复凑数。

## 可复算证据

私有材料保留完整 10 条看图审阅、原队列、新 Base 文本和 tokens、运行绑定、匿名 Judge 输入、13 条判定、隔离测试和三候选资格表；不公开患者图像／逐例 QA。公开代码 `stage18_base.py` 只生成 Base，`stage18_base_judge.py` 复用原封闭 Judge，`stage18_pilot.qualify` 校验绑定并取资格交集。

Base worker 源码为 `623c0f1ea2a77d33552523a0c2dc64dc9270bffb`；运行 `job-20260917-1802`。Judge 为 `gpt-6-astra`、high、协议 `MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1`、不可变 snapshot 为 null；五项本地隔离检查全部通过，没有模型替换。新 runner 没有改动 Stage17 已冻结结果。
