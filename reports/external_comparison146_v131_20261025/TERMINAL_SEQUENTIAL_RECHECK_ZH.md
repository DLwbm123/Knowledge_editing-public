# MEMIT sequential 终态实现复核

本次按用户2026-10-27要求核对最终7490个消费者、原始输出、评分绑定、292份编辑回执、首个single/sequence等价性、原生源码和已有独立GPU重放。新增GPU、模型forward/backward、编辑、生成、Judge均为0；未打断正在GPU0运行的MED，未修改原参数或旧评分，未重启MEMIT。

**结论：在已核验范围内，没有发现重复加增量、累积权重被重置/加载错、矩阵转置、前缀分片错绑或评分缺失置零的机械错误。现有结果明确显示这份固定MEMIT-MM在线适配在前几次写入中已经失稳。失稳唯一原因尚未经单因素干预证实，不能据此宣布原生MEMIT sequential失效。**

## 结果确实来自模型输出

| 已完成前缀 | 有效生成数 | 达到1024-token上限 | 正常EOS |
|---:|---:|---:|---:|
| 50 | 771 | 771 | 0 |
| 100 | 1474 | 1474 | 0 |
| 146 | 2093 | 2093 | 0 |

第6次即时插入输出首次达到上限；重复输出在首次并行、后续恢复之前已经出现。首个小组的详细性能仍保留私有。4条历史生成UNKNOWN位于prefix50，未当作错误分数补齐；7490个有效消费者的评分UNKNOWN为0。

## 实现与流程核对

| 排查项 | 证据与范围 |
|---|---|
| 首次single与sequence是否从同一个Base开始 | 首次优化完整trace及五层写入范数记录相同；50个重合输入的完整运行绑定和输出tokens均完全一致 |
| 是否实际累积、是否重复或遗漏编辑 | single146、sequential146份训练/保存回执齐全；逐次actual_inserted为1至146；5608次实际latent更新与终态账本相符 |
| 是否拼接独立single增量 | 原代码每次在当前权重上重求latent与各层key，按顺序写入；single结束才恢复一次Base进入独立顺序轨迹；恢复后从saved_owner+1继续 |
| 写入公式、转置、求解器 | 复用原GPU后端独立重放前6次，30次写入逐元素误差0；10个固定实际key与官方直接线性求解最大方程残差1.33e-13 |
| lookup位置与生成缓存 | 已有6次native因果前缀/teacher激活完全一致；6次生成前缀逐token复现；已有分片cached-forward及checkpoint加载准入保留 |
| 生成是否用错前缀 | 全7490项actual_inserted_prefix、mode/prefix/owner、查询/图像/参考与原计划一致；并行helpers只推理不可变的累积checkpoint |
| 评分是否串到别的问题/方法 | 全7490项原payload-key、Judge可见Q/参考/回答、tokens及high合同逐项通过；完整输出和评分记录都保留 |

语义评分允许相同Q/参考/图像/prompt/完整回答及tokens、相同Judge合同共用载荷。1027个消费者共用载荷的首来源运行信息与自己的运行来源不同，这不意味着预测借用了另一个模型：各消费者自己的运行绑定仍在其不可变输出中，复核已分别检查。不能把共享载荷最早记录的runtime误认为所有消费者的runtime。该复用合同未改动。

上述独立物理写入重放只覆盖前6次，足以覆盖已经出现重复输出的起点，不是全部146步均独立重放。元数据/账本完整不能单独证明所有浮点计算正确；同一源实现的重放也不能排除共同的算法或适配错误。

## 失稳发生在哪里

| 累积编辑序号 | 初始目标激活范数 | latent delta范数 | 五层平均写入范数 |
|---:|---:|---:|---:|
| 1 | 3.547 | 14.189 | 0.851 |
| 2 | 11.320 | 45.279 | 2.594 |
| 3 | 40.553 | 162.214 | 9.297 |
| 4 | 139.098 | 246.616 | 14.564 |
| 6 | 256.166 | 217.188 | 13.347 |
| 50 | 1572.837 | 197.792 | 10.059 |
| 146 | 4041.392 | 665.288 | 32.708 |

前三次delta均达到原生clamp4所允许的 `4 * 当前初始激活范数`。当累积状态本身放大时，这一相对约束的绝对上限也跟着放大。第6次source目标求解NLL仍为8.85，第146次仍为9.46，已经无法有效拟合新编辑，而非单纯遗忘旧编辑。

对原独立重放已保存的两组key做CPU分析：第4层key余弦0.943；第一次写入在另一组key上的线性响应约为自身响应的92.1%。这一层输入不受4–8层既有写入影响，所以支持存在较强跨事实耦合。后续层的key来自不同累积状态，不能将其当成相同Base的跨事实统计；两组key也不足以声称全146方向都塌缩。

## 原生设置与多模态适配的边界

重新读取了锁定的[官方MEMIT源码](https://github.com/kmeng01/memit/tree/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/memit)和[EasyEdit Mistral配置](https://github.com/zjunlp/EasyEdit/blob/b56ce8a300cf09410e93eb30643df86dd0a0ff6d/hparams/MEMIT/mistral-7b.yaml)：层4–8、25步、lr0.5、clamp4、moment权重15000等与本轮锁定值相符；原生fact_token为subject_last，本轮则是图像展开后的最后一个聊天prompt位置。

原代码的分层残差分配、两类context-group key平均、KL方向、Adam及停止顺序与官方算法相符；最终model logits对应Mistral原配置的最终loss layer31。native+FIT4、答案EOS、图像通用提示KL、512-token Wiki统计上限、最后prompt位置均是开跑前已披露的项目适配，不能把它们写成作者原生VLM实现。

更值得怀疑的机制是：最后聊天prompt位置未必能像subject_last那样区分不同事实；文本Wiki二阶矩没有覆盖图像/chat token分布；文本Mistral层位也未经过这台VLM的因果定位验证。它们与实测key耦合、相对clamp放大及source拟合失败相容，但现有数据不能分别确认其因果贡献。

因此，本轮0分应标为**固定MEMIT-MM在线适配失稳**。作为论文对照时应披露适配准入不足，不能把它当成已验证原生MEMIT不适用的证据或主线方法优势的充分依据。没有发现可直接修复并沿用旧分数的机械错误，因而未通过改层、改clamp、改位置或改统计分布重跑来追高分。

## 交付与限制

新增复核入口：[terminal_sequential_audit.py](../../experiments/external_comparison146_v131/terminal_sequential_audit.py)。完整匿名数值：[TERMINAL_SEQUENTIAL_RECHECK.json](TERMINAL_SEQUENTIAL_RECHECK.json)。既有GPU机械复算及数值后端差异仍见[CUMULATIVE_IMPLEMENTATION_AUDIT_REPORT_ZH.md](CUMULATIVE_IMPLEMENTATION_AUDIT_REPORT_ZH.md)，原始Q/A、tokens、key张量、主机路径与身份不公开。
