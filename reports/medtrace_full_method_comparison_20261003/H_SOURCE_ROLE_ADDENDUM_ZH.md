# 原始来源角色最终准入补充

在未启动任何模型或 Judge 的数据审阅中发现：VQA-RAD 的公开 JSON 同时包含 `freeform/para/test_freeform/test_para`。旧角色汇总不足以区分全部作者测试记录。初次补充的 77 关系/10 modality 证据全部保留为 preliminary，不以它们直接执行训练。

最终 H 数据准入重新应用以下固定保守规则：任一 VQA-RAD 原始记录为 `test_*`，整个图像及已知 case URL 都进入保护集合；只用 `evaluated + freeform` 原始问答，不用 `para`，不借其答案填缺口。沿用整个历史角色/Stage17/formal/reserved 的排除。不读取新模型结果，不改变损失/队列/门槛。

已有原始作者 `qid_linked_id` 与 `strict agreement` 提供两条 modality 问法的对应证据：`What image modality is this?`、`What type of image modality is this?`。这是无额外医学限定的同一变量询问；补齐模板解析，继续以互斥 CT/MRI/X-ray 大类作唯一直接 H 证明。没有新的含糊文本匹配或临床签核。

按该规则重新生成一份独立 final 清单及预算回执，不覆盖 preliminary 原始证据。原始数值型答案的 representation 修复、服务器下载网络失败及全部成本继续保留。有效来源/关系仍有缺口时如实报告，禁止自动将 146 改成部分队列或 NO_H。
