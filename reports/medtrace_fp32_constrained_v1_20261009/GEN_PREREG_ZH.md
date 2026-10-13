# 条件自由生成：在局部通过之后、生成之前锁定

局部主63题KL降低56.3167%，八专家实际编辑收益均严格保留至少90%，因此执行此前授权的唯一条件消费者。旧FP32资格失败、原63→62及全部历史阴性保留。本阶段不更新参数、不求解、不重建候选、不改变阈值；复用8个已冻结小候选包。

## 固定执行

- 原8专家、原seed和RAW/BOUNDED/MATCHED_RAW三臂；96留出及每专家native+4FIT。原63主面板不按FP32重筛；全96次面板，不新增BASIS输出。
- 3×8×101=2424候选消费者，加40 FP32 Base编辑输出，共2464消费者。原FP16/FP32留出Base复用已完成PR43/49输出。最多2464新生成，每条仍greedy、最多1024 tokens、原EOS与prompt。
- CPU在任何生成之前逐一检查RAW与MATCHED_RAW全部TT张量。若完全相同，同一输入/同一运行配置下只生成RAW，MATCHED_RAW保留显式别名及独立消费者；否则分别生成。此复用基于参数身份，不依据输出或分数。若8对皆相同，1656实际新生成，808复用。仍是TT映射范数匹配，不是编辑收益匹配。
- 冻结原FP16多模态prefill，然后原加载权重值转mathSDPA FP32、TF32关闭。沿用PR49已通过六个原生control的实现；本轮不追加控制生成。每次prefill恰好消费一次、无Base梯度、结束复原TT、总forward hook计数不超过实际新生成×1024，0 backward/optimizer。
- GPU2–7六worker，专家按原序交错分配；启动需每卡至少48000MiB可用及存储探针。预计生成10–30分钟，极长回答可更久；后台有限依赖链不依赖交互会话。继承43.9942884898848 GPUh、12412 Judge，所有失败成本累加。

## 评分与复用

全部输出及tokens、图像/prompt/runtime、候选来源绑定先落盘，再启用原隔离Astra medium协议，4个不重叠worker、50条一批、每个新payload仅一次，无降级或隐式重试。Judge只看原协议四字段。RAW/MATCHED_RAW身份复用沿用同payload；完整payload相同的其他消费者也可共享。

固定输入query/image/question/reference及原始回答文本全部相同，可继承PR43/49已有接受评分，并记录来源。仅tokens差异不改变原文本Judge输入；不得将变更文本继承为正确，不根据正确率挑选复用。图像、问题或reference任一不同均拒绝复用。原始missing不自动补评。原FP32资格157条已有评分完整，复用其固定留出子集。

## 固定报告与判定

分别逐题记录相对FP32 Base及原FP16 Base的保持正确、新增损伤、恢复、仍错、缺失；公开仅聚合，私有保存逐题。主63与全96均按臂/专家汇报；原精度迁移造成的1条损伤仍保留。native+4FIT共40编辑输入分别报告Base与三臂语义正确数，不把90% NLL收益称作90%语义收益。

主比较BOUNDED−MATCHED_RAW正确率，以原expert和source两种单位分别配对10000次bootstrap（沿用固定stats seed）。只有两种95%CI下界都>0且编辑正确数不低于RAW，才称DEVELOPMENT_SEMANTIC_SIGNAL_ONLY；任何缺失为INCOMPLETE_SEMANTIC_EVIDENCE，另给正确数与缺失数；完全同逐题判定为NO_SEMANTIC_SEPARATION，其余NO_ESTABLISHED_SEMANTIC_PROTECTION_GAIN。不得因此改参重试、扩大24/146或完整训练。单步响应微小导致语义无差异是合法结果，不反推局部机制无效。无独立确认或临床保护声称。

## 生命周期与交付

8小包的最后消费者是本阶段完整生成。2464消费者、原始tokens及完整Judge绑定全部落盘、六worker结束后，按准确清单删除这些本轮包；不等Judge，不触碰历史数据/模型。中断保留尚需消费的包，禁止先删后重建。累计账本、失败、删除回执、源代码和匿名报告进入同PR50；医学原文/图像/tokens与私有路径留在原主机。
