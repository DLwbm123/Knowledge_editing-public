# V26实现与机械准入

按ponytail技能的最小改动原则复用原V24/V25执行与生命周期框架，仅新增源跨度映射、属性等权目标、原B重建及必要准入。模型、源监督、优化器、步数、生成路径不变；0新参数。

CPU使用本地实际tokenizer和原Mistral对话模板，重建source-only完整监督target ids，验证逐字源跨度/中文byte tokens映射、单位等质量、EOS及原模板末空白各保留native质量。全1权重损失与梯度精确，原B初始化/三步优化器/保存加载检查PASS。CPU的随机小模型架构检查属于继承的机械检查，不当作预训练模型效果。

首次GPU准入在任何正式更新/生成之前拒绝了CPU遗漏原模板末空白分隔token的映射。第二次在CPU/native target ids已一致时，重复前缀解码的非单调检查拒绝了一个分片；另外两个分片已数值PASS。最终绑定改用CPU冻结的字符→token组表并核对真实target ids逐项精确一致，不改变参考、单元边界、权重公式或容差；已完成两个REAL_PASS仅在其groups/weights与CPU表精确相同后复用，不重复成功模型调用。两次失败均全0正式update/gen，累计成本、原始24h/首次GPU2h截止完整继承。全量失败trace/身份/计数私有归档，未掩盖或改成科学阴性。

真实检查沿用rtol1e−5/atol5e−4：零残差、teacher/prefill/KV decode、原B mask，全1native损失/梯度精确，以及实际M源权重下有限非零梯度。五分片完整PASS才TRAINING_RELEASE；S原161点源损失及完整回答/tokens再对旧V24 B重建，任何不符停止。最新运行回执以STARTUP及私有STATE为准，不将启动/机械PASS称作科学成功。
