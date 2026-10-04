# 六臂评分与汇总实现

沿用父冻结 Qwen3-32B-AWQ 身份、单条上下文、32并发、温度/seed0和原Stage17提示，所有六臂完整生成后，按固定输入单元打乱盲评。正式分母仍为原146，H8另分层。GPU仅5/6，Judge驻留计原16GPUh和首次48h时钟。

同完整输入/图像/prompt/runtime/generation/rawtokens/Judge身份的父Qwen结果须带原批次、原record、原attempt和epoch证据继承；有效结果与永久missing都不新请求。新旧消费者训练/路由/全库祖先单独验证。不得按答案字符串复用，不把历史Astra/SOL写进共同表。新key原子预留、单attempt；失败不重判。

报告按冻结七项主/次对比和 `(TK_H1−TK_NO_H)−(CP_H1−CP_NO_H)` 交互计算共享key有理数精确差界。原mask主表、new-Base敏感性、owner/selected分层、原全库prefix和固定panel保持。原主表及H8面板按冻结seed做10000次编辑和来源连通组bootstrap；小样本、患者不明、无独立H_eval、无新压力面板均披露。

Tucker候选门仅比较TK_H1与FREE_H1，按预定8/146 T0、single H8和final全库T1G/T2G不劣于−1pp、T2L宏/微差界不低于0、参数载荷至少10倍压缩。空必要分母或仅因missing无法判定则INCONCLUSIVE；明确上界也未达则NOT_SUPPORTED。不得以此自动替换主线或追加第三轮。

存储报告分别记录参数载荷、实际pt序列化字节和推理展开参数；读取/展开耗时采用收尾CPU测量，不能冒充GPU峰值/训练计时。GPU/存储精确峰值未知，快照下界照报。汇总工具只读原结果；后续清理由已结束消费者清单另行执行，父依赖绝不删除。

CPU资格验证在合成临时数据库进行，涵盖完整身份变化、重复预留、永久missing继承、无请求恢复、篡改tokens拒绝、shared missing交互抵消、空分母与三值判定。实际正式队列必须等GENERATION_COMPLETE及父完整终结与绑定门，不在运行中按分数选择子集。
