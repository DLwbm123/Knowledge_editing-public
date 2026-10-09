# Base源回答资格核验

结论：SEMANTIC_SUPPORT_AVAILABLE。BASIS正确61/96，缺失0；HELDOUT正确63/96，缺失0。具体三题型、来源支持和缺失界见RESULTS。

严格内容全token正确0/192，EOS正确191/192。原标注与生成prompt前缀一致192/192；首teacher预测与实际首生成token一致192/192；生成文本往返token相同192/192。上述诊断不能单独证明医学正确或实现错误。

全部192Base生成和192标注前向完成，0更新/权重，原隔离Astra medium每payload一次，不隐式补判缺失。新增GPU进程小时0.076377，累计40.460981；新增Judge192，累计12406。原图、文本、tokens和逐题关联仅私有保存。尚未登记新的投影训练。
