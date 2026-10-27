# V47：COVER固定前缀视觉敏感性对照

V46的ADAPT相对Base平均TV较低、JS较高，不足以归因于自适应加权。本轮补齐COVER等权编辑对照，复用同四病例、原seed、160步、rank4/7168、原生含EOS CE及同四源；只有聚合恢复V36的torch.stack(grads).mean(0)。不修改ADAPT或选种子。

先Base62精确复现再训练；每病例COVER原始/SHORT token/text/EOS/cap精确复现历史V36后，冻结模型进行相同固定循环图像置换与共同参考前缀测量。两图labels完全相同、已有输入指纹不同、测量no_grad且参数/更新/反向不变。最终Base全部62逐题分布指标须与V46完全一致才能合并既有ADAPT指标。无评估梯度、无新准确率评分、不把donor参考当真值。

主指标等病例TV：ADAPT(V46)−COVER；同时完整报告Base/COVER/ADAPT、COVER−Base、JS自然对数与argmax变化。分上下文、四组、关系角色和病例，不选指标/阈值，不作grounding或医学正确性声明。

640更新、2570预计/2600硬反向、80000前向上限、124自由生成、62分布配对（248额外测量前向）、0候选。两卡各1h总2GPU进程h，原准备起24h；remote2GB硬/1.5GB软/local250MB，独立guard和8GiB余。窗口累计预留27GB，余3GB，不以历史清理抵销。8权重全部消费者和save-load后退休；失败保留最小恢复，无自动重试。医学输出和绑定私有，匿名汇总公开。
