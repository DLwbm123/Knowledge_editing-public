# TT / SVD H8 八臂独立冻结计划

用户明确要求TT、SVD分别FREE与非FREE，并确认保留有H/无H八臂，以及SVD所有因子可训练但左右奇异向量保持正交。

仅原H8八编辑single：TTKEEP/TTFREE/SVDKEEP/SVDFREE各H0/H1，共64条320步续训。TT和SVD分别独立零残差初始化，均按事先固定native步数N及A2 80/W0 320训练，再各自克隆四分支。N由已完成CP轮训练早停记录固定，绝不按本轮评价选择。两路线从相同零函数开始，但预训练后的函数可能不同；仅每条路线内KEEP/FREE必须同起始函数。TT/SVD不复制CP学习权重。

TT张量维度[64,64,112,128]、TT ranks[1,4,4,4,1]、3584参数。首输出core零，其余core用固定seed右正交初始化。四core保持可训练；沿输出/输入分界展开得到rank4 A/B。SVD U[4096,4]/V[14336,4]/s[4]共73732参数，初始s为0；每次forward通过正对角QR生成正交左右向量，所有raw因子和s可训练，s符号/排列不限制。SVD_KEEP仍覆盖rank4矩阵函数族，不能声称比FREE进一步压缩或容量更小。FREE为73728参数，承接相同更新函数后原规范自由续训。

Base、R0、原native/四fit/U/H、masks、生成、H1权重、优化器lr组/clip/阶段重置不改。TT无后重标定；SVD forward正交；FREE沿用原归一化。比较包含优化坐标差异，不是纯容量隔离或等算力比较。

先CPU零函数/确定性/非零展开残差与输入梯度/全部因子梯度/真实save-load/优化器检查及评分完整键检查，再31/35八臂真实机械。16机械产物纳入64，不重复训练。真实机械须检查零函数Base一致、warm KEEP/FREE native token parity、梯度与残差、诊断ON/OFF更新、save/load/resume、Base/hook/RNG不污染。机械失败保留原成本后技术修复，科学阴性不重试。

两个主对比：无H下TTFREE−TTKEEP、SVDFREE−SVDKEEP的H8 T2G macro。另报有H对应差、TT/SVD同形态对比、四条H效应与两条H×FREE交互。全部T0/T1G/T2G/T1L/T2L与Hfit、原mask完整分母、shared-key缺失精确界、10000固定seed配对edit/source bootstrap。区间描述性，无多重比较显著性或选赢家宣称；不自动切换主线。Hfit仅训练诊断；H_eval NA、患者UNKNOWN、locality空分母NA、已暴露小样本/单seed限制保留。无全146库扩展。

Qwen3-32B-AWQ沿用完整父协议32并发，三已完成轮的完全相同键继承有效或永久missing及原attempt证据，不混Astra/SOL或重判。新attempt最多2000，连续3transport失败停止，其他异常立即停。累计12GPUh、首次24墙钟h、生成权重2GiB+8GiB余量，GPU5/6；所有失败计费且不重置。有限后台controller完成训练→评分→报告，旧小时监测不恢复。结束后审阅并代理公开新draft PR；仅最后消费者完成的本轮owned清单权重清理。

方法参考：[Tensor-Train原论文](https://epubs.siam.org/doi/10.1137/090752286)、[PyTorch QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)。这里没有对Base完整矩阵做截断SVD；SVD路线是可训练谱参数化的rank4更新。
