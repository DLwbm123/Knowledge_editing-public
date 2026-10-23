# V91：来源可辨识特征与谱初始化的三臂完整实验

V89固定SGB迁移失败，结果保留。其训练侧审查发现：来源prompt key只有8个，中心化后rank≤7；实现保留8个SVD向量，第8维尺度落至1e-4 floor，末维head权重非零。该数值/可辨识性风险不依赖任何T2G输入或答案。它不是已经证明的泛化损伤原因。本独立运行只检验这个来源侧实现假设，不能以改名恢复V89，不能回载已退休的5份V89状态。

来源仍为原8编辑/seed/native+4FIT共40，冻结Base及21down四核TT分支；新运行冷启动。所有方向、缩放与谱回归只读训练来源，无评价答案、seed/系数搜索。先前DEV结果导致审查，但本次有效维数阈值1e-6由来源矩阵数值rank定义，不按测试性能选择；新结果仍是已曝光8编辑DEV，非独立确认。

与V89共享的机制：40来源共同warm一步canonical RMS，C首步平均实际算子步长；8个不同编辑native梯度中心化PCA前2方向及均值建立3个TT原型，有符号合成branch0+Σc_j(branch_j−固定warm)，c=2tanh(head/2)。不新增LoRA或训练Pure-W；这是TT谱分支迁移，不是SGB原密集矩阵/EARA完整复现。

来源prompt key中心化SVD只保留σ>1e-6σmax，实际rank r必须1≤r≤7，随后仅在这些方向按来源std缩放（最小1e-4）；不保留未辨识方向，不读取评测prompt决定r。三臂同3×7168可训练TT、同r→2 head（2r+2参数）、同固定7168参考与4次TT算子：

- SHARED：输入恒定特征，head权重/偏置全0初始化；允许常量有符号系数。
- IDENT_ZERO：真实可辨识来源特征，head同样全0初始化。主要IDENT_ZERO−SHARED，隔离输入依赖路由，不混入谱回归初始化变化。
- IDENT_LS：同IDENT_ZERO真实特征，head权重用原来源梯度谱坐标最小二乘初始化。次要IDENT_LS−IDENT_ZERO，隔离初始化影响。

同标称参数及算子流程不意味着同路由函数能力；常量路由不能随问题变化。所有source-only特征/原型在三臂训练前冻结，不随评测改动。每臂1280全局更新，固定8编辑轮转、每编辑160、native .5/cyclicFIT .5、Adam lr1e-4/clip1，不早停/改步数。3终态全freeze后当次Base+三臂各155，共620自由消费者。机械同前缀rtol1e-5/atol5e-4、save/load精确、原route、完整全文隐藏臂评分、原51保护分母不变；原Base支持0则NA，新Base支持另报，不宣称临床保护。完整来源事实、关系、限定、时态与无源外断言标准不降低。

主要继续门槛IDENT_ZERO相对SHARED T2G净增≥3/32、跨≥2编辑且native/FIT/T1G逐编辑无损；次要初始化对比同标准。与已结束V89仅历史开发参照，不冒称直接配对只改一个因素。无论失败或成功都保留所有8编辑及全部消费者，不调权重、floor、rank阈值、温度或额外seed追结果。

物理GPU2、源UUID/启动35GiB余量，保留该卡其他作业。独立12h墙钟/7200GPU驻留秒，8GB硬/6GB软/余8GiB、本地200MB；F1e6/non-gen20000/NNB20000/opB20000/update25000/teacher0/gen640（620正式+故障余量，不重采）/Judge0。独立guard与逐调用预算，所有失败续接共原钟/费用；新实际模型异常STOP保存必要状态，不隐式重训。源码与精确部署入口先冻结，启动后核验中性argv/PID/startticks/GPU，不能只认文件名。

全部末消费者和必要状态检查后依明确文件清单退休本轮原型/终态/resume；原V89与历史Base/Fisher/C均不动。公开源码/匿名聚合，原来源QA/key/梯度/图像/tokens私有，GitHub只经有效7897代理。其他DPC/SCRO/Rank-JB及CanonEdit独立继续，不能让本来源修复替代尚未完成的其他路线。
