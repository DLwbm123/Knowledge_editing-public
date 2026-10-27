# MED-RA V134：用户确认后的冻结执行方案

用户在2026-10-27明确选择“采用修正方案，重建后继续”：允许按原冻结实现重建MED曲率，将候选风险分母固定在raw更新范数，source保底为每条native/FIT保留原MED非负gain的90%；原single筛选门不变，监测保持关闭。本次取代阶段0的等待状态，不改写阶段0发现或V133历史。

## 唯一方法变化

原146份V133 MED的160步终态全部保留，可直接只读复用为raw更新，核对实际state hash、原seed、R0、TT88/7168、layer21与source配置；不为了得到相同raw更新重复训练。原31医学锚点、30原FULL fallback、两个类别各两个别名、FULL_CTX trace标定、C端点λ、cap、去反向、native/FIT权重、mask、tokenizer、prompt、精度与decoder均保持。只在每个独立expert的最终有限TT算子上做统一缩放，缩放G1实现，其他三个core不变。

原曲率缓存已经清理。本轮按V133构造代码重建实际mask内的低秩方向；不训练GEN/FULL_CTX对照，不增加候选。核对原teacher绑定、31项trace/秩/类概率、原有限差分mask检查；缺旧tensor时不宣称逐元素或逐字节复现。

以MED实际61锚点operator之和计算能量与trace；逐anchor求trace后相加，不把不同锚点误当成同一个梯度方向。对146个raw专家计算

`q_i = R_MED(w_i) / (trace(F_MED) * ||w_i||_F² + 1e-20)`。

q0为所有146个q_i的75%分位，float64线性插值。固定grid `[1.10,1.00,0.90,0.75,0.50]`。候选风险为

`r_i(alpha) = R_MED(alpha*w_i) / (trace(F_MED) * ||w_i||_F² + 1e-20)`，应为`alpha²*q_i`；实际FP32组合算子校验相对容差3e-5。

在完整训练native+FIT4各条输入上计算Base、raw和候选CE。每条的原非负gain为`max(CE_Base-CE_raw,0)`；候选须满足`CE_candidate <= CE_Base - .9*gain + 1e-7`。alpha1.10还须逐条不差于raw（同一1e-7数值容差），且raw q低于q0。风险可行阈值为`q0*(1+3e-5)+1e-20`。选择最大同时可行alpha；无可行候选则按Pro原规则使用0.50并记录risk/source violation，不删expert、不静默退回旧MED、不根据正式结果改系数。

q0、alpha和source判断仅使用现有训练/校准输入，不读取T2G/T2L或最终损伤标签。source校准只做前向，不反传优化。记录原λ、raw/final范数、实际能量/trace/q、source五项loss与gain、固定阶梯/violation、实际mask计数、state hash与第一阶医学命题响应代理；不把代理称为临床风险或实际非线性保护loss。

## 依赖与筛选

全部146个RA终态冻结、保存/加载及原R0准入后，先single2956。Base、M13P、当前MED的原回答与同CORE/FULL评分按完整身份复用，旧有效标签不重判；MED-RA执行自己的正式生成，内部仅在实际路由、完整输入/状态/运行/解码/RNG/代码身份一致时复用。逐请求移除hook。

single全部评分终态后固定筛选：RA T2G CORE≥当前MED；RA−M13P的CORE增益≥当前MED−M13P；相对M13P无新增fresh CORE损伤；T2L CORE≥M13P；T0/FIT逐项无新的CORE/FULL退化；所有必要评分有效；实际mask/route/state合同通过。阈值不随结果改变。

失败则`SEQUENTIAL_NOT_RUN_SCREENING_FAIL`并交付完整single，评分缺失则`INCOMPLETE`。仅全部通过才按原R0逐步插入同一146专家、评估146 insertion及bank1/50/100/146的4392消费者，不第二次训练、不扩展笛卡尔积。两阶段完整则`COMPLETE_SINGLE_AND_SEQUENTIAL`。

## 有限资源和恢复

GPU选择当前显存足够的已授权卡，启动时核对UUID和至少45GiB余量，不停止其他作业。每服务器新增严格小于50GB、47GB预停、8GiB磁盘余量。独立资源guard执行预算；不创建聊天心跳或其他定时监测。

复用阶段0预检开始的原时钟，所有重建、校准、生成、评分、等待、恢复共享该起点与24小时截止，不重置。最多1500次NN backward、300次operator pullback、6000次source校准forward、9000000次逻辑forward、40000次vision forward、7600次实际generation尝试；不新增梯度训练更新，原23360更新记继承成本。新Judge尝试至多7494项，四个隔离gpt-6-astra medium worker，一次尝试/唯一新载荷，无自动语义/传输重试或模型fallback。原4754个有效载荷按同合同复用，累计成本另列。

每个曲率anchor、每条source loss、每个已冻结expert、每个正式consumer保留完成回执。工程恢复先确认旧PID及start_ticks已退出，沿同一输入/状态/时钟从未完成项继续，所有失败成本保留；不重做已完成专家或正式输出，不放宽科学/资源门。不可恢复数值错误停止。MED-RA全部FINAL、canonical和独立小备份均保留，V133任何权重/输出/分数/账本/报告不动。

完整报告对比Base/M13P/MED/RA及五种规定差值，逐编辑/来源组区间、fresh损伤/修复、同旧owner遗忘、实际路由/复用、alpha/q分布、费用和未知支持。仍缺全146 GEN/FULL_CTX，不宣称医学独有贡献或独立确认。匿名源码/聚合结果通过代理公开，原QA/图像/tokens/权重和主机身份保持私有。
