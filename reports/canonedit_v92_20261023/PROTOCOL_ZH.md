# V92：CanonEdit来源模型输出替换与TT部署协议

本协议在任何V92模型调用前冻结。用途是检验“来源可写的隐藏目标，经canonical最小行为代价投影，再压入TT”能否改善MedTRACE；不是论文对比方法的调参，不是旧512-root CE-QP改名。

## 数据与方法边界

固定原8开发编辑、原seed、native+4FIT、原155消费者和原路由；T2G只用于冻结后评测。每个请求的old是已冻结DPC V87 Base在该native来源问题上的完整原始回答，在该请求所有FIT共用；new为原来源参考。old明确为模型行为替换参照，不能当旧临床事实、真实反事实或CW/CF配对。只读取来源行匹配的Base输出，不能用其他评价答案设计目标。

完整序列含EOS，margin=sum logp(new)-sum logp(old)，不做长度平均；按实际训练标签长度还原compute_loss的token均值。若初始已满足margin，不强迫制造缺口；零写入也必须进入全部自由回答评测。概率margin满足与greedy答对分开报告。

采用第21层down投影、原assistant predictor token scope及原输入RMS约定，迁移为一个固定的双rank-one chart。每个basis从冻结Base、所有模型权重不训练，临时仅在最后prompt predictor位置注入4096维向量，Adam lr2、至少4至最多8步、norm40：rewrite-dominant的4[.25-native_margin]+²+.5 meanFIT[0-margin]+²+.002||r||²；balanced为native和各FIT等权hinge²+.002||r||²。无来源外医疗知识。native输入key按原RMS归一化，与两个残差配对形成E=r keyᵀ/(||key||²+1e-8)。由于部署使用原RMS输入和assistant scope，这是显式函数迁移，非原全MLP持久矩阵、多层atlas或sequential复现。

## 实际臂与控制

1. CE：原7168 TT冷初始化、160步native/FIT .5/.5 CE。
2. RO：rewrite-dominant第一个rank-one变化，系数(1,0)，不canonical投影。它与CANON共享来源目标/候选basis，用于隔离canonical选择。
3. CANON：两个4096维残差及固定14336维key形成双basis，优化2个组合系数；不能称7168参数TT。对5个target全序列margin求Jacobian A；对按原owner顺序循环的后5个不同编辑native来源、其冻结Base回答全序列分数求C。后者只保护模型行为，不宣称正确医学能力。M为dense rank-one变化Frobenius Gram除以平均对角，加1e-8 I；B=CᵀC/5+.04M+1e-10 I。CPU float64活动集枚举精确求解2维最小二次代价Az≥有符号目标缺口（已满足约束保留真实余量）。限制修正及总zᵀMz≤36；最多4次重新线性化，允许负系数，不选择seed/半径。
4. CANON_TT：将通过的canonical完整残差，在固定40来源中的本请求5个来源predictor激活上以归一化Frobenius误差内化为7168 TT，Adam .01、256步；不再加CE掩盖投影效应。必须再次通过同一完整非线性目标/行为预算；不通过则TT零算子，保留该失败编辑和全部消费者。故它是明确部署压缩检查，容量与RO/CANON不相等，不用混合容量比较声称公平单7168方法胜利。

CANON完整非线性准入：native margin≥.25、4FIT各≥.10（数值1e-6容差），5保护分数相对原Base的平均绝对变化≤1.5、平均平方变化≤2.25。采用原文允许的score-change代价；不冒称计算了全词表保护KL。每次候选临时完整前向验证；无合格则零写入并记录失败，不放宽门槛/用CE补救。TT部署重复同一门槛。RO原样进入评测，不按结果删除。save/load各张量精确；每臂非零/零状态同前缀完整/缓存检查按rtol1e-5/atol5e-4执行。

## 并行、指标和停止

GPU0/1按原owner交错各4编辑；每worker均只读相同40来源/new和40old缓存，写不同owner。全部32终态冻结后两卡并行生成Base+4臂=775自由回答；预留至800仅失败成本，无重采。两个worker共享原始12h时钟、8GB硬/6GB软新增存储、8GiB余量，GPU进程总14400秒，F1e6/non-generation40000/NNback40000/opback20000/update15000/teacher0/Judge0，每worker上限各半。独立guard硬控、UUID/45GiB加载准入、中性argv；不动他人作业，所有失败计费、异常STOP，不热修/隐式重训。ROOT旧RUN_ROOT仅只读导入，V81_ROOT为全部新写根。

主要CANON−RO，部署CANON_TT−CE，并报告CANON_TT−CANON压缩损失。完整原155消费者/臂、每编辑胜负和所有触顶/EOS；先隐藏新臂全文审核并冻结再解码，可用完整精确绑定继承既有分数。继续需要T2G净增至少3/32跨至少2编辑，native/FIT/T1G逐编辑无损；不能仅凭margin/局部QP宣称准确率。原保护51分母全部保留，Base正确支持0则损伤NA。8个已曝光DEV编辑不构成独立/临床确认。

原文单层Riesz/有限atlas公式是依据，双basis+RMS+TT压缩是本迁移，不声称忠实全文复现。所有结果包括零缺口、无可行写入和压缩失败必须报告。全部模型消费者和必要验证后按具体清单退休新final/resume，保留私有回答/tokens/标签/绑定/配置与费用；历史Base/Fisher/C不动。公开源码、协议和脱敏聚合经7897代理推送并验证匿名访问。

### 冻结实现标签澄清

部署rank-one沿用输入RMS形式，但源码中其epsilon为1e-8；原TT的epsilon为1e-6。不能称两者逐位同一归一化。RO与CANON共用同一rank-one实现，TT单独经过完整非线性验收；本注仅披露已冻结源码数值，不改变任何运行代码、公式门槛或数据。
