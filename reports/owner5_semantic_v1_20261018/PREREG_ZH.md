# Owner 5 语义保护最小对照：冻结计划

## 授权与范围

用户批准：只做 owner 5 两组，各160步；不扩样本、不调参；结果按语义标准分析、保留必要证据、不自动下一轮。新独立运行，历史运行只读。Control 与 Semantic-Protected 均从相同原 seed 的零专家开始，使用同一个原始模型、M13P、Fisher、coefficient、原C逐步 operator step norm、样本顺序与生成设置。第一步 proposal 必须逐张量一致。

## 数据

原 native + FIT4 + 正构造用于目标编辑；唯一现有负构造用于保护；现有正检查和负检查仅在两组160步冻结后读取。六个候选只是同一负构造的表达，不添加图像/问题。现有历史终点不重用，也不添加旧 NLL 保护组。

## 固定表达和损失

三对依次为：
1. No pleural effusion is present / Pleural effusion is present
2. There is no evidence of pleural effusion / There is evidence of pleural effusion
3. The image does not show pleural effusion / The image shows pleural effusion

每句只计实际内容 token 的平均 NLL；排除 prompt/image/padding 和模板尾空格/EOS。核对原始目标序列与内容 token 严格前缀匹配；保存实际展开 labels 与预测位置。

S = 三对 (NLL_wrong - NLL_correct) 的均值。零专家 Base 的S0事先固定。S0<=0则停止训练准入并报告；不更换候选。原编辑损失 .25native + .25轮换FIT + .5正构造；Protected增加lambda=1.0的ReLU(S0-S)，Control不加。梯度相加后进入原M13P更新，无旧NLL有限回退/缩步保护。

每步先无梯度算六个候选；仅hinge>0时顺序重算六个候选并按(+1/3,-1/3)反向，数学上等于活跃hinge梯度，避免保留六份大图。纯计算自检核对活跃、非活跃及边界梯度。两组都记录每步分数与三对差值，不根据其选择终点。

## 有限执行与证据

最多320更新、1920次神经网络反向、320次operator反向；只有两个160步终点。Base/Control/Protected各9条自由回答，共27次生成；无新外部Judge。有限总forward防护60000、6小时及8GiB剩余空间；这些是失败边界，不是扩展实验许可。失败保留证据，不自动重试、训练续跑或更换配置。

保存每步曲线、Base与两终点六个候选的逐token损失/完整目标位置logits、实际mask、生成原文/token、绑定和成本。两个小型最终专家状态保留用于本次语义审计和可复现性；完成分析后清理冗余active副本，不保留160步轨迹或复制大模型。原始私有材料不公开。

## 语义验收

两组冻结后，对全部九个原问题正常自由回答，不展示候选、不强制Yes/No。用事先已有的参考结论判定正确、错误、不确定/未回答、内部矛盾四类；同义词、大小写、句式及合理解释不影响正确判定。与参考结论冲突的新增内容不计正确。将组别隐藏并随机排列后逐条记录判定理由，再解盲汇总；这是固定规则的助手语义审阅，不声称独立临床专家或外部Judge认证。有歧义明确保留待复核。

分别报告native/FIT、正构造、负构造、正检查、负检查。Base负检查正确且Control受损、Protected仍正确，同时目标编辑与正检查不退步，才支持该最小对照内的保护获益。两组都正确不证明增益；只保构造不保检查无独立泛化；保护成功编辑失败不满足共同目标；分差改善但自由回答受损说明代理不一致。Base原有错误单列。一个owner/极少样本不能形成普遍医学保护结论。

完成后发布代码、匿名指标及报告，保留负结果与异常；不自动下一轮。
