# HC：两阶段拟合约束证据优化，冻结协议

2026-10-16用户明确授权准备、必要检查、启动本轮、每半小时监测并分析结果。仅本轮，不调参、重跑或自动新臂/146。HE、HM及其他历史结果与所有失败成本保持不变。独立原Base首源20槽，不续接HE终点。

## 单一政策变化

主要对照HC−HE，辅助HC−HM及HC−SH00。保留原层、native FP16/master FP32/统计FP64、原源/seed/五视图/概念、Base16/removed4与优化器。H沿用原答案odds hinge（a*=max(Base概率,0.95)）；E沿用两概念各自token平均后等权的ReLU(ln(1.01)−margin)，full/removed均求梯度。E权重1，间隔ln(1.01)，不加removed/conditional/prefix KL、不换更新器。

FIT阶段：每槽视图0和k=1+(slot−1)%4各1/2，优化H+E。每次正式接受后以五视图权重(.5,.125,.125,.125,.125)计算完整CE。首次严格CE<0.1时记录first_fit_slot，从下一槽切换EVIDENCE，不返回FIT。初始原Base若已达标则first_fit_slot=0，从槽1进入EVIDENCE；真实准入临时写始终使用H+E并回滚，不触发正式阶段。

EVIDENCE阶段：相同轮换视图仅优化E，H权重0。每个实际FP16候选必须满足原代理风险门、E目标Armijo下降且五视图完整CE严格<0.1；等于0.1或非有限均不接受。候选拒绝恢复native，master/统计/阶段不变，消耗槽。阶段及first_fit_slot写入checkpoint，recipe=CE_GUARD_01_TWO_PHASE_EVIDENCE_LOG101_WEIGHT1_V1；旧状态不可混用。

若20槽始终未达标，全程FIT；若第20槽才达标，则记录切换但EVIDENCE槽数为0。两者都不追加槽、不重置时钟。阈值来自原冻结CE门，不从已见测试或HE的第15槽选择。除阶段政策外不新增在线权重调节、数据、seed、margin容差或保护目标。

## 日志与成本

每槽记录阶段前后、完整CE、十个概念full/removed logp和margin。每个候选，包括风险已拒绝的候选，也记录相同源内标量、完整CE、拒绝原因及E的实际原生更新方向导数。记录未缩放E梯度范数，以及沿free方向（−0.01×原clip后的目标梯度，再用原0.05范数上限）、solver投影/retraction/cap后方向、实际candidateFP16−oldFP16的方向导数。导数比在分母0时记null，不能当通用保留率保证。不计算外部测试梯度。

FIT每槽及准入至多多一次E backward，EVIDENCE复用未裁剪E目标梯度。所有新增forward/backward、诊断、准入、失败、GPU驻留与保存计入原预算。导数和日志不参与动态调参。

## 原预算与AND门

首源20槽、2GPU进程小时、24墙钟小时；native写入21含1次准入临时；backward300；forward15000含vision；generation92含Base3+正式87+重载2；新Judge87且每payload最多一次；candidate160；active-set0。矩阵96GiB、主存可用96GiB、GPU空闲/自有45GiB等原限制不变。失败成本不清零，零硬中断重放，不自动重启。

全部AND：完整20槽、完整CE<0.1、10个有限严格正间隔、T0插入/终末正确、T1G全正确、T2G按原配对下界规则不劣于HM/SH00、Base正确36题零个体损伤、评分完整。ln(1.01)仍只是训练目标，不替代验收。新增HC87消费者，加七个只读历史臂609，总696；未知/缺分保持原规则，不补Judge，不选择成功子集。

## 当前工程证据与边界

生产CPU检查PASS：真实learner合成20槽路径、初始/首槽/不切换/末槽切换、严格CE候选门及回滚、阶段checkpoint、双侧梯度、投影方向和候选日志、评分696和原AND门、预算/未授权入口。三个CPU失败回执保留：缺少新增诊断所需的合成model；原高置信夹具未接受FIT更新，未触发预期首槽切换（两次含诊断复核）。仅修正测试夹具，不改协议系数。CPU检查不等于真实VLM通过。

本轮仍是单源开发对照，不证明医学因果、总体或连续编辑效果。CE候选约束不是外部保护保证；若证据/保护门失败，报告实际方向与拒绝日志后停止，不自动改方向或增系数。
