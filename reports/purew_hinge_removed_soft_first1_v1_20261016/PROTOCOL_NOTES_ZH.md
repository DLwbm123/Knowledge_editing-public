# Hinge 加 removed 软保护首源单因素对照

状态：`FROZEN_EXECUTION_AUTHORIZED_PENDING_ADMISSION`。用户2026-10-16后续明确授权源码推送后启动冻结HR及有限完成交付。权威机器协议为本目录 `PROTOCOL.json`；本次授权覆盖安装、GPU分配、模型调用与有限评分发布链；不创建周期监测。独立执行器已实现，生产路径CPU检查通过；存储/GPU和真实VLM准入未运行。

## 唯一实验因素

新增一臂HR，从原始Base开始；复用历史HM作为对照，不从HM终态续跑。每槽仍使用native与循环FIT各0.5：

`L_HR = 0.5[H(full_0)+KL(p0_removed_0||p_removed_0)] + 0.5[H(full_k)+KL(p0_removed_k||p_removed_k)]`，`k=1+(slot−1)%4`。

H为原HM、alpha=0.95、权重1、边界子梯度0的odds hinge。KL是当前源出生时Base的removed完整词表分布到当前分布的forward KL，原答案位置、teacher-forced原前缀，按token平均；权重固定1，复用SH10定义。不是HU的非目标条件分布KL。HR不加入HU条件/prefix项，不加margin损失或实际KL硬拒绝。`HARD_KL_NOT_ENFORCED`保持。

原首源、seed、五视图、removed变换、FP16 native、FP32 master/梯度、FP64统计、单层直接W编辑、Base16/removed4、风险0.01、eta0.01、clip5、proposal cap0.05、Gamma=I、48次dual迭代、8次line search、8候选及Armijo规则全部不变。候选实际损失必须用HR组合目标，拒绝回滚native/master/统计；不允许只改反传而沿用HM的候选损失。

## 比较与判定

主比较HR−HM；同时列示源内真实removed KL、十个证据间隔、完整CE、接受/拒绝。固定36题损伤和4题T2G为必要外部终点。仍须AND通过：完整CE<0.1、T0/T1G全对、十间隔严格正且有限、固定36题逐题零损伤、T2G相对HM及SH00不劣、评分完整。没有通过时不扩源、不补步、不换权重、不自动新臂。

历史SH00/SH10/PM/PU/HM/HU各87消费者只读复用，新增HR 87，完整参考报告共609。旧缺分不得补Judge；新payload每个最多一次尝试，完整绑定匹配才允许评分复用。历史可比性不通过则停止准入并报告，不能自行重跑对照。历史SH10和交互差值仅用于开发解释。

初始与终点新增当前训练视图上hinge/removed-KL梯度范数和余弦，并记录十个概念的full logp、removed logp及差值；只保存标量，计入预算，保存恢复RNG且不改变训练时钟。旧臂未记录的量保留NA。所有逐槽CE/实际KL、代理风险、候选拒绝记录继续保留。禁止评价梯度、以评价错误定向选样或训练。

## 固定预算和准备边界

仅一个新臂、首源20槽、2 GPU进程小时、24小时墙钟；21次native写入含一次临时准入，300 backward、15000 forward含vision、92 generation、87新Judge、160候选、0 active-set检查。所有诊断、准入、失败和保存驻留都计费；无旧预算借用、重置或硬中断重算余量。新增诊断若令生产路径超预算，停止准备并报告，不能静默扩容。

owned矩阵含teacher/原子副本≤96GiB，host available≥96GiB、磁盘保留8GiB、GPU可用45GiB、缓存43GiB、owned GPU45GiB。将来仅用pro5000的合格GPU，当前绑定GPU3，实际分配须通过准入；启动前再次检查。大文件仅`/data/bmw/`。

实现采用HM独立快照及SH10软KL定义，新状态recipe为`ODDS_HINGE_ALPHA_095_REMOVED_SOFT_1_V1`。需核对loss/候选调用者、worker状态与诊断、runtime预算、controller单臂屏障、bootstrap/deploy授权、queue/report/finish单臂及609消费者计数、CPU夹具；这些调用链已完成HR单臂适配与CPU检查。Base3、真实VLM准入仍须在未来授权后完成，且必须先于正式槽。

最小选择是一臂HR加历史HM。暂不做HM重跑或多权重扫描；任何历史兼容性缺口必须显式处理。
