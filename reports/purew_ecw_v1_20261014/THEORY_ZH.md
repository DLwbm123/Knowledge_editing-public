# ECW统一理论与量化残差界

ECW保持直接原生W写入、当前病例学习和零新增推理参数。固定写入空间中完整累计回答与医学证据函数的采样曲率，实际FP16偏离单独进入残差界。该方法的保护结论限于冻结采样线性代理；不存在对任意病例序列不冻结或临床零损伤的保证。前100项六臂设计未启动；当前执行为GPU4独立短程实现验证。后续用户条件授权：结果满意后，可冻结并执行原146最终ECW方法及历史比较；机械完成不足以满足条件，见CONDITIONAL_146_AUTHORIZATION.json。

## 一 固定写入坐标与数据边界

设可编辑层参数为向量theta。用原Base的layer21 down_proj建立源无关基底，P(x)=Uo mat(x) Ui^T，Uo和Ui各32列、FP64 QR正交化，坐标维度m=1024。具体可实现版本采用随机SVD q32、niter2、seed20260912，且不改变训练RNG；不声称获得精确前32谱子空间。P^T P=I在精确实数下成立，有限精度Gram误差准入为1e-10。

实际native参数分解为 theta_t=theta_0+P x_t+e_t。更新后x增量为P^T(theta_new-theta_old)，e增量为实际位移减P投影，不能把FP32 proposal当作实际native位移。基底、x、e、累计协方差和参数矩属于训练状态；推理只加载原结构和修改后的W。固定空间可遗漏编辑需要的方向，失败不触发增秩、换层或重选基底。

学习器只读当前源Native、四原FIT以及既有合格框视图。病例边界删除当前图像、QA、批次特征、教师概率和逐源VJP；不保存旧病例或逐源索引。CPU生产者/独立评价器持原登记不是学习器重放库。累计参数统计可能泄露训练信息，不是隐私或对抗沙箱保证。

## 二 统一功能散度

回答块A_i采用出生点当前native模型的token分布，定义五个源问答的加权KL，Native=.5、每FIT=.125。令p=softmax(z)，C=diag(p)-pp^T，则在出生点的精确二阶导为J^T C J；KL的一阶项为零，不需模型对源标签已最优。它是局部二阶性质，真实非线性KL并不等于该二次式。

医学块只在既有12项合格脑MRI粗框启用。源答案平均logprob差为d=s(full)-s(removed)，pi=sigmoid(d)，温度1。保护出生点Bernoulli(pi)与以后同函数的KL，局部曲率为pi(1-pi) grad(d) grad(d)^T。四独立高斯探针按五源权重合并，形成R_E；回答探针形成R_A。两个块独立产生，明确保留有限采样交叉项与真实期望的区别。Fisher协方差仅在期望中无偏，不是全曲率上界。

共同保护目标为D_i=D_i^A+a_i D_i^E，证据块权重1。没有区域资格、全局影像和KG题关闭区域块但保留原编辑。当前范围目标为删除视图编辑前分布对编辑后分布的KL，按Native/cyclic FIT各.5加入当前CE，权重1。只使用当前源答案前缀与当前临时教师，不新增健康/相反标签。删除视图采用原匹配粗框并集、灰128、padding0，存在分布外伪影；不是临床反事实。

来源审查100项=局部45/全局5/KG或混合50，仅12个脑MRI粗框合格，mask0、无关背景0；审查不是独立医生核实。原源含公共validation/test来源，不能声称官方train-only或独立患者测试。

## 三 多出生点累计量

对每源冻结的有限采样矩阵R_i，令A_i=R_i P。在固定坐标中

Phi_t(x)=.5 sum_i ||A_i(x-x_i)||^2。

在当前点表示为q+v^T delta+.5 delta^T H delta，其中H=sum_i A_i^T A_i。实际坐标位移delta后：q+=v^T delta+.5 delta^T H delta，v+=H delta。新增源在其出生点能量/一阶项零，仅给H增加当前A_i^T A_i，不重置旧q/v。

完整1024维协方差存储O(m^2)、更新O(rows*m^2)，与编辑数无关，不是全参数曲率精确化。Q0匹配消融保留rank4 FD及当前点PSD补全；Q1起去除FD。所有臂记录相同native残差界。不存在把旧风险重锚或按编辑数摊薄预算的步骤。

## 四 原生残差矩与界的证明

实际历史线性响应为R_i[P(x-x_i)+(e-e_i)]。定义kappa_i=||R_i||_F，

a=sum_i kappa_i^2，b=sum_i kappa_i^2 e_i，h=sum_i kappa_i^2 ||e_i||^2。

T(e)=a||e||^2-2 b^T e+h=sum_i kappa_i^2||e-e_i||^2。

实现采用加权Welford避免大数相减：保存mass=a、mean=mu=b/a、scatter=h-||b||^2/a；T(e)=scatter+a||e-mu||^2。新质量w出生于当前e时，delta=e-mu，mu_new=mu+w/(a+w)*delta，scatter_new=scatter+a*w/(a+w)*||delta||^2。mean是累计参数向量，不是病例特征或逐例锚点。与三原始矩在精确实数下等价。

将所有R_i响应纵向堆叠，由算子范数<=Frobenius范数和三角不等式：

Phi_native(theta) <= B(theta)=.5[ sqrt(2 Phi_t(x))+sqrt(T(e)) ]^2。

这是对冻结线性采样矩阵的确定性界。有限精度、采样与Taylor误差另列；B不界定真实临床错误率。q/v与残差出生矩一起保留才能处理不同出生点；只保留H、或每例重置残差原点不成立。

## 五 过度保守性审计

该界仍有两个松弛：把每个R_i在当前残差方向上的作用换成Frobenius范数，以及把回答坐标响应与残差响应的夹角换成同向。

严格反例：P只写W00，旧R只观察W11，实际残差只改变W22=.2。真实旧采样风险为0；kappa=1，T=.04，B=.02，大于epsilon=.01。真实风险为0时界/真实值比无有限统一上界。因此“已移除FD全方向补全”不能推出“native residual界不会冻结”。

加权出生点净残差矩满足T_net<=sum_i kappa_i^2(累计残差路径长度_t-路径长度_i)^2，优于把所有舍入幅度同向累加，但并不修复上述盲方向反例。CPU随机合成检查的最大有限比为约12.13，只是24个合成状态上的数值，不是真实VLM校准。

冻结决定：保留B<=.01硬约束，GPU4按原界测真实拒绝原因；不降低kappa、不忽略tail、不增加epsilon、不回滚成本。若它阻塞学习，应给出算法失败/未就绪结论，而不是称作工程异常或安全证明。固定大小方向性残差近似需要新的误差分析与另行修订，不在本轮偷偷加入。

## 六 白化与可实现更新

统计N累计当前源参数梯度在坐标中的外积，包括拒绝槽；首源五个训练输入各做一次仅统计初始化，无额外W更新。HA/HE/N分别按trace归一化后平均，完整特征分解并将度量特征值夹到[.1,10]。空统计使用I。Gamma在每个源内固定，并与谱缓存进入恢复状态；不得中途恢复时用已增长N重新生成Gamma。

求解min_delta g_c^T delta+delta^T Gamma delta/(2 eta)，s.t. q+v^T delta+.5delta^T H delta<=coordinate_cap。eta=.01、原梯度cap5。当前T给出coordinate_cap=.5[max(0,sqrt(2epsilon)-sqrt(T))]^2；它仅提供候选，不能预知下一次舍入，因此actual B必再次检查。

KKT候选delta(lambda)=-eta (Gamma+eta lambda H)^-1(g_c+lambda v)。源边界对白化H做特征分解，至多24次扩乘子及总48次根搜索；解按.05坐标范数上限截短，仍做actual风险核验。Q0/Q1使用I，Q2起使用Gamma。白化改变近端度量，而不是坐标变换后全部抵消。

候选来自FP32 master+Pdelta、舍入为原生FP16。最多8次减半。Q3起使用当前实际native模型的源目标Armijo检查，sigma=1e-4、实际g_full^T native_delta<0；CE与scope KL用FP64归约避免FP32损失相等导致伪拒绝。模型本身仍FP16，梯度/master仍FP32。非递减/零native/风险拒绝均丢弃候选master和native并消耗槽，不额外加步。

理想采样模型中若N0张成ker(H)，则v在range(H)，沿N0移动不改累计风险。若N0^T g非零、且有舍入余量，存在负近端目标的更新；其改善至少与eta*||N0^Tg||^2/(2 lambda_max(Gamma))同阶。这是条件性实数代理结论，不是任意序列、不受采样盲区影响的无冻结保证。basis表达不足、真曲率覆盖、残差界和native格点均可能阻塞。

## 七 借鉴与候选创新边界

[CrispEdit](https://arxiv.org/html/2602.15823v1)提供Bregman/Gauss-Newton约束思路；ECW不用其K-FAC投影器，也不继承其完整能力保证。[M-ORE](https://arxiv.org/html/2605.20273v1)提供固定坐标与在线递推思路；本实现不用其视觉projector/文本模态分层或Sherman-Morrison编辑器。[StableEdit](https://arxiv.org/html/2605.11836v1)提供统计初始化与完整白化的启发；ECW统计为参数敏感度，不保存旧病例特征，未证明其论文的稳定回路在此成立。

候选贡献是证据函数的多出生点累计风险、native残差出生矩以及保护/可塑性联合分析。Fisher、SVD、白化、二次配方、KKT及Cauchy界都是已有工具，不能各自作为原创。尚无穷尽新颖性审查、真实长期效果、临床或隐私证明。
