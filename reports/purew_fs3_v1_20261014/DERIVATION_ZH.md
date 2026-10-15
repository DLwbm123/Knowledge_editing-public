# F/S前三项机制对照推导

只用原前三项native及四个FIT，不增加病例、问题、区域目标、层或seed。F/S各160槽/源，新正式槽总960；M前三项480槽与输出复用，额外读取原M既有末端W进行登记评价，不重训。

对当前源第t个回答token，p_t=softmax(z_t)，C_t=diag(p_t)-p_t p_t^T。令xi标准高斯、u=sqrt(p)*xi-p*(sqrt(p)^T xi)，则E[u u^T]=C，E[u]=0。每个源函数块以sqrt(source_weight/T)/2缩放；四个独立VJP行R满足E[R^T R]=sum(weight/T) J^T C J。这里p和Jacobian都只在当前源学完时计算，之后不保存输入、输出分布或病例索引。统计不自动具备隐私保证。

在源i出生点W_i，采样局部目标为Phi_i(W)=.5||R_i vec(W-W_i)||^2。这是token平均KL的局部实数二次近似，非真实KL全局上界。累计Phi=sum_i Phi_i，在当前W表示为q+v^T delta+.5 delta^T H delta；新增当前R_i只给H加R_i^T R_i，q/v不重置，因新源在当前点能量和一阶项为零。接受实际FP16位移d后，q <- q+v^T d+.5 d^T H d、v <- v+Hd。它保留不同出生中心的累计影响，不把所有历史重锚到最新W。

F：沿用rank4 FD和在当前点中心化的PSD补全。对stack C，截断损失0<=C^T C-B_new^T B_new<=tau I（精确实数下），因此将tau加到rho保留先前q/v，新增的是当前点为中心的非负二次包络。误差包络亦保留。这只能界定所采样的二次代理，采样、Taylor和FP16网络shadow Jacobian误差仍未知。

S：前三项最多12个采样方向。每次对全部保留统计与当前4行stack做全维正交旋转，B_new=U^T stack，不截断。在精确实数下B_new^T B_new=stack^T stack，因此H累计精确保留所有有限采样曲率，不需要全方向rho I。旋转后的行没有病例索引；原逐源VJP行退役。实现以FP64小Gram eigensolve、分块CPU旋转，检查U^T U偏差<1e-12，误差属于数值近似。S仅为固定前三项完整子空间诊断；它不采用K-FAC、不宣称100项可扩展。这样S−F聚焦FD补全，避免同时引入另一种结构化近似。

联合更新仍解min g^T d+||d||²/(2 eta)，s.t. q+v^T d+.5 d^T H d<=epsilon。KKT给出d(lambda)=-eta(I+eta lambda H)^-1(g+lambda v)，沿用Woodbury和最多48次一维根搜索。部署时以实际native FP16候选位移检验risk<=epsilon及g^T d+||d||²/(2eta)<=0，最多8次回退；拒绝/零更新均丢弃候选master/native并计槽，不补满。

新epsilon=.01 nats，表示累计先前源token平均局部采样KL能量，完全前瞻冻结，不由评价正确率选择。原M的.01是未按输出分布加权的raw-logit能量，两者单位和尺度不同，F−M不能叫等保护预算或单纯幅度控制。F/S的metric、probe数、预算相同；有限轨迹上的采样状态仍会随模型路径改变。

每个native候选记录changed_elements/native_norm/risk/proximal，以及ZERO_NATIVE、RISK、PROXIMAL的独立失败标记；标记可重叠。原M旧合并日志不追溯伪造细分原因。不为重建原因追加旧模型调用。

CPU验证覆盖Fisher协方差恒等式、S完整采样累计能量/出生中心、F的PSD补全、KKT dense oracle、实际FP16回退与三种诊断标记、master/native/RNG和源时钟完整续步、损坏时钟拒绝、当前源I/O边界，以及483消费者屏障/共享missing/未知保护不通过。真实VLM分别验证Base3、原W真实梯度、上游/参数结构不变、真实Fisher VJP。通过后才正式训练。

保护检验使用原前100冻结的34个Base正确T2L和2个T1L，只进入独立评价器；任一错误或未知不通过，修复/宏平均不能抵销。新前三项T0/T1G/T2G采用原Base资格及active-target规则；输出一旦冻结后才一次评分。K没有同轨迹前三项结果，记NA，不重建旧权重或拿前100冒充匹配对照。
