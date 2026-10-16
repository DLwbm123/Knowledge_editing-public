# HE：HM＋直接证据间隔目标，冻结准备

2026-10-16用户授权协议、代码与必要检查，明确先不启动实验。HE为独立新臂；旧HM/HU/HR等源码、结果与全部失败成本保持不变。执行、安装、GPU分配、真实VLM、Judge、评分消费者、等待器、监测均未授权。后续启动需要新的明确授权、实际硬件绑定及准入，不因本准备文档自动执行。

## 唯一科学变化

从原始Base、原首源和seed出发，不从HM或HR最终权重继续。保持HM五个full/removed视图、token化、两概念划分、层、FP16 native/FP32 master/FP64统计、Base16与removed4保护统计、优化器和candidate时钟。

第s槽使用视图0与k=1+(s−1)%4，各权重1/2。H_v沿用HM：答案token平均ReLU(logit(a*)−logit(a_full))，a*=max(p_Base_full(y),0.95)。

m_vc=mean_{t属于概念c}[log p_W(y_t|full,y_<t)−log p_W(y_t|removed,y_<t)]。
E_v=(ReLU(ln(1.01)−m_v0)+ReLU(ln(1.01)−m_v1))/2。
L_s=(H_0+E_0)/2+(H_k+E_k)/2。

hinge系数1，evidence系数1；间隔目标ln(1.01)=0.009950330853168092自然对数单位。先每概念token平均，再两个概念等权，避免长概念支配。两个分支都求梯度，不detach removed，不加概率floor；ReLU边界次梯度为0。候选Armijo检查同一L_s，不额外添加CE候选门或CE损失。removed全词表KL、条件KL、prefix KL系数均为0；实际removed KL仅诊断，原有限Fisher代理约束仍保留，HARD_KL_NOT_ENFORCED。

系数1是两个平均log尺度缺口的等权默认；1.01是预先选定的1%几何平均概率比目标，不由测试题、旧最小间隔或梯度大小拟合，不声称最优、临床阈值或精度误差保证。禁止系数/间隔扫描、gradient balancing、按已见测试表现换源或选seed。

## 数据边界与解释

学习只读原当前源、原五视图、源答案两概念及既有参数统计，不读T0/T1G/T2G/T1L/T2L评分或测试标签。原removed变换与概念映射不根据测试题修订；不新增临床范围数据，不实现UltraEdit。原首源与已有面板属于开发证据，不能作独立确认集。

每槽及初末记录十个full_logp、removed_logp、margin、训练缺口和目标达到数。最终报告展示每概念Δfull、Δremoved、Δmargin；正间隔可能来自降低removed置信度，不能仅凭它宣称证据利用改善。若剩余输入本可支持答案，强制相对间隔也可能不合理；本轮仅检验既定资格定义，不证明证据因果必要性或医学正确性。

## 冻结通过门

全部AND：20槽完成；完整五视图FIT CE严格<0.1；十个证据间隔有限且严格>0；T0插入和终末均正确、T1G全部正确；T2G按原配对下界规则相对HM及SH00不下降；原Base正确36题逐题零损伤；评分完整、无未知。ln(1.01)是训练目标，不替代原严格正验收门。缺分/早停记NA或失败，不放宽门、不补跑。

仅比较HE−HM为主要对照，HE−SH00保留原非退化门；六个已完成历史臂只读复用87×6=522消费者，新HE最多87，共609。HR原记录完整保留，不重跑；它不加入此次主比较或609汇总，避免扩展范围。单源不生成总体置信区间，不宣称连续编辑能力。

## 预算与停止

单臂首源20槽；GPU进程累计2小时，墙钟24小时；native写入21次含1次准入临时写；backward300；forward15000含vision；generation92含Base3、正式87、重载2；新Judge87，单payload最多一次；candidate160；active-set0。96GiB自有矩阵上限、96GiB可用主存、至少8GiB磁盘余量、45GiB空闲GPU准入及45GiB自有GPU上限，沿用原约定。

所有教师构建、诊断、失败、驻留与保存计入未来一次激活的预算；拒绝也消耗槽。不借旧预算、不重置时钟、硬中断重放额度0。未启动，因此activation/deadline均为null。连续五次numeric-only且高CE拒绝等原停止规则保留。机械失败留现场，不自动重装/重启。无参数搜索、额外槽、seed扩展、下一臂或146自动扩展。

## 已完成的工程检查

PRODUCTION_CPU_VALIDATION.json记录真实生产函数在合成CPU张量上的检查：独立损失/梯度oracle、两侧有限差分、概念长度归一化、边界/异常标签与span拒绝、native候选与异常回滚、状态recipe/时钟、实际learner两条20槽终态路径、预算、评分AND门、缺分和不启动入口。评分测试只复用已有只读payload或一次性沙箱中的合成分数，不请求新Judge，不据结果改变目标或系数。

初次CPU兼容性测试因日志字段scope_loss→evidence_loss命名差异失败；在测试中统一字段名后数值/状态对照通过。失败尝试回执保留，不是实验失败或GPU运行。CPU通过不等于真实VLM准入；后者仍NOT_RUN。
