# V76：源答案目标外token条件分布保持

V75保留B全部正确T2G并新增1，但仍损失Base唯一正确T2L；停止最坏视图候选，不搜混合系数。独立假设：学习新目标token时，保持其余token之间的相对概率，能否降低无关输出变化同时保留泛化？约束仅来自合法训练源，不新增保护数据、评价答案或病例标签。不同于V68错误图像上的整个分布锚，也不把Base错误源目标强制保持。

对原源及既定两训练改写全部30绑定，在未编辑Base上teacher-force完整源答案。在各监督预测位置排除参考token，将余下logits重新log-softmax成条件分布，CPU64冻结。目标外token集合覆盖其余词表，EOS预测位置也纳入；不是手工挑保护词。训练每个绑定用原完整答案CE + KL(Base_non_target || edited_non_target)，系数固定1，两个绑定仍原问+交替改写等权平均。KL均值按监督预测位置，排除目标后归一；不把“目标token梯度为0”夸大为参数/其他输入变化为0，也不保证全局locality。

KL全FP64，目标位置在logp与probability中显式屏蔽；要求有限且KL>=-1e-10，仅将该范围内舍入负数钳为0。此阈值运行前固定，不搜索。全零TT根时要求KL<=1e-10。Base概率提取30前向0反传，训练前全30冻结。教师张量实际小于1GB，否则STOP不转存越界；其大文件只远端保存，全部训练及机械消费者完成后按清单退休，保留小描述/成本和来源可重建，不下载到本地。

复用V69全10未训练初始化，原无门控ALL_TOKENS、原欧氏平均梯度/clip1/双组RMS/canonicalize、7168核心、rank4、layer21、lr/beta/eps及160步，生成配置不变。记录每步各CE与KL。不搜索权重/保护词/温度。完整原准入必须通过，失败STOP。

CPU检查causal目标屏蔽、Base KL0、仅改目标logit时KL不变、改非目标时KL>0、目标logit梯度0、条件归一与生产cachefixture。自定义loss安装到runtime.compute_loss后再由PrefixCache包装；value统一走compute_loss，避免V73绕过入口。真实30zero/nonzero/cacheloss-gradient精确/teacherKV原阈值/RNG/saveload，额外每状态cache miss3hit3和每owner训练miss3hit317断言，不放宽。

全219消费者，N_COMPLEMENT_KL，seed760023，V62–V75完整三元组/scope四元组精确复用、其余混盲全文冻结后映射。主V64 A/B、直接V69；原联合gate不变（B35正确T2G保留、vsA净>=2且2正编辑、T0/T1G/改写正确保持、locality总数和Base正确保持、源断言A子集、无新增cap；歧义阻断）。所有数据曝光DEV、独立确认0；非临床专家验证。

3400back预计3380、1600upd、230gen预计221、400kF；教师提取额外30F单列，V69旧初始化成本保留。首次2小时墙钟/GPU进程与总11:59:35取早，独立守护提前300秒停止，重试不重置。远端1.5GB、本地.25GB，累计27.25GB，未分配2.75GB，删除不返还。正式10状态在完整消费者和评分后清单退休，小初始化/输出/冻结标签私有保留；唯一tt-24监测继续。
