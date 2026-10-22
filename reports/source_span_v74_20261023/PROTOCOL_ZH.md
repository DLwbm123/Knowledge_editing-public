# V74：合法源预测状态子空间的连续残差门控

V73固定token加权未改善完整答案，且发现真实缓存入口被绕过，原cache PASS不能算实际cache准入；保留全部失败与成本，不重跑/搜权重。新独立假设：V69全局线性TT残差在非源相关token上也生效，可能造成保护损伤；固定源输入子空间的连续投影门能否保留泛化并限制越界？此为TT主线新作用机制，不是更换对比方法或为测试题设计路由。

仅在未编辑Base上，从原合法源问题和既定两训练改写的teacher-forced答案预测位置（label非-100的位置减1，含EOS预测）捕获第21层down_proj输入。全10源、30前向，0反传。每编辑合并全部三问的矩阵X，在CPU64作reduced SVD；保留singular value > eps64*max(X.shape)*smax的全部数值非零模式，U为右奇异向量列，转FP32部署。此阈值只用于数值秩判定，固定无搜索；不是按测试选择投影维度。全10基底在训练前冻结并save/load精确；报告维数和源投影覆盖，源gate_min>1-1e-5，否则STOP。

每token使用其当前原down_proj输入x，g=clamp(||xU||²/||x||²,0,1)，全零x取0，分母仅用FP32 tiny避免零除。TT残差乘g；不中心化、不加阈值、温度或可训练路由。训练/teacher/prefill/单token decode同一公式，固定基底不吸收评测状态。基底本身是新增存储和计算，不能声称参数/成本完全相同。投影比例不是语义相关性真值，不保证保护，也不据源覆盖接近1宣称训练轨迹精确相同。

恢复V69全10未训练初始化、均匀完整答案原CE，原源+交替改写两CE平均、全7168核心、clip1/两组RMS/canonicalize、lr/beta/eps、160步、rank4、layer21、ALL_TOKENS及生成设置不变。参考V64 A/B，直接机制对照V69。无新训练数据/评价答案/保护标签/网格。

CPU检查源空间gate1、正交/全零gate0、混合gate0.5、缩放不变、批/逐token一致、真实TokenHook teacher/generation一致、零残差精确、生产cachefixture。真实30绑定原zero/nonzero/cache loss-gradient精确/teacher-KV rtol1e-5 atol5e-4/RNG/saveload全通过才训练。额外显式检查每状态cache miss=3/hit=3，每160步训练miss=3/hit=317；必须调用原runtime.compute_loss入口，阻止V73空缓存验证重现。任一失败STOP不放宽。

全219消费者，L_SOURCE_SPAN，seed740023，V62–V73精确完整三元组及scope四元组冻结标签复用，剩余混盲全文评判后映射。V73描述性输出可复用语义标签，但其机制准入不完整，不充当已准入模型证据。主联合gate不变：保留B35正确T2G、相对A净>=2且2正编辑、保持正确T0/T1G/改写、locality总数不降且无新增Base正确损失、源断言A子集、无新增cap；歧义阻断。全为曝光DEV，独立确认0，无临床安全结论。

1600upd/3400back预计3380/230gen预计221/400kF；新增30Base提取前向单列，V69旧初始化成本保留。首次2小时墙钟和GPU进程，与总11:59:35取早，提前300秒独立停止；失败不重置。远端1.5GB本地.25GB，累计23.75GB，未分配6.25GB，删除不返还。10正式状态所有消费者/冻结评分完成后清单退休；小初始因子/基底/完整输出标签私有保留。唯一tt-24半小时监测继续。
