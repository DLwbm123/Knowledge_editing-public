# E01固定SignedBalance有效增量只读响应分解（预注册）

必要假设：同一失败native有效增量在可编辑层FP32参考链路内仍可能有有限步反号；若该链路全步仍下降而native上升，则局部执行精度干预差异有实质贡献。二者可以并存，不能把参考残差直接称纯曲率或全链路精度结论。新的观测仅为机制诊断，不是重试SignedBalance、续跑其失败状态或选择半步编辑。

范围：唯一E01、OrthantBalance成功起点、已失败SignedBalance固定有效native增量q。方向/解析系数/14.94248258401418幅度/坐标保留及唯一镜像删除前缀完全冻结。只内存重构q并核旧cost/dot/坐标数，历史完整失败native终点forward0，直接复用已公开读数。历史权重直接比较NA，不凭标量称逐字节一致。

不做新解析/匹配/尺度或mask搜索/新secant/生成/Judge/接受编辑。必要原控制和成功路径按既定固定内存重构，成功补偿只冻结alpha内存重建，不再探测。先绑定当前成功起点3forward控制及2forward局部梯度，保持原门、旧F0/trust及最初基座null不变。

诊断固定次序：F0三forward读取NLL/null/固定与最强竞争margin；F0与刚取得梯度基点NLL/null/margin绝对误差≤1e-5。F_half_unrounded三forward用W.float+.5*q；F_half_rounded三forward用half(W.float+.5*q).float；F_full三forward用原失败native全步参数.float，不再原native全步前向。F0返回两forward复核。退出已有editable_fp32上下文后核native基点权重/dtype恢复；native半步三forward记录原三门和margin，只作诊断，原门拒绝不触发救援或挑候选。native基点返回两forward复核。诊断累计19，硬上限20；全部读数与状态先保存再判资格。任一实际前向权重/input dtype/调用数/复读/有限性/绑定/恢复失败立即停止，不重试、不降级、不补点。完整链路FP32不调用。

F仅已有可编辑Linear参数/输入暂FP32、输出回原dtype，其他链路仍混合精度。对NLL、固定竞争者margin、null，各位置t定义P=t*g_F·q；R=F_unrounded(t)-F0-P；Q=F_rounded(t)-F_unrounded(t)；E=[N(t)-N0]-[F_rounded(t)-F0]。恒等式绝对闭合≤1e-10。全步两端参数均原native格点，在同q下Q=0；半步Q单独记录插值位置写入舍入。E是该局部执行干预差，不是全链路精度误差。半/全R比值只报告，不以不等于1/4唯一证明某种机制。最强竞争者变化单列，不将固定竞争margin误当最强竞争margin。

候选上限0；固定诊断位置0/.5/1，不追加位置或基点。全部位置均恢复且不得接受，原native门只对native读数诊断记录。若完整步参考F已反号，可说明参数写入舍入不是此固定有效增量反号的必要条件；不能由此排除其余混合执行影响。若资格失败，缺点NA，不补旧缺失或重放旧工程失败。

预算：单GPU900秒含加载、控制、原校准复用和恢复；工作870秒/580forward，总600forward；新增64/工作60留4恢复，诊断20独立上限；新增磁盘50MiB/checkpoint0。12小时累计≤4GPUh、最多24轮，用户新增合计20GB、内部18GB；现累计1519.627683秒，完成18轮，按账本预留本轮900秒。不得以预算有余为由重复调用。2026-10-20 11:32停止新启动。

任何失败保留现场/成本/缺失并恢复，完成后匿名代码/全标量轨迹/中文报告经既定代理推送、核远端与匿名访问，再分析有无独立必要干预。私有QA/token身份/向量/mask/权重不公开；无subagent。
