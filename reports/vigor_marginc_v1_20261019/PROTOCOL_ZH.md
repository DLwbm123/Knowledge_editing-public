# E01受NLL下降约束的唯一C候选

用户明确直接执行。仅重建原正式E01旧16步终点一次，原nativeFP16/旧F0/旧trust/null/visual门/secant保持；原历史16全+4半/20secant只是必要控制，不为新C救援。复用已有A/B结果，不再评估A/B。共同起点NLL1.5107299089431763、margin -5.8671875。C成本C_A=5.368434395105091e-13，实际NLL下降底线d_A=0.07246601581573486，均冻结为旧A主更新记录。

当前源完整目标NLL梯度gL、固定位置1 margin梯度gm、原null NLL梯度g0；竞争者由共同native起点最强非目标token选定（含EOS可竞争），梯度固定身份，读出重取全词表最大。先用旧F0算null投影后的qL=P*gL与qm=P*gm；记EL=gL·qL、Em=gm·qm、cross=gL·qm。

解析求 max gm·delta，约束g0·delta=0、gL·delta≤-d_A、1/2 delta^T F0 delta≤C_A。先检查无NLL约束解deltaU=sqrt(2C_A/Em)*qm；若gL·deltaU≤-d_A，则NLL约束不激活、解退回旧B方向，保存标量后立即停止，不形成/量化/写入/模型评估旧B或C。若最小NLL成本d_A²/(2EL)>C_A不可行，或剩余方向/预算退化，停止。否则delta0=-(d_A/EL)qL，qperp=qm-(cross/EL)qL，Eperp=Em-cross²/EL，deltaC=delta0+sqrt(2(C_A-d_A²/(2EL))/Eperp)*qperp。Eperp≤1e-6 Em视为退化，与旧求解退化量级保持。方向的线性NLL需满足底线（数值容差1e-5）、预测margin增益>0、F0成本相对误差≤1e-5，否则停止，不换解。

仅解析解真正激活时最多形成一个C；一次FP16 seed成本预估及一次sqrt成本比修正，始终从同一终点重建，不累加seed。native成本误差>1%或量化后线性NLL条件失败即停止，不再投影/改比例/换候选。图像目标与null前向共2次即可求3个梯度，不重估F0。旧A/B的梯度能量、FP32 NLL/margin/null值按原1e-5绝对/相对条件复核，先保存检查值再判失败。

写入后记录真正最大竞争者margin及原门；实际main NLL降幅必须≥d_A-1e-5，否则立即恢复停止，不用secant救NLL。主NLL合格但null门失败时最多一次原secant，null参考仍最初基座。补偿后再次验原全部门、实际NLL底线和margin进步；任何失败回滚停止。只对合格C最多一次官方单token生成（max_new_tokens=1）；未形成/未合格C的生成翻转记NA，不补生成。局部翻转必须margin>0且实际首token目标命中；不能当完整T0/语义成功。主效果与旧A主margin收益0.4453125比较，最终成本（含补偿）单独报，不冒称最终等成本。

总900秒/600forward含加载/旧F0载入/控制/候选/恢复；工作870秒/580forward，预留30秒/20forward恢复，新盘≤50MiB/checkpoint0。新增C≤1、匹配≤1、secant≤1、单token生成≤1、Judge0，无半步救援/权重扫描/新方向/扩样本/146/重评分。约束不激活、不可行、退化、匹配误差、实际NLL不足、复核、恢复、数值或预算失败均结束整轮，无重试/追加/attempt2。保留失败现场及成本，匿名代码/全控制轨迹/解析标量/所有可用C读数/中文报告经7897代理推送，核远端/匿名访问。私有QA/答案/目标竞争token身份/向量/路径/权重不公开。中性主子argv，所有远端文件/data/bmw，不影响他人进程或历史资产，无subagent。
