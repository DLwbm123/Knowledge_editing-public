# 源/null双残差恢复对照

前轮末候选null已恢复而source恶化，因此新机制必须修复实际source违反，而非只一阶保持source。固定SOC2对照与BRC2新臂，从共同原可行种子出发，各最多4步、原lr .05/六尺度/最多3次修正/半径.05，不增加任何次数。两臂都刷新source/null梯度，预算上限相同，实际成本另列。

SOC2仍只有null残差且source一阶不变。BRC2每轮计算r_s=max(0,S_native(candidate)-S_native(previous accepted))、r_n=N_native(candidate)-N_native(W0)，以最小范数同时求gᵀdelta=-r_s、hᵀdelta=-r_n；source已不恶化时退化为原SOC。只有null通过且source不恶化才能提前结束BRC恢复。使用双精度正交分解解两行系统、退化/范数>.05拒绝。每轮实际native复测，不把线性解当实际可行保证。其他source逐端点不增/成对严格下降/null<=.01/F0<=.001与最终H1标准不变。独立新实验，旧配方/失败不改写，不以教师拟合或旧S/A32改名。

仅E01开发源，原F0/rank4/真实KV资格，H2/H3未测。四EOS各128F、最多4去重盲评分一次。最坏3base+48seed+32梯度+192预测复测+1152双臂修正+4恢复+512生成=1943F；模型反向1+32+576=609，上限640。硬2100F/2080工作/20恢复、640B/paths、工作1080秒TERM1140/KILL1200/整体1h/256MiB。GPU固定UUID48GiB入场36GiB owned，新目录启动前mount/容量/写读probe，独立guard。所有新增成本计入，达到限即停，不加步/回溯/修正或改目标。只原生最后接受端点，一次评分不回流求解。
