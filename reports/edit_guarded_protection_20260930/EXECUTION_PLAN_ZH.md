# Edit-Guarded Protection 执行锁定

基于公开 PR8 冻结提交 `06620956d9cea39936e80a03d49fc91f93c9290e`，新工作树、新分支和新 run namespace。旧 PR7/PR8 的权重、结果、账本、永久缺失和科学判定保持只读。

P0 零训练、零Judge：对每个唯一缺失key使用同一个Boolean变量、先合并系数再计算精确界；重构记录中可识别的CE/anchor与保护梯度。S只保留了实际联合保护梯度，缺少单独KL/残差向量；SP固定batch诊断与实际训练clip曲线分开描述，不冒充完整实际训练梯度。

P1：historical REG bank12的A0冻结权重，完整FULL/REMOVE_9/REMOVE_10/REMOVE_11/REMOVE_12，每条件生成固定47个非目标输入与固定前8编辑的正例队列，重新运行真实R0。没有按专家ID推断答案或复用未经验证的生成。此诊断不参与rho选择，不等于本轮候选REG准入。

P2/P3：只改变clip之前的梯度合成。CAP为cap-only，EGP为one-sided projection+cap，EGP_A额外沿用已有0.1 positive anchor。lambda=0.1、rank4、layer30 down_proj、R0、seed20260929、共同INIT_POST80、旧负例池、数据/采样、precision/backend、optimizer/normalize顺序、80步与global clip1固定。投影eps=1e-12；代数检查使用float64内积，实际float32投影允许max(1e-8,2e-6*范数积)的舍入容差。raw D_plus只是一阶量，不是Adam单调保证。

先做CPU代数测试和独立GPU机械smoke：禁用surgery复现旧S/SP完整80步tensor-exact，Base/reference不更新、hook/mask不变、诊断不改RNG、断点恢复轨迹精确、surgery不新增backbone前向。smoke与正式训练/费用分开。

P4：冻结旧pilot8=[4,5,6,11,12,15,16,21]，六个分支CAP/EGP/EGP_A × rho{1,2}，48条续训。仅生成CHECK供选择；新正式DEV输出在共享rho冻结后才生成/评分。rho不增加第三档。共享rho资格要求三方法CHECK_POS相对H不新增已知错误，缺失依赖界不妨碍资格，所有代数检查通过。默认跨三方法CHECK_NEG source-macro等权平均，并列选2；此聚合口径已提前向用户询问，若用户在新结果揭盲前指定其他口径，明确修订锁定。pilot clip<10%为报告的mechanism目标，不是替代accuracy指标。

P5：选定后其余DEV16运行三方法（48条），加选定pilot分支，完整single/bank4/8/12/24、fixed early cohort、EXPOSED_REGRESSION old47。合计96正式续训；旧E_orig/A0/H/S/SP只读复用。所有共同输入采用同一依赖-aware missing模型和固定完整分母。

P6/P7：每step记录真实g_plus/g_minus内积与norm、projection/cap/conflict/clip、D_plus、final-positive cosine、参数update norm；事后只分析暴露结果，不重选rho、不改更新规则。T0无新增已知错；T1G对A0/E_orig无新增已知错；T2G最多新增1个明确错且精确缺失界不能妨碍判断；bank24压力比H至少+3已知正确且完整分母差下界>=0；35个冻结Base-correct每前缀相对H最多新增1个损伤；正式T2L新增>=2已知错则FAIL。若accuracy合格但full-DEV clip>=10%，记PERFORMANCE_PASS_MECHANISM_UNRESOLVED，不声称dominance解决，也不作为明确joint PASS准入。

P8：只有明确DEV PASS才按任务条件选择REG，非支配候选最多2个，每个24条、80步；不自动运行所有方法。REG前完全锁定，不调rho/loss/gate。所有候选FAIL/INCONCLUSIVE就收口，不为8–16小时目标窗口增加实验。

评分：固定GPT-6.1-sol/high、原模板和原payload key定义。新成功payload严格绑定后可复用；PR8的40永久失败keys在新阶段仍为永久缺失，不重判。新提交统一预注册为最多一次语义/transport尝试，失败整批记缺失，不按成绩选择性重判，不改key/prompt。累计账本继承6867并持续加记；用户已取消评分条数限制，内部cap保持None。连续3次传输失败暂停新提交并保留现场。

GPU仅5/6/7，启动前UUID/显存/租约检查，每卡一个resident backbone。新增存储soft10GiB/hard20GiB，保留单latest optimizer/RNG槽与必要final adapters；无依赖的未选pilot及完成恢复槽按清单删除，旧权重不动。依赖顺序P0→P1与实现/机械验证→pilot→rho→fullDEV→有条件REG→收口。复用有限控制器后台推进，不依赖聊天常开。

启动状态：P0已完成；P3真实模型机械测试PASS，480步且零Judge。有限控制器与本地评分进程已启动，当前执行P1完整五条件路由移除块；正式pilot尚未启动。小时监测此前在PR8交付后暂停；新阶段尚未自动恢复定时监测。最终实际结束后发布源码、配置、测试、脱敏聚合及限制到独立审阅分支/PR，验证远端SHA与匿名访问；不公开医疗输入/回答、逐题私有评分、weights、tokens或凭证。

最终状态：96条续训及DEV评测完成，三个候选均FAIL，REG_NOT_ADMITTED。共享rho=2只由CHECK选择；资格读取故障已修复，原训练/评分未重跑。详见FINAL_RESULTS_ZH.md。
