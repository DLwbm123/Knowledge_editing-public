# Scope-positive Guard 固定实验计划

基于PR10冻结公开提交8363aad03e7dd8ad2198b5044325e5721e049314。用户2026-10-01授权每小时维护、自主审阅设计并执行后续阶段；本轮只GPU6/7，禁止GPU5。

PR10三个候选均因T1G误拒FAIL，尽管Pressure有救援；不重跑旧轮、不改旧门槛。新轮只测试一个支持保护假设PLOO：对native+四个S_fit的每个anchor，以其最近另一正anchor距离构造闭球（重复anchor半径保留0）；原R0最近winner激活且NEG0 tau=0通过或落入任一正例球即保留该winner，否则OFF。不得回退其他expert，不扩大R0原半径；不以formal input/正确性拟合球，不搜索乘子或阈值。这是独立的新科学规则，旧轮规则不改。

该球由同图fit支持定义，不是已验证的cross-image scope标注，也不保证排除所有U_bg。先role audit，构造/冻结全部48编辑的球，再实际模型token/RNG/hook/权重机械验证。仅保证原R0已经激活的合法fit支持不会被该guard额外拒绝，不扩大原R0范围。DEV包含R0/NEG0固定对照和唯一候选PLOO，各自完整single+bank4/8/12/24+old47，15任务；全DEV完成再判定。REG仅明确PLOO PASS才运行R0+PLOO，10任务。依完整绑定复用旧生成；PLOO仅保留原H或OFF，禁止无意义重新生成/评分。

沿用原资格和共享missing精确界，T0/T1G零新增错误、T2G最多1、Pressure bank24已知+3且下界>=0、old35各prefix最多1、非空正式T2L最多1，至少一处已知改变路由救援。无加权总分，缺失无法决定则INCONCLUSIVE，空分母不可评估。DEV/REG/old47均暴露诊断/回归，没有独立CONFIRM。

预算：新轮墙钟最多6小时，累计最多4 GPU小时（含机械/失败/恢复），新Judge最多500项，原累计7156不重置。GPU6/7各最多一个本轮模型；存储soft10GiB/hard20GiB、剩余>=8GiB，零新adapter/checkpoint。Judge固定原sol/high/prompt/key，41永久missing不重判，同payload最多一次新请求，连续3transport失败停止。

FAIL/INCONCLUSIVE/NO_ROUTING_EFFECT时REG不启动，不自动调本轮规则；完成报告、公开源码与脱敏聚合，新draft PR，验证SHA及匿名访问后按新授权审阅后续阶段。CAL_SCOPE仍缺独立verified跨图支持，优先为真正scope标注收集建立合法隔离与验证流程，不伪造临床确认。

最终状态：15/15 DEV完成，PLOO FAIL；REG=NOT_ADMITTED。代码51f211f固定；初始化与评分启动故障均已保留，无GPU/新Judge消耗前修复、原始计时不重置。
