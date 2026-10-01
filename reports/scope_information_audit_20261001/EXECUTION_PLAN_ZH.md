# Scope information audit 固定计划

用户授权自主逐轮审阅设计执行，基于PR11公开冻结d072028ac134a35a4ba2a3f910baa0716a6c8eb7。PLOO与NEG0全部989正式消费者路由相同，且T1G失败，下一步不再盲调球半径。

执行一次固定、暴露的几何与来源诊断：完整single/bank4/8/12/24/old47消费者，保留所有missing；按已知positive rejection、Pressure/old47 negative rescue、missing及其他已知分组，报告guard覆盖比、NEG0 quotient、最近全U_bg/positive距离比、原R0距离/radius，以及same-image/cross-image归属和不同input计数。分组只用于解释已知结果，不拟合新router、阈值或classifier，不做部署判定，不读取REG输出、不新解封数据。

只用冻结Base latent、完整绑定的原输出/Judge缓存；GPU6计算（许可范围6/7，禁止5），无需驻留backbone，不生成答案、不发Judge、不训练。GPU最多0.25小时，墙钟最多1小时，128MiB输出上限、磁盘至少8GiB余量。算法与汇总定义在输出前冻结，失败/修复消耗保留，预算不重置。

同步构造私有review queue：继承CHECK候选明确UNVERIFIED和已暴露，不能计作新verified CAL或独立CONFIRM；不把formal正例搬为新训练支持，不自动改写冒充cross-image。该盘点仅限已有冻结资产，不宣称全量原始数据检索。真正合格的下一轮scope实验依赖每pilot至少4合法verified同图和2隔离verified跨图样本，需要真实验证证据。

执行后公开代码、聚合几何、CAL readiness、限制与账本；逐题几何/QA/图像/待审明细私有。若仍缺verified scope数据，明确后续科学实验阻塞、保持小时维护检查实际新材料，不用无依据新GPU实验替代标注。

最终状态：COMPLETE，989消费者审计完成，0训练/生成/Judge，GPU6驻留约1.78秒；后续NEEDS_VERIFIED_SCOPE_CALIBRATION。16条私有CHECK提案未验证，尚不能启动使用新CAL的科学实验。小时维护保持启用，等待真实新标注或用户提供来源。
