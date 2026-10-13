# 执行实现与机械修复

冻结科学方案d85ac1d；两个GPU前/准入期修复仅涉及运行绑定和checkpoint状态，不改结构、数据、优化器配方、loss或门槛。首时钟和失败成本不重置。

- 首次CPU评分自检因新run尚未创建private/tmp停止；补齐本轮owned目录后完整CPU准入PASS，0 GPU/0 Judge。
- GPU检查器import worker命中旧源中的同名模块；改为显式传入活跃worker接口。失败12.905511 GPU秒计入原账本。
- GPU save/load检查发现map_location=CUDA把非capturable Adam的CPU step计数器也搬到GPU，状态对比失败。采用CPU反序列化，再由optimizer.load_state_dict迁移参数对应moments，并显式恢复hook张量设备；下一步参数/optimizer/RNG一致性仍严格验证。此前GPU消耗累计30.223767秒全部保留。

所有旧失败日志、执行源码和资源快照保存在本轮private/recovery；后续以GPU_P0_DETAILS与实时回执为准，不能将本说明当成GPU检查已通过。

TT最终部署通过G1–G4实例直接接入原R0 hook，临时B/A每次从核收缩，不持久化也不注册参数。参考PR26的结构代码只用于CPU TT44一致性测试，未调用FREE/SVD训练。

复制的只读兼容帮助代码来自PR26已固定运行：audit.py、admit.py、legacy_worker.py、legacy_queue.py、legacy_metrics.py；GPU_SOURCE_VERSION按文件绑定完整实际执行集合。原基础执行source为2b3f308f2bcb442cd18e66ac98a83c1528c0692b，官方LLaVA为30697ca50b5c29a8e955c99330b259776aef27b9；使用既有本地模型和图像，不下载替代来源。

复现需要合法私有原始ledger、图像、12H证据及原Base缓存；公开代码不捆绑这些材料。teacher仅为原冻结Base生成的U完整词表分布，方向Base||student，逐预测token平均；共享U不算独立CAL。
