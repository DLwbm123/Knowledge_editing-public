# Stage24E-R2 A CPU 审计与重分析

状态：**R2-A 已完成；R2-B/R2-C、Stage25 均未启动。**

本轮严格按 `R2-A_CPU_AUDIT_FIX_AND_REANALYSIS_ONLY` 执行：新增 GPU=0、新增 Judge=0；Stage20/21/22/23/24E-R1 正式工件、输出和判定保持只读。R1 的 N=19、B0/B1/B2 结果原样保留，仅用已有缓存做一致性重分析，未新增语义判定。

## 发现

静态代码审计确认 F01–F06 的协议偏差，但运行绑定证据不足，故不能把 R1 291 条输出解释为完整的事件、teacher、mask 或早停覆盖：

- F01：`slot_by_t={...}` 会按前缀覆盖槽位；编译计划有 23 个槽，其中重复前缀 5 组。
- F02：U teacher 解析含静默 `next(iter(teacher_map.values()))` 回退。
- F03：计划含 19 个前缀的 U mask，但 worker/training 更新路径没有消费 mask 状态。
- F04/F06：有 291 条生成记录，但没有独立 runtime event、teacher 一致性、early-stop 或 release-barrier 账本。
- F05：输出键只含 arm/prefix/query/mode，不含 event 或有效参数版本。

## 校验与修复

R2 隔离 patch 位于 `reports/medtrace_stage24e_r2_audit/private/patch/`，未部署到活跃 worker。其 CPU 测试 5/5 通过；配套包测试 31/31 通过，公开 fixture 结构审计通过。

## 资源与账本

共享 GPU 历史累计 25484.515684/28800 秒，账面剩余 3315.484316 秒；本轮不消耗。R1 已发布回执记录 30 个新 Judge、92 个精确复用，但远端公共 Judge 快照显示 0 dispatched，账本存在未解决不一致，不能据此推算可用余额。远端磁盘约 8.4 GiB，达到 8 GiB 保留线，未做清理。

## 收口结论

当前分类为 `PROTOCOL_DEVIATION_CONFIRMED_STATIC_RUNTIME_UNRESOLVED`，下一阶段 **NOT_READY**。在 runtime event/teacher/mask/consumer 绑定和 Judge 账本对账前，不应启动 R2-B/R2-C；若后续单独授权，应使用隔离 patch 和新鲜 N=19，不能复用 R1 的训练状态来替代合规执行。

R2-A 本地交付文件见本目录；未获得本轮明确公开授权，故未推送 R2-A 新文件。
