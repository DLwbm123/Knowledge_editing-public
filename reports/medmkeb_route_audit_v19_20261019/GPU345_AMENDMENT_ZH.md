# 用户授权额外GPU的调度修订

2026-10-19用户明确要求用GPU3/4/5加速今晚取得结果。V18两条GPU0/1累计训练及所有科学条件不变，不中断/重训/重复生成。V19允许物理3/4/5，本次用3/4提前完成两分区各14个Base query-key，GPU5保留，不能为了占卡重复工作。

先在GPU3/4执行仅query-key预取，原V18已冻结输入与精度、每卡14语言forward/14视觉调用、0生成/训练。预取结束释放模型，只保存小型QUERY_CACHE.pt与输入绑定；这28次是原V19预算中的28次，不新增重复key。待V18全部消费者结束，GPU3/4最终worker复用缓存，不再计算key；原生成prepared多模态前缀仍按V18路径构造，视觉总上限100不变。每卡forward20000/生成15/0back更新、全40000/30、128MiB/8GiB预算不变；总4GPU进程小时包含预取与后续模型，最终2h墙钟减去最大每卡预取耗时，且不超过原全链截止。每张输入绑定及源key对原router的1e-6验收保留。

原CPU协调器尚在WAITING_V18_DEPENDENCY、无START/无GPU调用时，按PID/start_ticks精确退出并登记SUPERSEDED；不删除旧ONCE/日志，不触碰V18或其他进程。新协调器在相同截止等待V18及两PREFETCH_EXIT0，仅恢复未执行最终消费者。0/1为逻辑分区目录，START/LAUNCH分开记录partition与physical GPU，不能混淆。两阶段成本累加，不把预取伪称完整路由结果。
