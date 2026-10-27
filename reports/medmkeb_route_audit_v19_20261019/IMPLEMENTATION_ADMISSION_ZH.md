# V19 实现与准入

复用父runtime、真实MemoryRouter、既有TT generation_request和原V18 prepared FP16→FP32路径。CPU父module_run先导入后全部新模块集成、OFF/本专家/另一专家分支、同输入复用、缺失/重复拒绝及真实欧氏路由已PASS，0新模型工作。首次集成harness遗漏V18_RUN环境在import前失败，0模型调用；已保留CPU_HARNESS_FAILURE1且补齐harness，未改实验配置。

协调器只读V18小回执，每60秒等两FORMAL_EXIT0/350消费者；完整后核验100 BE单专家别名，并按显存和全链截止启动中性GPU0/1入口。它不改V18训练/生成代码，不终止任何已有进程。新生命周期依赖锁已落盘，25文件原退休清单须等待这个新增有限消费者。

预处理实际image/prompt/runtime/generation/seed逐字段对既有Base绑定；本专家源key对保存router anchor按1e-6验收。每张输入一次query-key，共28；五个TT臂共享决定。非owner专家仅在没有同输入forced时新生成，最多30。全部140自然消费者与100已有BE单专家别名覆盖后，统一隐藏四分类；V18未完成评分时不能把评分依赖冒称完成。

本轮尚无GPU模型调用；冻结并启动的是依赖队列，不能称补测模型已开始。
