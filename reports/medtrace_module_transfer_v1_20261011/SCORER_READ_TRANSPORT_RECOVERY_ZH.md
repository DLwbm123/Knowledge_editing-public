# 评分等待期 SSH 中断

2026-10-12 02:19 UTC 检查时，37/40 新臂训练已完成，M123 owner8 到80/160。活动租约计入后成本22.80546/24 GPU小时。生成屏障未达，评分队列0消费者/0payload，新增Judge0。四个本地worker在等待期的 `next` 只读查询上遇到SSH exit255后退出；这不是Judge请求失败，也不是GPU训练退出。

保留原SCORER_FAILURE、START、日志和远端故障回执，仅恢复同一状态/隔离/epoch的有限等待worker。共享remote函数对next/status/reserved三种只读操作的SSH255最多尝试三次，间隔2、4秒；reserve/publish、Judge调用及其他错误不重试。每完整payload一次的规则、Judge台账、矩阵和24GPU小时上限不变。小型自检覆盖只读255三次、reserve/publish255一次、非255一次。

剩余预算约1.19 GPU小时，训练之后仍需完整生成/评估；不保证预算内完成，不延长上限，不以部分owner宣称模块收益。硬停时保留活动恢复点并公开真实PARTIAL。
