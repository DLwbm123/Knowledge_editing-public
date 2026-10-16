# 生成完成与评分中断报告

2026-10-16 21:09 北京时间核验。本轮生成完整结束，评分因代理/TLS预检失败停止；完整性能评估尚未完成，不据部分覆盖判断科学门槛。

## 已完成生成

CM_GENERATION_COMPLETE1记录17081个消费者输出：706完整身份复用，16375新生成。两个single分区3645/3654，三个bank分区3019/3027/3030，三bank均COMPLETE及EXIT0，所有评估GPU租约已结束。正式计数447833 forwards、14617非生成forwards、0训练更新/反向/operator backward/solve；独立工程探针沿原回执单列。

## 评分覆盖与停止原因

四路评分沿用gpt-6-astra、medium、原提示词/判定标准及一次尝试规则。SQLite共有5002唯一payload、17081消费者，当前覆盖如下：

| 状态 | 唯一payload | 消费者 |
|---|---:|---:|
| FORMAT_VALID | 1399 | 4393 |
| PENDING | 3603 | 12688 |

有效payload中699条为历史身份匹配复用，700条为本轮实际评分，全部格式有效。没有RESERVED或已尝试缺失项；3603条PENDING均未绑定评分批次。

worker3在下一批预留之前触发`Proxy/TLS preflight failed before reservation`，留下SCORER_FAILURE。ingest等待器检测到故障后失败，控制器记录`Child failed; no automatic retry: cm_ingest_0`并EXIT1；其余三路按OWN_STAGE_FAILED停止，四个本地评分进程均已退出。未生成全局ALL_WORKERS_COMPLETE、四路SCORER_DONE或最终性能报告。此为评分接入故障，不是模型性能失败；当前证据不足以细分代理连接或TLS故障原因。

没有自动重试、重判或重新生成。保留所有有效评分、待判项、原故障与成本；恢复需遵守新的明确授权和原准入，不复活失败入口或清除失败记录。原暂停训练状态保持，未清理共享权重。

## 成本与结论边界

本轮新增GPU成本23468.750889秒，新增Judge700，GPU和Judge成本转移回执均存在。采样项目累计GPU成本1083117.115427秒（包含暂停租约及独立工程成本），累计Judge19152；仍在原预算内。暂停租约继续计费，该GPU累计值仅代表采样时刻。

仅交付生成完整与评分中断证据。完整FIT/T2G/T2L及damage逐owner门槛暂不判定，也不启动依赖完整结果的后续科学实验。公开内容仅匿名聚合与工程错误摘要，逐题回答、评分记录和权重保留私有。
