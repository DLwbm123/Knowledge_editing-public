# Stage21B 完成分析

B 的45条编辑、当次写入与真实11/19/32/45前缀全部生成并评分，298项正式输出全部覆盖。两次机械回放另计，不当独立样本。

|方法|H面板|总体正确|Base正确子集保持|Base错误子集Fix|来源宏保持|来源数|
|---|---|---|---|---|---|---|
|Stage20_FACT_H|old_H_eval|4/6|2/4|2/2|50.0%|2|
|Stage20_FACT_H|new_H_eval|11/24|8/18|3/6|39.6%|8|
|NO_H_HSIC|old_H_eval|2/6|1/4|1/2|25.0%|2|
|NO_H_HSIC|new_H_eval|12/24|7/18|5/6|37.5%|8|
|FACT_FIXED_L31_DOWN|old_H_eval|3/6|1/4|2/2|25.0%|2|
|FACT_FIXED_L31_DOWN|new_H_eval|10/24|7/18|3/6|35.4%|8|
|Stage20_BE|old_H_eval|1/6|0/4|1/2|0.0%|2|
|Stage20_BE|new_H_eval|8/24|3/18|5/6|16.7%|8|

L31 down_proj 保留了45/45 native纠错，但没有改善H保护：新H保持7/18，与L30 NO_H相同；严格新U保持0/8，低于L30 FACT和NO_H的7/8。旧U保持1/7，L30 FACT为3/7、NO_H为6/7。说明这一固定DEV流上存在明显位置相关退化；不能把它单独归因于H、模块或容量。

与BE L31 up_proj相比，B新H保持7/18对3/18、旧H1/4对0/4；native同为45/45，新U均0/8。只支持局部描述性优势，不能宣称全面优越。B与BE只同Transformer block，并非同模块writer-only对照；B与历史L30结果还跨已登记的GPU设备lane。预定1/19/45 Base-OFF token一致及两次save/load/OFF检查通过，不代表全输入跨设备数值等价。

Q1–Q3：H未提高已饱和native；H/U仍有取舍，不能普遍归因保护。Q4–Q6：原L21前提已由append-only修订改为实际L31，现有结果不足以分离层位、模块、writer和训练差异。Q7：历史HSIC均L30，动态选层收益不支持。Q8：真实路由、输出差异和切换正误转移见ROUTING_MECHANISM_ANALYSIS；条件诊断不替代全输入结果。Q9：纯writer张量、含optimizer文件和GPU时间分别测量。Q10：45条已查看DEV、单顺序和有限来源，不支持长序列、患者独立或临床验证。

B GPU进程墙钟8234.0秒；A+B累计11083.2/12600秒。B新增26项判定，A+B累计38/800；缓存命中558是消费者累计，不是独立QA。峰值已分配显存15.45GiB；每writer纯张量0.281MiB。实际货币费用提供商未暴露，NA而非0，tokens在账本。

Stage22 H主线L30在B结果前已锁定；全部新旧评价均DEV/regression。后续28,800秒/2,000项总授权不增加，Stage21余额不转入。保留现有bank/router/W0和原始评分绑定，当前空间足够，不作清理。
