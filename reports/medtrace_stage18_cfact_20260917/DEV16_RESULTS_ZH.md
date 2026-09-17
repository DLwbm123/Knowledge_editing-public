# Stage18 DEV16 实测结果

四方法训练、single 与真正逐条插入专家库的 sequential 生成、统一 Astra 判定均已完成。主顺序最终 16 条的 R0 PairCorrect：C_FACT **56.25%**，C_NO_H、C_EXTRA、BalancEdit 均 **31.25%**，差 **+25.00 个百分点**。这支持本次探索 pilot 的 H 监督增益，不能据此宣称患者独立、确认性或临床优越性。

## 主结果：最终 sequential，R0

PairCorrect = 每条编辑的 native 正确性 × 其 H_eval 正确比例，再对全部 16 条编辑取均值。native 失败必须保留在分母；本轮四方法恰好均为 16/16 native 正确，因此 PairCorrect 与 H_eval 的 edit-macro PostAcc 相同。Base native 为 0 是预先筛选 Base-wrong 的结果。

| 方法 | native 正确 | H_eval / PairCorrect（edit-macro） | H 正确槽位（micro） | U_eval 保留率 |
|---|---:|---:|---:|---:|
| Base | 0/16 | 43.75% / 0.00% | 9/18 | 100.00% |
| C_NO_H | 16/16 | 31.25% / 31.25% | 5/18 | 0.00% |
| C_EXTRA | 16/16 | 31.25% / 31.25% | 5/18 | 100.00% |
| C_FACT | 16/16 | 56.25% / 56.25% | 9/18 | 100.00% |
| BalancEdit | 16/16 | 31.25% / 31.25% | 5/18 | 0.00% |

U_eval 只有一个不同 QA／图像，在 final bank 下被 16 条编辑共享，不能把 16/16 当成 16 个独立 locality 样本。H_eval 的 18 个槽位来自 7 条不同 QA、4 张图像；编辑宏平均与探针微平均不相等。

## Single、前缀与训练诊断

| 方法 | single R0 PairCorrect | sequential N=1 | N=10 | N=16 | single U 保留率 |
|---|---:|---:|---:|---:|---:|
| C_NO_H | 0.00% | 0.00% | 20.00% | 31.25% | 31.25% |
| C_EXTRA | 0.00% | 0.00% | 20.00% | 31.25% | 81.25% |
| C_FACT | 68.75% | 0.00% | 50.00% | 56.25% | 87.50% |
| BalancEdit | 0.00% | 0.00% | 20.00% | 31.25% | 25.00% |

C single FORCED_OWN 的 H_eval / PairCorrect 与 R0 一致；完整固定 RC、FORCED_OWN、早期锚点和插入→最终轨迹见 JSON／CSV。前缀变化同时包含队列扩展，不能直接解释成遗忘；16 条 native 从各自插入时到最终均保持正确。

训练诊断（FORCED_OWN）中，C_FACT 的 H_fit 为 16/16，C_NO_H 与 C_EXTRA 为 0/16；三分支 U_fit 均为 16/16。这是训练拟合，不能替代 H_eval 或 U_eval。

## 仍然存在的问题

- 全部 H_eval 在 single 和最终 sequential 都被 R0 激活，路由没有拒绝这些范围外请求。C_FACT 收益来自激活后的回答改善，不能称为路由边界问题已解决。
- C_FACT single H_eval 为 68.75%，最终 bank 为 56.25%；这两者有不同专家选择语义，不是继续训练导致 writer 遗忘。
- 最终 C_FACT 对 Base 原本正确的 H_eval 仅保留 4/9 槽位（micro 44.44%；7 条相关编辑的 macro 57.14%），仍有 5/9 被破坏。其他三个方法为 0/9。对 Base 原错的 H_eval，四方法都是 5/9 修复。
- native 为 16 条 QA、13 张图像，H_fit 为 6 条 QA／4 图像，H_eval 为 7 条 QA／4 图像，且全部编辑共享 U_fit，训练支持连接图只有一个连通组。患者／研究身份未知；既往训练曝光已在 DEV 版本声明。
- 未新增独立 T1G/T2G/T1L/T2L 泛化探针；这些指标为 NA。未独立执行冻结语义下的 native-target-copy Judge，不用字符串相等冒充语义复制比例。

## 差值与不确定性

最终 PairCorrect 的 FACT−NO_H、FACT−EXTRA、FACT−BE 都是 +25.00 pp：4 条改善、0 条变差、12 条相同。按预设 seed=20260912 做 10,000 次成对重采样，编辑级描述区间为 [+6.25,+50.00] pp；13 个 native 图像组敏感性区间为 [+5.88,+50.00] pp。共享支持的依赖没有因此消失，这些区间不能作为患者独立显著性证据；完整支持连通组／患者级区间不可估。

## 执行和评分证据

- GPU3 任务 `job-20260917-1805`，worker 退出 0；16/16 编辑、48/48 C continuation 完成，每分支 320 步；H/G 各 5,120 次非零梯度，16 个 BE 各 50 步且损失／梯度有限。
- 共享 W0、训练步序、损失权重、Base OFF、路由隔离、保存恢复与完整面板绑定核验通过。424 条部署评估记录、192 条训练诊断完整。13 项 CPU 检查通过。
- GPU 总墙钟 7,808.07 秒（2 小时 10 分 08 秒），北京时间 2026-09-17 18:46:42 完成；此前估时略保守。后续评分未随 GPU worker 自动触发，本次询问时补接，不能把中间等待计作计算耗时。
- C 初始化／continuation／训练诊断合计 7,439.99 秒；C_NO_H、C_EXTRA、C_FACT continuation 分别 1,504.31／1,986.93／2,015.05 秒。共享初始化只计一次；不能把这几个阶段重复相加。BE、路由和 I/O 未分别计时，不补造分项成本。C 阶段观测峰值分配显存 15.45 GiB，不冒充全任务 BE 峰值。
- 完全相同的图像身份、问题、参考和回答文本复用本轮既有 Base 判断；其余 68 个不同文本统一匿名评分，Astra/high，snapshot=null，68/68 通过格式与覆盖校验，无语义重试、无工具／记忆／项目上下文。评分调用耗时 105.39 秒，输入 15,638 tokens、输出 3,216 tokens。
- GPU 实现提交 `0e9e8458ac7dab5018930837047d549aa294d20b`。评分新增读取器不改训练、权重、队列或 frozen Judge。

[完整聚合 JSON](DEV16_RESULTS.json) · [全部部署 CSV](DEV16_RESULTS.csv) · [机械验收](DEV16_ACCEPTANCE.json) · [来源建设](CURRENT_DATA_DEV_V2_ZH.md)

私有原图、QA、逐样本输出／tokens、Judge 映射和权重不公开；只公开源代码与脱敏聚合。所有已登记权重消费者完成后，已删除本次 144 份无后续依赖的 checkpoint／optimizer／初始化状态，约 3.56 GiB；保留一份约 4.86 MiB 的最终 C_FACT bank（16 个小专家）用于部署复现，下一阶段启动前复核去留。router、前缀、原始输出／tokens、评分与绑定材料保留；未触及历史文件。被删权重无备份，重建须重训。见 [清理回执](DEV16_CHECKPOINT_CLOSEOUT.json)。
