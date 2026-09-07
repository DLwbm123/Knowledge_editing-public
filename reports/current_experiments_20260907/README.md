# 当前实验核查与已完成轮次交付（2026-09-07）

MedTRACE visual-verifier R1 已恢复并完成 21/21 个任务、42 个 verifier fits，以及新增 275 / 复用 759 个完整元组的 Judge 判分。M1–M3 均未通过既定开发保留标准；详细结果与最终独立状态见[恢复轮次审阅报告](../medtrace_visual_verifier_recovery_20260907T045800Z/GPT_PRO_REVIEW.md)和[完成记录](../medtrace_visual_verifier_recovery_20260907T045800Z/RUN_COMPLETION.json)。此前的 LoRA-Perf `QUAL_VALIDATION_FAIL` 和 MedTRACE execution-preservation 未改善结论保持不变。这不代表完整 M3Bench、TIME/MedTRACE 或临床验证完成。

## 1. 恢复前的运行状态（历史快照，保留失败事实）

以下为 2026-09-07 较早一次通过 `my-gpu` 进行的只读核查；该次核查没有启动训练或重新执行 Judge。后续已授权恢复工作另存于上方链接的新报告。

- 最新运行：`/remote-home/wangbomin/medtrace_runs/20260907T032521Z`。
- 存在 `ORCHESTRATOR_FAILED`；`worker2_rc=1`、`worker3_rc=1`、`queue_complete=0`。
- 两个 worker 因缺少 `private/CAMPAIGN_START.json` 退出；任务账本中的 21 个任务仍为 PENDING。
- `prepare_judge_rc`、`judge_rc`、`finalize_rc`、`commit_rc`、`bundle_rc` 均为 99（未执行），不是成功退出码。
- 检查时 GPU 无计算进程；GPU 空闲并不代表实验完成。本次没有修复或重启这一轮。

## 2. LoRA-Perf-v1：已完成，QUAL_VALIDATION_FAIL

运行：`m3bench_lora_perf_v1/20260905T012834Z`。DEV 比较 3 个学习率与 5 个训练步数检查点，共 15 个候选；只选定一个配置进行一次 QUAL16 验证。

选定配置：rank=16、alpha=16，所有语言模型 MLP 的 gate/up/down projections，学习率 5e-4，80 steps。数据来源为运行目录中的公开 CSV 及最终汇总；QUAL Judge 进度为 255/255。

| 指标 | 选定配置 DEV16 | 一次性 QUAL16 |
|---|---:|---:|
| T0 编辑正确率 | 15/16（93.75%） | 16/16（100%） |
| T1L macro | 41.43% | 20.00% |
| T1G | 89.06% | 100% |
| T2G | 89.06% | 100% |
| 同问题换图后复制编辑目标的比例 | 92.05% | 100% |
| target NLL 下降 | 16/16 | 16/16 |
| 空输出或错误 | 0 | 0 |

QUAL 的 T1L=0.20，低于冻结门槛 0.25，因此没有通过资格验证。编辑与泛化指标较高，但换图复制比例表明图像条件区分能力不足；不能将复制目标答案计为视觉编辑成功。没有使用 QUAL 再选参，也没有在本次启动正式大规模实验。

证据：[DEV 候选表](lora_perf/DEV_PARETO.csv)、[QUAL 指标](lora_perf/QUAL_METRICS.csv)、[冻结协议与门槛](lora_perf/PERF_PROTOCOL.json)。旧源码中的 `JUDGE_PENDING` 报告已被本次读取的最终产物取代。

## 3. MedTRACE execution-preservation：长实验已完成，未改善

运行：`/remote-home/wangbomin/medtrace_runs/20260906T141606Z`。训练、生成与 Judge 均已完成；108/108 个主要训练轨迹、36/36 组匹配三元组、108/108 个 P1 任务完成。记录的工作 GPU 时间为 20.951 GPU-hours。Judge 覆盖 5,628 个唯一元组，其中 4,597 个新判决、1,031 个严格匹配复用，解析全部有效。

主比较：additional step 800，SAFETY_FIRST，以下为 micro 指标。

| 方案 | 强制执行正确率 | 门控后正确率 | 困难负例误触发率 | 广泛负例误触发率 |
|---|---:|---:|---:|---:|
| C1：固定 Q，延长输出恢复 | 180/180（100%） | 169/180（93.89%） | 20/84（23.81%） | 13/636（2.04%） |
| C2：共享 Q，联合训练 | 180/180（100%） | 171/180（95.00%） | 23/84（27.38%） | 14/636（2.20%） |
| C3：联合训练加执行锚定 | 180/180（100%） | 166/180（92.22%） | 23/84（27.38%） | 10/636（1.57%） |

冻结结论：`EXECUTION_RETENTION_NOT_IMPROVED`、`ROUTING_EXECUTION_PARETO_NOT_IMPROVED`。强制执行能力较好，但困难负例上的误触发仍明显；C2 的门控后正确率略高，同时困难负例误触发也更高，不能认定整体优于 C1。

三个种子重复同一批 12 个事实，困难负例支持仅覆盖 7 个编辑；180 和 84 是重复评估的分母，不是独立事实数。这些是已查看的开发集结果，不是盲测或临床验证。

证据：[结果摘要](medtrace_execution_preserving/PAIRED_E2E_SUMMARY.md)、[逐编辑与种子指标](medtrace_execution_preserving/PAIRED_E2E_BY_EDIT_SEED.csv)、[训练曲线](medtrace_execution_preserving/EXECUTION_PRESERVATION_TRAINING_CURVES.csv)、[完成记录](medtrace_execution_preserving/RUN_COMPLETION.json)、[Judge 完成记录](medtrace_execution_preserving/JUDGE_EXECUTION_AND_CLOSURE.json)。原始完成记录的 `publication=RETRY_REQUIRED` 是当时的发布状态，原样保留；本次提交补充公开交付。

## 4. 代码来源、验证和公开边界

- LoRA 源码来自远程干净工作树提交 `90c93e6189218c097090f25e9f6c19d703be127f`，位于 [LoRA 源码快照](../../experiments/lora_perf_20260905/)。
- MedTRACE 源码来自远程干净工作树提交 `f00d933a5ebe02a31348d031b5f4a820520a6a2f`，位于 [MedTRACE 源码快照](../../experiments/medtrace_execution_preserving_20260906/)。两个实现单独保存，保留各自运行时依赖版本。
- 本次没有修改算法。发布包在服务器既有 Python/Torch 环境、GPU 隐藏的条件下完成 6 项相关 CPU 测试；Python 语法及 CSV/JSON 可读性检查通过。本次未重跑 GPU 实验。
- 发布内容为相关源码、已有测试、冻结配置、结果 CSV、训练曲线与报告。没有发布原始图像、问题答案、患者数据、模型权重、checkpoint、私有 Judge 包或映射、完整运行日志、私有仓库历史。
- 这是源码和可公开结果的复现材料。重新训练仍需要依法获取模型和数据、准备原协议输入与初始专家，并匹配运行环境。快照不包含原始 Git 历史；源码的提交/祖先检查仍保留，重新执行应建立新的明确来源记录，不能伪装成原运行。

最小可运行检查（使用已有 Torch 环境）：

```bash
cd experiments/medtrace_execution_preserving_20260906
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m unittest discover -s tests/medtrace -p test_execution_preserving.py
cd ../lora_perf_20260905
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m unittest discover -s tests -p test_m3bench_lora_perf_finalize.py
```
