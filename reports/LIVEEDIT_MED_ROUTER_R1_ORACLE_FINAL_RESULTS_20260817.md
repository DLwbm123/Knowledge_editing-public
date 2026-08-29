# LiveEdit-Med Router-R1 最终结果与下一步待判断问题

更新时间：2026-08-17（Asia/Shanghai）

## 1. 给 GPT Pro 的核心摘要

本轮实验已经完整结束。它是 **LiveEdit-Med EqKey-clean Router-Only Adaptation R1**，不是 ENGRAM、TIME 或其他编辑方法。

最终结论为：

```text
ROUTER_R1_NO_ELIGIBLE_CLEAN_VALIDATION_CHECKPOINT
selected_checkpoint = null
```

这并不意味着 LiveEdit-Med 的 generator/expert 完全无效。Forced-on 自由生成在 128 个 validation positive 上成功 72 个；Oracle O4 也逐样本精确复现了这 72 个结果。失败发生在多专家 Router-R1：原始 routed generation 只有 54/128，而且 validation safety 也明显不合格。

Oracle 的主机制标签为：

```text
PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE
```

在 step 640 上，单纯补回 hard gate 漏掉的 target expert 只恢复 2 个样本；在保持 target 原始执行系数不变的条件下删除所有 distractor residual，却恢复 12 个样本。这说明当前主要问题不是 target expert 本身、hard recall 或 sigmoid/softmax 强度，而是多个候选 expert residual 同时注入后的破坏性干扰。

当前最有证据支持的 Router-R2 方向是：

```text
Sparse / Top-1 Expert Execution
```

Safety 结果同时说明 Router-R2 必须包含可靠的 `NO_EDIT`/拒识机制，不能简单地对所有输入执行 top-1 expert。

---

## 2. 实验边界与完整性

正式 Router-R1：

```text
training steps = 640/640
checkpoints = 80, 160, 240, 320, 400, 480, 560, 640
validation main = 8/8 complete
validation safety = 8/8 complete
```

所有 checkpoint tensor hashes 均通过验证。以下内容保持不变：

```text
canonical bank hash =
35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db

frozen module hash =
d1d5ce232ad2aeb7a29c4c5586a6af5dcdee064321cfc94c3393d576b1bc2249

base model hash =
d8b7032a563e32f22fd51eb65d92bbb0177c913d19c5c1e6ce6ad73d0e5ca75d
```

整个诊断阶段：

- 没有训练或更新参数；
- 没有创建 optimizer 或执行 optimizer step；
- 没有重新选择 checkpoint；
- 没有修改 canonical bank、expert tensors、repository membership、generation config 或 evaluator；
- heldout、record 953 和 sealed blind set 均未加载；
- Stage-2 和 Router-R2 均未实现。

最终聚焦测试：

```text
61 passed, 7 warnings
```

O0 frozen routed parity、O4 forced-on parity，以及 deterministic no-cache/cached/HF parity 均通过。

---

## 3. 正式 checkpoint selection

Frozen forced-on upper bound 与 effectiveness floor：

| Role | Forced-on | Frozen floor |
|---|---:|---:|
| Native | 19/32 | 18/32 |
| Textual | 17/32 | 13/32 |
| Visual | 18/32 | 14/32 |
| Paired | 18/32 | 14/32 |

所有 checkpoint 的 repo-32 routed generation：

| Step | Native | Textual | Visual | Paired | Effectiveness eligible |
|---:|---:|---:|---:|---:|:---:|
| 80 | 17 | 13 | 10 | 8 | No |
| 160 | 15 | 13 | 7 | 8 | No |
| 240 | 17 | 13 | 8 | 9 | No |
| 320 | 16 | 14 | 10 | 8 | No |
| 400 | 16 | 15 | 10 | 10 | No |
| 480 | 16 | 14 | 10 | 10 | No |
| 560 | 16 | 14 | 9 | 9 | No |
| 640 | 16 | 15 | 11 | 12 | No |

Step 640 的总成功数最高，为 54/128，但 Native、Visual、Paired 分别低于 18、14、14 的冻结门槛。因此即使 safety 完美，它仍然不能成为 eligible checkpoint。

`source_validation_loss = 0.8798159862` 在所有 checkpoint 上不变。它测量 frozen generator/expert，而不是 Router 的区分能力，因此只作为 invariant diagnostic，不参与有效选模。

---

## 4. Validation safety

每个 checkpoint 均在同一组 160 个 EqKey-clean hard negatives 上评估。

| Step | Exact-S0 | False activation | Clinical/canonical failures | Target contamination |
|---:|---:|---:|---:|---:|
| 80 | 120/160 | 156 | 24 | 21 |
| 160 | 130/160 | 143 | 24 | 18 |
| 240 | **134/160** | 149 | **17** | **12** |
| 320 | 127/160 | 126 | 24 | 14 |
| 400 | 127/160 | 122 | 29 | 19 |
| 480 | 125/160 | 126 | 28 | 16 |
| 560 | 128/160 | 124 | 25 | 18 |
| 640 | 128/160 | **121** | 26 | 18 |

Step 240 在 exact-S0、clinical failures 和 contamination 上相对最好，但它的 routed effectiveness 只有 17/13/8/9，仍明显不合格。Step 640 的 false activation 最少，但仍为 121/160，也不能接受。

因此 Router-R1 同时存在：

1. positive 上的多 expert generation loss；
2. hard negative 上的过度激活和输出污染。

Safety 不能把 effectiveness-ineligible checkpoint 变为 eligible，最终没有选择任何 checkpoint。

---

## 5. 正确条件分母下的 routing retention

对每个 validation positive 定义：

```text
F = forced-on generation success
H = target expert 通过 visual hard gate
T = target expert 为 text soft top-1
R = 原始 routed generation success
```

Step 640 的条件保留率：

| Role | P(R｜F) | P(R｜F∩H) | P(R｜F∩H∩T) |
|---|---:|---:|---:|
| Native | 84.2% | 84.2% | 94.1% |
| Textual | 88.2% | 88.2% | 93.8% |
| Visual | 61.1% | 68.8% | 84.6% |
| Paired | 66.7% | 75.0% | 92.3% |
| Overall | 75.0% | 79.4% | 91.5% |

关键解释：当 forced-on 本身成功、target 通过 hard gate、并成为 soft top-1 时，原始 Router 仍保留了 91.5% 的成功样本。Visual/paired 的 hard recall 确实较弱，但这不足以解释 O0 与 O4 之间的全部差距。

---

## 6. O0–O4 Oracle 因果诊断

### 6.1 Step 640：主要诊断 checkpoint

| Oracle | 操作 | Success | 增量 |
|---|---|---:|---:|
| O0 | 原始 routed execution | 54/128 | — |
| O1 | hard gate 漏掉 target 时强制补回，保留 distractors | 56/128 | +2 |
| O2 | 只执行 target，系数保持为 O1 原始 final weight | 68/128 | **+12** |
| O3 | target-only，移除 relative softmax dilution | 70/128 | +2 |
| O4 | target-only，full strength = 1 | 72/128 | +2 |

各 role 的 O0 → O4：

| Role | O0 | O1 | O2 | O3 | O4/Forced-on |
|---|---:|---:|---:|---:|---:|
| Native | 16 | 16 | 18 | 19 | 19 |
| Textual | 15 | 15 | 16 | 16 | 17 |
| Visual | 11 | 12 | 16 | 17 | 18 |
| Paired | 12 | 13 | 18 | 18 | 18 |

First target-token rank-1 count 从 O0 的 81 增加到 O4 的 104；mean target NLL 从 1.2957 降至 0.8798。

### 6.2 Step 80：低候选数对照

```text
O0 = 48
O1 = 56   (+8 hard recall)
O2 = 68   (+12 remove distractors)
O3 = 68   (+0 remove softmax)
O4 = 72   (+4 full strength)
```

Step 80 中 hard recall 的影响更大，但删除 distractor residual 仍是最大的单项恢复来源。两个 checkpoint 独立得到相同的 primary label：

```text
PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE
```

Role-specific labels：

```text
native  = MIXED_POST_RECALL_ROUTING_BOTTLENECK
textual = MIXED_POST_RECALL_ROUTING_BOTTLENECK
visual  = PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE
paired  = PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE
```

### 6.3 当前可支持的因果结论

- Generator/expert 是有能力的：O4 精确复现 forced-on 72/128；
- visual hard recall 是次要问题：step 640 的 O1 只增加 2；
- distractor residual interference 是主要问题：O2 增加 12；
- relative softmax dilution 不是主要问题：O3 只增加 2；
- sigmoid/full-strength under-scaling 不是主要问题：O4 只再增加 2；
- 不能把 54/128 到 72/128 的差距归因于单一训练不足；当前执行结构本身会让多个 expert residual 相互破坏。

---

## 7. Gradient conflict 与 feature-scale 诊断

只读梯度诊断覆盖：

```text
router states = step 0, 80, 320, 640
repository sizes = 1, 4, 8, 16, 32
```

结果：

- 没有创建 optimizer，没有 optimizer step；
- 每次测量后 gradients 均重置为 `None`；
- 所有 checkpoint hashes 前后不变；
- `input_extractor.text_branch` 的 positive-vs-negative mean cosine 为 -0.0985，20 次中 13 次为负；
- 但按预注册规则，这不足以判定 persistent gradient conflict；
- 其他 parameter group 的对应测量退化为零梯度，也没有 persistent conflict 证据。

因此，目前没有足够证据把失败归因于“positive recall 与 negative suppression 的稳定梯度冲突”。

Feature-scale 诊断认为 score/candidate 行为变化主要来自：

```text
angular_separation_drift
```

而不是单纯的 feature norm 增长。Repo-32 mean candidate count 在诊断 batch 上从 step 0 的 2.0 变为 step 640 的 1.5；该 counterfactual 是描述性分析，不是逐样本精确干预。

---

## 8. 建议的 Router-R2 方向

根据 Oracle，第一优先级应是：

```text
Sparse / Top-1 Expert Execution
```

更完整的候选结构是：

```text
Null-aware Selective Top-1 Routing
```

概念上：

1. Router 在 `experts + NO_EDIT` 中选择；
2. top-1 expert 必须同时超过 `NO_EDIT` 和 runner-up 的训练集校准 margin；
3. 只执行这一个 expert，禁止多 expert residual 求和；
4. execution coefficient 应作为独立消融：先验证“保留原系数的 target-only”，再比较 sigmoid/full-strength；
5. 所有阈值和 margin 只能在 clean-train/calibration 上确定，不能使用 validation Oracle、heldout 或 record 953 调参。

为什么不能只做普通 top-1：当前 hard-negative false activation 为 121–156/160。如果没有 `NO_EDIT`/拒识门，top-1 会进一步放大 locality 与 contamination 风险。

为什么不优先修改 softmax 或直接 full-strength：step 640 中 O2 恢复 12 个，而 O3、O4 各只恢复 2 个。当前最强证据首先支持“删除 distractor execution”，然后再研究执行强度。

---

## 9. 希望 GPT Pro 判断的问题

1. 上述 O0–O4 结果是否足以把 Router-R2 主线确定为 `NO_EDIT + selective top-1 + single-residual execution`？
2. 面对 121–156/160 的 hard-negative false activation，最合适的训练集内拒识目标是什么：显式 `NO_EDIT` prototype、energy/margin loss，还是 target-vs-sentinel/runner-up 双 margin？
3. O2 已恢复大部分成功，而 O3/O4 增益较小。Router-R2 是否应先保留原 final coefficient，仅改变执行稀疏性，再把 full-strength 作为独立消融？
4. Feature 诊断指向 angular separation drift，但梯度诊断没有 persistent conflict。是否应该加入 cosine-normalized key、angular margin 或 supervised contrastive routing objective？如果加入，怎样避免改变过多组件而失去与 LiveEdit 的可比性？
5. 对论文/复现报告，是否应将当前结论表述为：`expert/generator reproduction passed, end-to-end multi-expert Router-R1 gate failed`，而不是笼统称为 LiveEdit reproduction failure？
6. 下一阶段是否应继续严格限制在 clean-train/calibration，只有 Router-R2 同时通过 effectiveness 与 safety 后才允许一次 heldout evaluation？

---

## 10. 当前权限与未完成工程项

```text
Router-R2 implemented = No
heldout permitted = No
record 953 permitted = No
sealed blind permitted = No
Stage-2 permitted = No
```

实验、Oracle、梯度/特征诊断和最终报告均已完成。当前唯一未闭环的工程项是：

```text
source_commit_sha = PENDING_SOURCE_ONLY_COMMIT
```

在继续新实验前，应先完成 source-only commit，并把 exact Git SHA 回填到 final manifest；不得上传数据、图片、cache、checkpoint、expert tensor、router tensor、outputs 或 blind-set 内容。

## 11. 权威产物位置

正式 Router-R1 run：

```text
/remote-home/wangbomin/Knowledge_editing/outputs/
liveedit_med_eqkey_clean_fast_confirmation_v1/20260815T122907Z/
router_r1/schema_fix_recovery_01/formal_recovery_02_20260816T071851Z
```

Oracle 最终 recovery：

```text
/remote-home/wangbomin/Knowledge_editing/outputs/
liveedit_med_router_r1_oracle_diagnosis_v1/
20260816T143038Z_selector_recovery_02
```

优先查看：

```text
LIVEEDIT_ROUTER_R1_ORACLE_FINAL_DECISION.md
oracle_diagnosis_summary.json
r1_safety_closure/ROUTER_R1_FINAL_SELECTION.json
r1_safety_closure/safety_aggregate.json
mechanism_attribution/sequential_gain_decomposition.json
gradient_conflict/conflict_aggregate.json
feature_scale/candidate_inflation_final_analysis.json
router_r2_recommendation/ROUTER_R2_RECOMMENDATION.md
focused_test_report.json
run_manifest_final.json
```
