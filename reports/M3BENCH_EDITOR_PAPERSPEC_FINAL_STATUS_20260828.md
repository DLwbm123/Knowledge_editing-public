# M3Bench V4 Editor Paper-Spec Runtime 与 Smoke Gate 最终状态报告

- 报告日期：2026-08-28
- 冻结完成时间：2026-08-28 06:54:37 UTC（北京时间 14:54:37）
- 报告范围：Editor paper-spec reimplementation 的运行时可行性、8-record smoke、4-edit mini-stream 及 fresh-process replay
- 最终状态：**PASS**

## 1. 一句话结论

LoRA、GRACE、BalanceEdit 和 BELoRA 四种 editor 均已通过运行时 preflight、8-record 独立 smoke，以及冻结 4-record 顺序 mini-stream 的 run/fresh-process replay。最终 gate 已冻结为：

`M3BENCH_EDITOR_SMOKE_PASS__PAPER_SPEC_REIMPLEMENTATIONS`

该结论证明当前四种实现可以运行、保存、重新加载并按冻结协议复现，但**不等价于正式 editor 效果比较已经完成**。Formal single-edit、formal 200-edit sequential、prefix evaluation、post-edit full Judge 和 formal comparison 仍未获授权、也未启动。

## 2. 冻结运行环境与实现身份

| 项目 | 冻结值 |
|---|---|
| 服务器 | `my-gpu` |
| 授权物理设备 | GPU2，NVIDIA A100-PCIE-40GB |
| Worktree | `/remote-home/wangbomin/worktrees/m3bench_editor_paperspec_runtime_v1_20260828T032447Z` |
| Run root | `/remote-home/wangbomin/Knowledge_editing/outputs/m3bench_editor_paperspec_runtime_v1/20260828T032447Z` |
| 原始实现 commit | `28037d407781d36e1162b9f556ead62022b1c058` |
| GRACE 恢复 commit | `27d7cdf66bfb727ab9089e50eeae931052a1a83c` |
| 实现分类 | `M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V1` |
| BELoRA 身份 | `AUTHOR_IMPLEMENTATION_NOT_AVAILABLE_FOR_BELORA` |
| 最终工作树 | clean，位于恢复 commit |

BELoRA 当前是依据论文规格完成的独立重实现，不能在论文中表述为作者官方实现复现。

## 3. Gate 结果

### 3.1 Runtime preflight

四种方法均完成独立 single-run 和 fresh-process single-replay：

| 方法 | Single-run checks | Single-replay checks | Base weights | 峰值 allocated / reserved |
|---|---:|---:|---|---:|
| LoRA | 9/9 PASS | 7/7 PASS | unchanged | 19.12 / 19.72 GiB |
| GRACE | 9/9 PASS | 7/7 PASS | unchanged | 14.36 / 14.61 GiB |
| BalanceEdit | 9/9 PASS | 7/7 PASS | unchanged | 15.37 / 15.73 GiB |
| BELoRA | 9/9 PASS | 7/7 PASS | unchanged | 14.46 / 14.70 GiB |

Runtime gate：`M3BENCH_EDITOR_RUNTIME_GATE_PASS__PAPER_SPEC_REIMPLEMENTATIONS`

### 3.2 Approved 8-record smoke

- 每种方法：8 records
- 每条记录：独立 run + 独立 fresh-process replay
- 总模型进程：`4 methods × 8 records × 2 = 64`
- 结果：`64/64 PASS`
- 所有记录均满足 run checks、replay checks、reset parity 和 base-model unchanged

8-record smoke report SHA-256：

`710ef89bba289ba846963c20fc013393fb058374133956f5ed465dd7815bf3b8`

### 3.3 Approved 4-edit sequential mini-stream

冻结顺序为：

1. `m3bench-v2/SLAKE/xmlab399/xmlab399_11`
2. `m3bench-v2/SLAKE/xmlab502/xmlab502_6`
3. `m3bench-v2/VQA-RAD/synpic42951.jpg/1940`
4. `m3bench-v2/VQA-RAD/synpic47191.jpg/1896`

| 方法 | 4-edit stream-run | Fresh replay | 状态合同 | Base weights |
|---|---:|---:|---|---:|
| LoRA | PASS | 4/4 exact | 单一 adapter 连续优化；edit history 精确为 4 | unchanged |
| GRACE | PASS | 4/4 exact under amended contract | 4 个独立累积 entries | unchanged |
| BalanceEdit | PASS | 4/4 exact | 4 个独立累积 entries | unchanged |
| BELoRA | PASS | 4/4 exact | 4 个独立累积 entries | unchanged |

所有方法同时满足：

- 冻结记录顺序完全一致；
- 每一步 checks 全部为 true；
- 每一步 checkpoint 非空；
- fresh-process replay exit code 为 0；
- replay 的 4/4 records 完全满足各自冻结比较合同；
- base-model hash 前后不变；
- run/replay 输出目录互相隔离。

## 4. GRACE replay 协议修订

原 GRACE 4-edit stream-run 成功，但第一次 fresh replay 因两个 SLAKE route radius 在 checkpoint 序列化前后分别出现约 `1.32084e-9` 的差异而失败。审计显示：

- raw token IDs：4/4 exact；
- decoded text：4/4 exact；
- route 中除 radius 外的字段：4/4 exact；
- routing identity、activation、nearest distance 和 entry count 均未变化；
- base weights 未变化；
- 两个差异值转换到 IEEE-754 binary32 后均完全相等。

经 operator 授权，协议采用严格的 float32 canonicalization：比较前双方均转换为 IEEE-754 binary32，再执行 exact equality。**没有引入任意数值容差**，其余字段仍要求 exact match。

恢复 replay 最终满足：raw tokens 4/4、decoded text 4/4、non-radius route fields 4/4、float32 radius 4/4、base unchanged，状态为 PASS。

原失败 replay、原 checkpoint 和原 blocked marker 均保持只读、未覆盖。

## 5. LoRA 最终汇总器假失败修正

第一次最终汇总报告错误地将 GRACE/BalanceEdit/BELoRA 的 `entry_count=4` 规则套用于 LoRA，因此生成了一次 aggregator-only FAIL。

LoRA 的 paper-spec 状态模型是持续优化同一个 adapter，并不为每个 edit 创建独立 entry。LoRA 的正确状态合同为：

- exact 4-record edit history；
- adapter parameter count > 0；
- adapter parameter bytes > 0；
- target count > 0；
- `entry_count` 不适用。

原 LoRA stream-run、fresh replay、4/4 replay exact、base unchanged 等实际结果从未失败。此次修正仅影响最终报告汇总逻辑：没有启动模型、没有重跑 LoRA、没有修改 generation、checkpoint、cohort、evaluator 或任何模型产物。

原 aggregator-only FAIL 报告仍保留用于审计。

## 6. 协议修订与审计文件

| 文件 | SHA-256 |
|---|---|
| `GRACE_ROUTE_RADIUS_FLOAT32_PROTOCOL_AMENDMENT.json` | `ae433540b2e8c66e1017872b354f45ad28de3cc2557605cb2eb8964a33c38fbb` |
| `GRACE_ROUTE_RADIUS_FLOAT32_CODE_LOCK.json` | `f6f90f4b8eb06e0c5b88e7dadd2767cc3e3105703615c0241c4639dc0b23b212` |
| `REMAINING_METHODS_PARENT_COMMIT_EXECUTION_AMENDMENT.json` | `51c924f36a305bd263e39d12604e1257c722c1f4898cc0c960c6c4bc6cb8802a` |
| `FINAL_AGGREGATOR_LORA_STATE_CONTRACT_FIX_AUDIT.json` | `182cbc75ebe1d288c7f7605f4f7a8200481f9435fafd4af7b69d0ce69c8741ed` |
| 最终 aggregator 脚本 | `526e5277652d30a9ea4731a17083aa7744646132d9c4a9a1f6fc53f57c3c8437` |

BalanceEdit 与 BELoRA 在原冻结 commit `28037d...` 下执行；其运行语义、cohort、prompt、evaluator 和方法配置未改变。恢复输出使用独立目录，未复用或覆盖失败/部分产物。

## 7. 最终冻结产物

| 产物 | SHA-256 |
|---|---|
| `M3BENCH_EDITOR_SMOKE_GATE_REPORT_AMENDED_V2.json` | `7dbc861637ab068bffcaa2b26fdc8a9368694d093560a2df082bda222f05eaf2` |
| `M3BENCH_EDITOR_SMOKE_GATE_REPORT_AMENDED_V2.md` | `16f24346d04018b44c66cd7f9acff9efb7f8fa61877213dad4bb1be9a78bd727` |
| `M3BENCH_EDITOR_SMOKE_PASS__PAPER_SPEC_REIMPLEMENTATIONS` | `3280aca0791a87384dccf8227df2bbb3af68e276f5dc2548b52db0ac6acaeb6e` |
| `STOP__FORMAL_EDITOR_EXPERIMENTS_REQUIRE_NEW_AUTHORIZATION` | `2af183be253fd571e13dd3d310357f3a1ccc7b18502a83ed918e0b88a1e5355f` |
| Runtime gate report | `da19892f1c001edc75c5828f7092aef72c8d4caea29d287c0871559a74cd7319` |

最终 JSON 报告远程路径：

`/remote-home/wangbomin/Knowledge_editing/outputs/m3bench_editor_paperspec_runtime_v1/20260828T032447Z/M3BENCH_EDITOR_SMOKE_GATE_REPORT_AMENDED_V2.json`

## 8. 保留的失败审计证据

以下历史失败均未删除或重写，也不代表当前方法 gate 失败：

| 历史事件 | 性质 | 保留 SHA-256 |
|---|---|---|
| 原 GRACE replay FAIL | float32 序列化边界导致的 exact-radius mismatch | `8e96768e6bba2c60bcc38eb486e2de807c0454ea5c121f7f802b6a96289f8780` |
| 原 GRACE blocked marker | 与上项对应的原始 hard-stop | `fcc490333f45fe6116a01bc3ee0226ea9d5e3b483bcc1c3de2f89655a54c2d4b` |
| BalanceEdit 首次 pre-edit stop | 全局 provenance 与恢复 commit 冲突；模型尚未启动 | log `d6952f791192166026f21b36723ca50b42860f797ebc6cb1dd8ec462f9f63013` |
| 第一次最终 aggregator FAIL | LoRA 状态合同汇总错误；非模型失败 | `32e7cb77686d87554bf34f4ec0da0454b6c675321144fc973418164988558582` |

## 9. 当前仍然生效的边界

- `formal_experiments_authorized = false`
- `JUDGE_AUDIT_PACKET_PENDING_SIGNOFF`
- 未运行 formal single-edit
- 未运行 formal 200-edit sequential
- 未运行 prefix 1/50/100/200 evaluation
- 未运行 post-edit full Judge
- 未运行 formal editor comparison
- 未访问 validation raw、heldout、record 953、sealed blind、Stage-2、T5 或 PadChest-GR
- GPU2 在最终核验时为空闲状态，无本项目计算进程

## 10. 建议 Pro 决策的下一步

1. 确认是否接受 GRACE 的 IEEE-754 float32 canonicalization 协议修订；该规则不使用自由容差。
2. 确认 LoRA 的方法特定状态合同，即连续单一 adapter，而不是四个独立 entries。
3. 如认可当前 smoke gate，可另行明确授权 formal editor workflow，并继续保持 gate 顺序：formal single-edit → formal 200-edit sequential → prefix evaluations → post-edit Judge → formal comparison。
4. 在论文和实验表中将 BELoRA 明确标注为 paper-spec independent reimplementation，而不是作者官方实现。
5. 单独完成并签署 200-record human audit packet；当前 smoke PASS 不豁免该要求。

## 11. 结论

当前可以正式陈述为：**四种 paper-spec editor 实现均已通过 runtime、8-record smoke 和 4-edit sequential replay gate，Editor Smoke Gate 已冻结 PASS。**

当前尚不能陈述为：**四种 editor 的正式 M3Bench 性能实验或最终论文比较已经完成。**
