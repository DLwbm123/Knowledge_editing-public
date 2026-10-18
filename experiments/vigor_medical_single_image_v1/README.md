# VIGOR 医学单图适配 v1

状态：**仅实现与合成 CPU 验证；未启动实验。** 依据 *VIGOR: Editing What the Video Contributes*（ICLR 2027 under review）的 Eq. (9)–(14)、Appendix A.5 重建。未取得作者补充代码，不称为官方实现或严格复现。

## 实现范围

独立 `Editor` 直接更新一个语言骨干 FFN `down_proj.weight`；其余参数冻结。没有 Adam、额外编辑器网络、推理 adapter、路由器、旧编辑回放或现有 H-loss/保护器依赖。既有编辑器和历史协议不变。

每步计算视觉输入与 Visual-Null 的同一目标序列平均 token NLL，重新求当前参数的两个梯度。设 `F = diag(b) ⊗ diag(a)`，按权重 `[out, in]` 布局计算：

```text
iv = F^-1 gv; i0 = F^-1 g0
t = iv - (gv^T i0 / g0^T i0) i0
E = t^T F t
d = max(Lv - 0.01, 0)
d_eff = min(d, sqrt(2 * 1e-3 * E))
delta = -(d_eff / E) t
```

最多五步，达到 `Lv <= 0.01` 时提前停止。预算不足时缩短一阶目标下降量；不会增加预算、优化步或重试。退化梯度/近共线剩余方向不更新，并返回原因。非有限值或不稳定投影报错且恢复本次编辑前的权重。Visual-Null 的约束是局部一阶标量损失约束，不保证有限步 null 输出不变，更不保证旧医学编辑保留。

## 与论文的对应和差异

| 项目 | 论文 | 本适配 |
|---|---|---|
| 模型与输入 | Video-LLM / VMEB 视频 | 复用现有 LLaVA 医学单图 runtime；仅一个 `[1,C,H,W]` 图像张量，不支持视频、多图或多裁剪网格 |
| Visual-Null | 视频预处理后像素置零，保留结构 | 图像预处理后 `zeros_like`；复用原始 token/标签，检查展开后的长度、dtype、位置、mask、标签一致。不是原始黑图片，也不是零视觉 embedding |
| 配对构建 | 官方 VMEB 路径 | 复用 `build_edit_batch`、`adapter.prepare_inputs` 和 `_expand_multimodal`。接口未暴露缓存像素，故第二次预处理并额外验证原图展开与第一次完全一致；每对有三次视觉展开，防止随机预处理混入对照 |
| Fisher | 训练集输入/输出梯度二阶矩，固定对角 Kronecker 因子 | 同结构；明确用原图 mean-target-NLL 的反传梯度，所有非 padding 位置的逐特征平方均值。论文未充分说明 token 聚合和梯度归一化，本实现固定此口径，不声称与作者数值相同 |
| 阻尼 | 相对 `1e-3`、floor `1e-12` | 各因子 `max(moment + 1e-3 * mean(moment), 1e-12)`；具体相对尺度解释显式固定 |
| 层选择 | 训练集 E profile 固定层 | 候选 FFN 层的训练样本平均 E 最大者，平局取传入顺序首项；不沿用其他骨干层号。医学候选集与真实层号尚未校准 |
| Trust region | 每步 Fisher 预算 `1e-3` | 将预算解释为 `0.5 delta^T F delta <= 1e-3`，缩短 d；论文没有提供详细截断代码，因此标为实现解释 |
| 精度 | W、梯度和 Fisher 为 FP32，其他 native | 单次编辑内部 FP32 W + 临时输入/输出 cast hooks；结束后将 W 转回原生 FP16/BF16/FP32，移除 hooks，再测 native NLL。量化是明确差异；不把 FP32 约束说成 native 权重的精确保证 |
| 数值退化 | 推导假定有效非零几何 | 零梯度停止；`E <= 1e-6 kvv` 停止；投影相对残差大于 `1e-4` 报错回滚。这些是本实现的数值防护 |
| 评估 | 独立编辑恢复；顺序100次后 Video-MME | 接入现有医学 T0/T1G/T2G、旧编辑保留和保护集评估入口；本次未运行。独立/顺序由调用方明确选择，不默认100或146条 |
| 随机数 | 默认 seed42 | CPU夹具 seed42；编辑器不改全局随机状态或现有实验种子；未来真实协议单独冻结 |

## 复用方式（示意，不是启动脚本）

`runtime` 是已加载的现有 `LlavaMedEditorRuntime` 兼容对象，`record` 使用现有 `EditorRecord`。代码不创建模型、下载数据、访问服务器或调用 Judge。CPU 夹具直接读取现有 `PreparedBatch` 类型验证接口。

```python
from experiments.vigor_medical_single_image_v1 import Editor, calibrate

# 必须在另行授权后，由已核实的训练/评估清单提供；不可拿评估集选层。
binding = dict(model_revision=model_revision,
               dataset_revision=dataset_revision,
               preprocessing_revision=preprocessing_revision)
calibration = calibrate(runtime, train_records, candidate_ffn_layers,
                        train_ids=train_manifest_ids,
                        excluded_ids=all_edit_and_eval_ids, binding=binding)
editor = Editor(runtime, calibration, binding=binding)
base = editor.state_dict()
receipt = editor.edit(record)
# 随后调用原有 runtime.generate / 现有消费者评估流程；生成设置不变。
# 独立编辑：editor.load_state_dict(base)。顺序编辑：保持当前 W。
```

`binding` 应绑定基础模型、数据清单和预处理版本；显式清单交集检查不能替代数据来源审计，也无法识别不同 ID 下的同图/同病例泄漏。校准拒绝非训练 ID、排除 ID 和重复 ID；编辑拒绝校准 ID。真实训练清单及病例级拆分尚未提供和审计。

`state_dict` 只包含所选层原生 W、固定 Fisher 因子、版本和绑定信息；由调用方使用既有持久化流程保存，加载应使用 `torch.load(..., weights_only=True)`。它不是完整模型，也不是跨任意模型的迁移格式。固定其他模型权重、生成设置、运行计数与随机状态仍由既有执行器管理；当前没有后台作业执行器或中途逐步恢复协议。单次编辑异常原子回滚；成功返回后的 W 可以直接由原生模型生成。Fisher、master 和梯度均为编辑期资源，不增加部署参数。

## CPU 检查与当前边界

从仓库根目录运行已有含 PyTorch 的 Python，通过 stdin 避免进程命令行暴露方法名：

```sh
python - <<'PY'
import json
from experiments.vigor_medical_single_image_v1.cpu_check import run
print(json.dumps(run(), indent=2))
PY
```

检查包括独立 dense KKT 解对照、Fisher 预算和 null 一阶约束、退化/NaN 拒绝、预处理后零像素与结构守恒、结构破坏拒绝、训练集隔离、校准不改 W、目标 token NLL、梯度重算、已满足请求不更新、原生精度写回、无 hook 生成、状态保存重载及更新后异常回滚。覆盖 FP32/FP16/BF16 合成 CPU 模型；不是医学模型的收敛或低遗忘证据。

真实模型加载、真实训练 Fisher/层选择、显存与实际算子兼容、生成/评分和后台启动全部 **NOT_RUN**。本目录不包含启动入口，不修改旧实验授权，不恢复旧作业或新建监测。
