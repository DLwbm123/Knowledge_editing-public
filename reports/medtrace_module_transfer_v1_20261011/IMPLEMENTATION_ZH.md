# 实现与验证边界

新增实现集中在 `experiments/medtrace_module_transfer_v1/`，复用既有冻结 runtime、TT 类、原路由、生成、Fisher probe、函数步长匹配和隔离评分工具。没有新增依赖，也不修改历史实验实现。

- `module_math.py`：TT 确定性规范化、标量 RMS 状态、七目标 KKT 投影、组合算子 Fisher 惩罚。
- `module_run.py`：有限六臂训练/评估链、原 C 精确复现屏障、计数/预算/活动恢复状态/最后消费者清理。
- `module_route.py`：固定 R0 八专家库及完整输出身份复用。
- `module_queue.py`：生成前等待、完整身份去重、一次评分、永久缺失继承与失败停止。
- `module_report.py`：所有预登记比较、逐 owner 和版本分层、自然/强制结果、双重配对区间、成本和机制统计。
- `module_check.py`：CPU 数值及控制逻辑检查，读取原生 TT 状态；不加载 VLM，不调用 Judge。

CPU 检查覆盖零根和非零根、三个一般可逆 bond 变换、三步标量优化器状态、相互冲突/重复/零目标、显式 dense 算子的惩罚值和梯度。真实 FP32 7168 参数状态的规范化相对输出误差：BASE=0，FINAL约 1.05e-6。七梯度加权累积与原 cyclic 源梯度一致；有限检查的接受和拒绝分支通过；三 GPU 分区与原基线可读性通过。详情见 `CPU_ADMISSION.json`。

CPU 通过不是实际模型训练通过，也不是模块提升证据。真实首个 Fisher 的链式梯度一致性、C 的精确复现和各臂生成均由后台屏障检查。M1 是本项目对文献机制的适配，不声称复现 TTN 的完整 Riemannian Adam；M2 只有源侧 NLL 非增加约束；M3 只有 Base 处的有限 probe 近似。

本轮未添加视觉干预、DPO、额外监督数据、rank 扩展或新路由器；未扩展至独立数据、额外种子或全 146 编辑。
