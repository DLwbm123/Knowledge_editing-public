# 独立补丁使用与等价边界

该补丁只新增文件，不修改或自动导入任何正式训练入口。`equivalent_reuse.patch`可独立应用；真实回放的基线模块须解析到已冻结生产源码，不能混用旧工具目录。下面的context必须由独占该worker的调用者显式开启：

```python
from equivalent_reuse import correction_reuse
with correction_reuse(original, pure) as reuse_counts:
    selected, decision = original.correction(
        runtime, hook, expert, before, pure, step, sources, losses,
        cache, cpu_cache, budget, 'FISHER_MIN')
```

缓存只活在本次correction内部，异常时也恢复原函数；不写进checkpoint，不跨科学步或owner，不供多线程共享。模型必须eval，纯候选和source batch保持冻结，Fisher缓存固定。首次pure loss实际调用须验证Python/Torch/CUDA RNG未推进；命中时仍执行原权重加载和teacher routing副作用。

候选规范化/Fisher行矩阵复用要求输入dtype、device、shape、stride及张量原始字节一致（包括signed zero）；Fisher行矩阵还绑定缓存张量身份、version和token_counts。无梯度图的Fisher标量仅按原对象身份与tensor version复用并保留输入引用，需梯度的Parameter惩罚始终重算。缓存返回独立结果，防止下游原地修改污染后续调用。

逐位验证覆盖选中/pure权重、clip前后源梯度、每次修正的anchor/protection梯度、最后残留梯度、两标量RMS、Python/Torch/CUDA/NumPy RNG、hook状态和完整原决策字典。减少的前向/算子计算按实际记录，不要求物理调用计数逐位相同，也不冒称执行了被复用的调用。

不执行推测性剪枝。原4 fraction、3 trust纠偏、MULTI/root fallback、finite检查、0.1C/90%及逐步匹配均保留。完整前5步两例回放通过不等于证明全部146轨迹或科学性能；结果必须注明样本范围和墙钟测量边界。当前正式GPU0/1链继续原版，补丁验证不构成上线或重启授权。
