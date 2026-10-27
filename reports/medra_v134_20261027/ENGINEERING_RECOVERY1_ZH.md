# MED-RA 曲率回执工程修复

首次GPU执行已通过真实mask有限差分检查，并保存第1个医学曲率锚点；随后因回执代码未导入time而退出。退出时累计6次逻辑forward、36次backward、4次operator pullback和4次vision forward；source校准、缩放选择、正式生成、Judge调用、新梯度训练更新均为0。这是工程失败，未产生方法效果结论。

修复仅补充time导入，并初始化两个断点恢复分支可能跳过赋值的临时变量。原始失败、退出记录、源码版本及累计成本保留；第1个锚点复用，后续从未完成项继续，原输入绑定与24小时时钟不变。原科学源码版本仍记录为初次冻结提交，补丁提交另外记入工程恢复回执。

恢复工具为 `experiments/medra_v134/recover.py`。中性入口由私有部署配置绑定；不把项目路径或方法名称放进实际进程参数。

```sh
python /private/tmp/q.20261027.134/r.py
APPLY_RECOVERY=1 python /private/tmp/q.20261027.134/r.py
```

第一条只检查；第二条只适用于本次尚未评分的缺失导入故障，执行前核实旧GPU进程PID与start_ticks均已退出、本机旧保护器/评分器均不存在、原时钟尚有余量、指定GPU显存和UUID符合原准入。完整旧记录移入本次engineering_recovery1目录，保留WORKER_ONCE/GUARD_ONCE，写RECOVERY_REQUEST后才启动。它拒绝运行中重复启动、资源STOP、已有恢复记录或已开始Judge的情况；不是通用重试命令。

恢复后仍须核对完整argv、nvidia-smi显示、健康回执、锚点数实际增加，以及失败/退出标志。通过启动检查不等于完整31锚点校验通过；随后仍以原single筛选决定是否执行sequential。硬截止保持北京时间2026-10-28 14:15:35，监测关闭。
