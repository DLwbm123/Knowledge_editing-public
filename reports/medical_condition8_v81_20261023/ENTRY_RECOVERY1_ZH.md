# 首次入口修复（零模型工作）

首轮两worker在import旧工具时因RUN_ROOT缺少历史private/tools/train.py失败；0模型forward、0目标生成、0反向和0训练更新。失败退出、日志、PID/start_ticks与已消耗GPU进程驻留秒保留，不重置原始时钟/预算。独立guard继续原进程，不停止或重启。

修复仅将旧模块的RUN_ROOT绑定到既有只读父工具上下文；本轮所有输出、计数、checkpoint、目标、状态由独立V81_ROOT控制，预算入口覆盖到本轮guard。父ledger和预算不写入，不恢复旧controller。先做CPU导入与读取路径检查，再允许首次模型加载。原科学矩阵、目标采样和门槛不变。
