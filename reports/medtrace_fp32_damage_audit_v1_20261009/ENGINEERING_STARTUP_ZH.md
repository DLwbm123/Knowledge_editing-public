# 评分入口启动修复

首次4个本机评分器在首次队列 `reserved` 查询时退出：远程 `runpy.run_path` 入口未将 private/tools 加入模块搜索路径，报 `ModuleNotFoundError: astra_queue`。失败发生于任何 payload 预约或 Judge 调用之前，原日志和启动回执保留。

按既有 optimizer_queue 入口补齐 sys.path 后，同一 reserved 查询成功返回空列表；随后重新启动尚未尝试任何 payload 的评分器。无 GPU 重启、无训练或生成变更、无预算/科学协议变化、无语义或传输重试。GPU源码仍为 d1cbd44，评分入口修复单独记录其提交。
