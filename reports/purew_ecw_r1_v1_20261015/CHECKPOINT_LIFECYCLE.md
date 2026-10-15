# R准备状态与未来运行生命周期

当前仅实现和CPU准备，无新RUN、无INSTALL_ATTEMPT、无GPU激活、无scorer或tail、无监测。CPU使用/data/bmw/tmp中性临时目录和独立小型合成状态，未改动任何历史运行。所有CPU失败与成功回执保存在本地.local_run/ecw_r1_preparation_20261015；不计为真实VLM准入。

未来执行必须使用独立R身份及新RUN，不能以旧J内部kernel标签冒充新结果或写入旧J目录。生产kernel继承J的state格式和完整q/v/B/数值包络；备用求解不新增持久化状态，只有当前窗临时参数梯度与小型证书，当前窗结束后退役。实际native/master、RNG/NumPy、时钟、末事件与全部调用账本必须在20槽、源末、评价前和协作停止保存。无硬中断重算余量，不盲重放或重装。

末端87输出持久化、最后W新进程2题精确重载、FINAL_CONSUMED及GPU租约关闭后才允许评分。每新payload一次；原J失败不补判，所有实际消费者终态后公开报告。只有新R统计完整消费并已精确重载后，才按获批代码退役；历史资产与原146/M恢复点不可动。CPU准备不授权任何清理或GPU启动。


## Authorized execution override

Latest direct user authorizes this R-only independent run and30-minute monitoring. Execution/storage/VLM gates are implemented by the R-specific chain; preparation-only flags remain historical. R native/master checkpoints carry external R identity and internal memory arm J, validated together, with at most38 rows. First-source success also saves final native and emits all87 evaluation consumers; no source2. Only R active statistics may retire after exact fresh reload and FINAL_CONSUMED_R. Failures and old J native/statistics remain immutable.
