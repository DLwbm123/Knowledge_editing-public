# 真实准备启动回执

**兼容性通过，Fisher 与层选择后台运行中；不是完成报告。**

复用原生 FP16 LLaVA-Med-v1.5-Mistral-7B。真实图像/零像素配对结构检查通过，两支非零梯度均为4096×14336，原生训练样本生成19 tokens，兼容性检查未改变权重。远端 CPU 路径也通过。见 [COMPATIBILITY.json](COMPATIBILITY.json)、[CPU_CHECK.json](CPU_CHECK.json)。

新固定32个 SLAKE 训练图像组，与官方 validation/test 和现有编辑评估图像组均无交集；不是复用旧32条清单。旧清单与官方 heldout 的交集已在 [DATA_AUDIT.json](DATA_AUDIT.json) 保留。没有以评估结果选层或选样。

准备任务 `vigor-medical-preparation-v1-20261018` 的 attempt2 在 pro5000 GPU5 启动，中性进程名和 GPU 进程显示已核实。启动检查时2/32层完成，第三层正在计算 E；第一层结果没有触发提前选层，仍执行全部32层。最终选择和保存重载后的一致性检查尚待完成。该进度是 [STARTUP.json](STARTUP.json) 的快照，不是持续更新状态。

首次尝试因入口缺少既有模型/视觉塔路径，在加载前失败，保留3.239559秒耗时。确认零校准、进程已退出后补齐绑定并从独立尝试目录继续，未覆盖失败回执。当前没有正式编辑权重更新、Judge 或新增监测；后台准备结束后自动停止，不会转入编辑实验。

原始 worker 的 `visual_token_count=1` 实际取自 runtime 的图像占位符数量；公开兼容性回执已明确重命名为 `image_placeholder_count`。后续源码同时报告展开视觉 token 数；本次没有为修正文案重启正在运行的 worker。
