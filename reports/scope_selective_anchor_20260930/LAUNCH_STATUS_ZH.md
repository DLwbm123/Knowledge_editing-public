# 已启动：选择性正例锚定实验

用户已授权取消本轮内部 Judge 次数上限，累计尝试不重置。Judge 为 GPT‑6.1 sol / high，模板不变，单独评分缓存，候选和所用参考统一评分。先前 WAITING_FOR_JUDGE_RESERVATION 已解除。

独立运行目录：`/data/bmw/Knowledge_editing/outputs/scope-selective-anchor-20260930/run`。三个 resident worker 使用物理 GPU 5/6/7，中性入口 `/tmp/v3.py`；有限控制器 `/tmp/v4.py`；本地评分入口 `/tmp/v5.py`，评分日志 `/tmp/ss2/score.log`。

首个完整 pilot 为固定8编辑×5训练分支，共40条续训、每条80步；另有4档缩放CHECK（gamma=1严格复用H）。保守评分预留2620次，未知去重不打折。pilot完整结束后，只按CHECK锁定共同beta和gamma，补齐DEV剩余48条；DEV至少一个预注册候选明确通过联合门槛，才准入完整REG72条和固定缩放对照。失败/不确定不会当作通过。

后台有限流程不依赖当前聊天持续开启，不恢复小时监测。评分失败键不自动重试；已成功分数不重复判。新阶段默认存储soft/hard为10/20GiB，原始实验和共享模型只读。原35个Base-correct资格继续冻结，同时另报Sol的Base判分，避免偷偷改变分母。

启动不等于实验完成。开始时三张卡均实际占用约15GiB，三个训练任务正在运行；评分首批已启动。当前结果以LAUNCH_RECEIPT.json和实时运行目录为准。实验结束后再核验结果、完善解释并按公开边界推送独立GitHub分支/PR。
