# 有效区ignore-mask独立CPU覆盖

只补生产接口上轮明确未覆盖的有效区ignore-mask，不重跑上轮CPU、Q1、来源普查或任何优化。先冻结本文件/FREEZE再执行，一次已有CPU环境，远端0文件；CUDA_VISIBLE_DEVICES空、单线程，signal30秒/client45秒，整体1800秒，本机1MiB。最多8合成forward/3梯度路径，无VLM/GPU/Judge/生成/端点。

使用既有coefficient_runtime与原editor.nll代码，不改实现。固定6输出9输入Linear，系数2×2，确定性sin/cos QR基；5位置输入，labels=[-100,1,-100,3,-100]，因果shift后只有位置0和2有效。对比真实nll与只取这两个位置的显式交叉熵及系数梯度，损失atol1e-7/rtol1e-6，梯度atol1e-8/rtol1e-5。将被忽略预测位置1和3的logits固定加不同类别偏置，损失和梯度必须不变；保留位置4未被因果使用。空目标batch必须在前向前拒绝。检查正常与异常退出原权重/dtype/flags/modes/forward/grad恢复。

不执行优化或双可行性检查；FAIL保留并停止，不改输入/容差或重复运行。PASS只关闭有效ignore-mask CPU覆盖，真实VLM与KV/native整链、数据用途仍未准入。双成功为后续H1结果，非启动前提。原上轮缺失记录保持，不回填旧16项检查。无新来源证据，不再查同池。
