# null一阶分解：优先复用已有轨迹

原冻结协议来自上一轮NEXT_MINIMAL_ZH.md。本次实现前确认diagnose.candidate在同一可编辑层FP32上下文使用gradient_pair及实际未舍入权重增量actual调用update_metrics，已有fp32_update.null_linear_change即g_null·u，且fp32_before/after及native_before/after已保存。故用四条既有记录完成主要分解，不重建路径、不生成候选、不重新求梯度。

这是对冗余执行的消除，原假设、端点、方向、范数和读数定义不变。原计划额外梯度范数/余弦缺失，保留NA，不用已有normalized_null_residual冒充Euclidean余弦。增量范数可复用quantization.fp32_delta_l2。原门和恢复结论引用已交付实测，不声称本轮新验权重；不接受任何拒绝探针，不启动GPU。

四记录定义：T=fp32_after_null−fp32_before_null，L=fp32_update.null_linear_change，R=T−L，P=native_delta_null−T。报告每端、同方向D−A及F0差减F1差的交互；检查T=L+R及native=T+P。若L/R异号或均大于总差明确抵消，不能唯一归因；残差只称有限增量/数值残差，不能唯一认定为曲率。无新增配置/尺度/候选/模型调用，新增磁盘<50MiB，纯CPU审计。
