# HSIC 参考核查与机械验收

已实现 native-only 的 HIB Top-1 选层、逐专家 layer_id 绑定以及同编辑同层共享 W0 的训练原语。CPU 检查及 **一条真实 Base 输入的 GPU3 机械检查通过**；没有训练 HSIC 方法、没有七配置性能结果。

依据 [LOKI 论文](https://arxiv.org/html/2606.19679v1) 的层选择部分与 [作者代码固定提交](https://github.com/neu-spiral/LOKI/tree/bbb5819be8af00ff5ae9708bb02734cc70958c4c)，本适配使用 `0.001*nCCA(Z_l,Z_final)-0.001*nCCA(X,Z_l)`。作者实现的核为维度缩放 Gaussian，右中心化 `K@H`，正则 `1e-5*n`，归一化依赖量为 `trace(Rx Ry)`；本实现用 float64 solve 替代 float32 inverse，并做公式与排列不变性测试。没有把另一个归一化 HSIC 公式冒充作者实现，也没有引入 LOKI 的零空间模块。

以下是明确的适配选择，不声称原样复现 LOKI：Top-1；候选零基层 1–30（作者 Mistral 配置 offset4，即 4–26，Top-3）；每条编辑仅用 native 及四条固定 native-only 问法形成五个观测。每个观测对展开后的非 padding prefill token 做均值，不输入参考答案、H/G、评价或未来编辑。输入表征为第 0 block 输入，最终表征为第 31 block 输出，候选为各 down_proj 输入。核、候选、token 处理和退化拒绝规则均在机械测试前确定；详见 `METHOD_AND_JUDGE_LOCK.json`。

真实检查中，五条问法各有 576 个视觉 token、16–22 个文本 token，视觉 token 占大多数；仅有相关问法的五个观测，统计稳定性有限。该一个预选编辑选中 **第 30 层**，不能据此声称逐编辑自适应或选层分布已经得到验证，也不能把与 Base 错误最终表征的相关性解释为纠错因果作用。

真实 GPU 检查使用未训练的合成专家，依次激活第 3、21、3 层。每次 prefill 只有一个 predictor 激活，随后 13 次 cached decode 均走生成生命周期；残差作用非零，回到第 3 层生成一致，保存加载与 Base OFF token 一致。模型参数保持冻结。该合成专家在消费者结束后已删除。

CPU 另覆盖：常数/重复样本拒绝、禁止评价字段、平局规则、跨层切换、异常退出清理，以及三分支调用中 W0 只初始化一次、起点状态相同、批准队列与任务绑定拒绝。旧 Stage18 的 13 项回归检查通过。新训练原语把 writer 层显式传到 native/A2/W0/continuation，并给初始化目录与分支 checkpoint 绑定层；不同层的 W0 不混用。

预登记控制在小型 DEV 上比较随机层和开发选出的固定层；其评价来源仍可能共享，不能称为独立确认。未获得队列和预算批准前，不执行这些训练控制，不根据当前诊断 oracle 或 H_eval 调整 router。
