# 实际系数接口有界验证协议 V1

先冻结后执行。目标为实际可复用系数前向接口，不复制Q1优化，也不训练多端点。CPU阶段验证生产模块函数和现有editor.nll的相同计算路径；真实VLM整体、官方greedy及cached decode仍需独立设备烟测，不能由CPU替代。

## 来源证据核查

本轮仅一次查看permission38报告提交之后的本路线Git新增文件清单、原明确DATA_ACCESS_POLICY和最新人工指令。只在存在具体新用途登记/新批次清单时定向核该条；不重新执行201/38审查、不SSH读取原TASKS、不搜索其他项目或未授权目录、不下载新数据。当前指令授权核实及准备，不自动转为38组重新用途grant。没有具体肯定证据则SOURCE_GAP，记录本次有限证据范围，不宣称全部潜在来源不存在。用途、开发暴露、来源组/病例分离、关系认证分别判断；163保留不改，未知不转允许。

## 生产接口实现与CPU一次验证

新增独立coefficient_runtime接口，只接受无bias、无hooks的唯一Linear权重，拒绝别名共享参数、已有forward覆盖、参数pending grad。临时前向计算W0+U A V^T并通过F.linear，让A保留autograd；输入转计算dtype、输出回输入dtype。不得通过.data复制系数计算结果。模型其余参数冻结、eval；退出恢复forward/flags/modes/原grad，基座参数值和dtype不写动；非有限输入拒绝。native临时合并用独立上下文、finally恢复原矩阵；临时系数接口必须移除后才测native。

CPU固定6x9无bias Linear、非零FP16或FP64基座，rank2正交U/V由固定arange的sin/cos QR构造，无随机种子、求基或优化扫描。用真实editor.nll函数源定义，因果logits[:-1]/labels[1:]和有效token均值不变；合成批含ignore mask。两个固定系数矩阵只测梯度/串扰，不判断双成功。visual/null输入重叠，要求两者对系数梯度均非零；这不是双可行性实证。

固定检查：FP64连续函数与显式组合矩阵输出一致≤1e-12；系数中心差分eps1e-6相对误差≤1e-5；两系数梯度不串扰/基座grad为空；因果位置与mask使用真实nll；FP16 write后移除接口，prefill形状和单token形状与同native矩阵Linear输出精确一致；重复native检测；正常/异常/非有限恢复；共享参数/已有hooks/pending grad拒绝。实际CPU最多64 forward/16梯度路径，一次已有CPU环境signal45秒/client60秒、1线程、CUDA_VISIBLE_DEVICES空；整体从FREEZE计1h，本机新增16MiB、远端新增0，不保存向量或权重。失败保留、停，不改fixture/tolerance或重试。

## 条件设备烟测冻结边界

当前GPU不准入。只有具体来源用途通过且CPU子路径通过，才形成绑定具体唯一源的smoke执行清单：固定layer6 down_proj、原模型/native精度及layer6 F0，检查唯一参数/无共享/hooks、基座绑定与恢复。最多1源、64全模型forward、4backward、2完整源生成（每请求128forward预算，故总forward上限应为320，完整生成max_new_tokens1024保持）；总320forward、GPU工作240/TERM270/KILL300秒、整体1h、内部256MiB与显存≥48GiB、工作余量12GiB，独立出生tick guard。此上限含基准/求系数梯度/合并前后/生成cached路径与恢复。无优化、无locality生成、Judge0；生成仅核绑定及路径，不宣称医学成功。设备结果超容差或生成差异保存后停，不调整系数找一致。具体系数及原生/连续对照容差、源ID/输入绑定尚未冻结，执行清单缺任一项禁止启动；本段是硬预算上界，不是可执行launch许可。

## 后续IND2/JOINT2准入逻辑

数据许可+生产CPU及设备接口准入后，另冻真实可行性实验。两个端点能否完整源正确、满足native visual/null/trust与实际不同，是H1结果，绝非启动前条件。IND2/JOINT2同W0、同源梯度基/初始化/步数/调用预算；只有跨端点多样性项不同。真实rank/lr/罚项/距离量纲须另立固定依据，绝不照搬Q1的rank2/.02/.25/64步。缺这些字段不启动；不改原null .01/trust .001/visual门。不读评估QA选基、调参、选步，不补第三端点/种子/历史最佳。

源端诊断可与关系面板准入分开：合法源有而关系包不足时，仅可另冻源端H1，无传播/损伤/选择器结论；关系认证与Base正确支持足够后才进H2；候选固定、选择/最终评估隔离后才H3。无双成功保留负结果，不评价KL；双方都好且相同不判写入失败。旧DQ失败及全部终态保持。
