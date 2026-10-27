# MED-RA 阶段0审计：INCOMPLETE，未启动GPU实验

已读取[本次Pro分享](https://chatgpt.com/share/6ae0413a-8184-83ea-9c1b-82f8e40a3b10)中的最新分析与最终执行prompt。范围是一个MED-RA方法、先审计、single通过固定筛选门后才做sequential，监测保持关闭。原V133数据、权重、输出、评分、账本和报告均未修改。

**状态为INCOMPLETE：实际F_MED低秩方向缺失，且所给风险缩放规则需要澄清；没有启动训练、模型forward/backward、生成或Judge。** 预检时两张72GB GPU分别有72,830和68,619MiB余量，显存不是当前阻塞。

## 已完成的实际资产与归因审计

- 原146份MED FINAL、完整canonical及独立小备份可用；逐专家160步、源版本和校准回执可定位。原MED λ为1.6778–62.8992；有限组合TT算子的真实Frobenius范数为2.1880–4.5620，通过小Gram矩阵收缩计算，没有构造巨型dense更新/Fisher。
- 读取MED与M13P各7494个已有consumer的实际路由；选中专家、激活、nearest逻辑ID、半径和距离类型全部一致。124项数值距离相差超过1e-3，最大差1.0356，因此只报告语义选择一致，不声称所有距离数值逐元素一致。
- 重新检查保留下来的124条alias teacher binding，共1190个预测位置；位置与`labels[:, 1:] != -100`对应的实际shifted mask逐项一致。原实际有限差分回执仍为PASS。这里只读既有mask，不声称重建了已删除的曲率方向。
- 146行expert风险账本、逐edit/source-group/role结果、7494行route/state账本与损伤item对应表已生成。原M13P权重缺失保持，不伪造其weight hash或重训。

### 额外损伤和T2L下降是同一事件

MED相对M13P新增的fresh CORE损伤，与T2L CORE从54/54降至53/54，是**同一个probe、同一个owner56、实际选中expert56**。single、bank100、bank146均如此；该专家尚未入库的bank50没有这项损伤。两方法在这些请求中的实际专家选择相同，不能将这项差异归因于两方法选择了不同专家。

owner56的原λ为3.1954，有限算子范数2.7587。其日志末项`penalty_before`为0.00586694，但这是**第160步更新之前，即state159**的能量，不能冒充最终state160的风险，也不能据此判断其真实q位于高风险分位。

### T2G收益与来源组

|阶段/标准|两者正确|MED正确/M13P错误|M13P正确/MED错误|两者错误|
|---|---:|---:|---:|---:|
|single CORE|431|9|4|113|
|bank146 CORE|423|9|4|121|
|single FULL|412|10|4|131|
|bank146 FULL|406|10|4|137|

CORE的9个MED-only正确事件分布在8/125个来源组，4个反向事件在3个来源组；FULL的10个MED-only事件在9个来源组，4个反向事件在3个来源组。这里是描述性归因，不用这些已暴露评测事件挑选alpha、q0或额外训练对象，也不足以证明医学特异收益。

### sequential相对single的退化

|最终bank146的同一问题集合|专家选择改变：MED/M13P|原正确→错误：MED/M13P|原错误→正确：MED/M13P|
|---|---:|---:|---:|
|T1G，578项|101 / 101|77 / 77|0 / 0|
|T2G，557项|17 / 17|10 / 10|2 / 2|

两方法丢失的是相同问题集合；T2G净减少8项。这里只描述已记录的选择变化和结果对应，不通过新干预宣称唯一因果机制。完整prefix分解见JSON。

## 阻塞1：实际MED曲率方向未保留

V133收尾时，`MED.pt`、`FULL_CTX.pt`、`COMMON_FALLBACK.pt`三份已消费曲率缓存已按明确清单清理；这不是146份自有专家权重的丢失。核查了V133/V132记录的远程运行根、本地private/评分目录及另一授权服务器的相应输出入口，未找到这些缓存的副本。原清理回执、mask、trace、alias绑定、训练曲线和源代码仍在。

31个医学rank-1方向不能由trace、λ或一个旧step的能量标量恢复。保留下来的旧通用Fisher也不能替代MED方向。146个真实最终q_i、q0及风险/损伤关联目前均未知，CSV中明确留空。

Pro最终prompt在阶段0明确规定：无法找到实际F_MED低秩表示时停止实现，不用近似代码代替。因此本轮没有静默重建曲率或把旧通用Fisher当作MED。若后续允许重建，应使用原冻结实现与输入，核对mask、trace、类概率及来源；缺少旧方向tensor时，不能声称进行了逐元素或逐字节一致验证。

另一个需固定的实现细节是source-fit保底标准。已有末步日志是更新前loss，且循环FIT条目随步变化，不能将其简单相减冒充state160在全部native/FIT上的最终source gain。

## 阻塞2：归一化q对alpha几乎尺度不变

原prompt定义`q(w) = wᵀFw / (trace(F) ||w||² + ε)`，随后要求缩小`w→αw`来使`q(αw)≤q0`。

但对固定F：

`q(αw) = wᵀFw / (trace(F) ||w||² + ε/α²)`。

ε为0时严格尺度不变；数值保护ε可忽略时几乎不变。沿同一方向缩小更新可以降低二次能量，却不能实质降低这个归一化方向风险。原规则因此会让高q专家一直无法满足q0，最后落到它已规定的alpha=0.50且记violation的fallback；不能将此写成成功的风险约束backtracking。

已用当前V133的factorized penalty实现完成小型合成代数检查，五个alpha的归一化q均约0.1472924967，而二次能量比依次为1.21、1、0.81、0.5625、0.25。它是数学检查，不是实际MED性能或真实q分布实验。

## 尚未执行的最小修正建议

如继续，建议先明确允许按原实现重建曲率，并将候选风险分母固定在raw更新：

`r_i(α) = (αw_i)ᵀF_i(αw_i) / (trace(F_i) ||w_i||² + ε) = α² q_i(1)`。

这保留原q排序、75%分位q0、候选grid、实际mask和医学方向，使缩放能真实改变受约束的风险能量。source保底建议预注册为每条native/FIT保留原MED的90%非负loss gain；alpha1.10仍需source fit不下降。所有判断仅用训练/校准信号，single门和监测关闭要求保持不变。

这属于对科学规则的实质澄清，**尚未当作已批准方案实施**。已有146个raw MED终态可直接只读复用，不必为了得到相同raw更新重新训练160步；如果重建准入通过，只新增曲率/训练期校准、alpha选择和必要新回答。真实q及候选source fit仍须实际计算，不能按本次损伤owner手工选系数。

公开交付为审计源码、匿名expert标量账本、来源组分解和代数检查。完整问题级路由/状态及损伤映射保留私有。single、sequential和MED-RA训练实现均尚未启动，不写成COMPLETE_SINGLE_ONLY或COMPLETE_SINGLE_AND_SEQUENTIAL。

数据：[阶段0回执](STAGE0_AUDIT.json)、[匿名expert账本](EXPERT_RISK_LEDGER.csv)、[差异矩阵](DIFFERENCE_MATRICES.json)、[来源组分解](SOURCE_GROUP_DIFFERENCES.json)、[尺度检查](RISK_SCALE_CHECK.json)。
