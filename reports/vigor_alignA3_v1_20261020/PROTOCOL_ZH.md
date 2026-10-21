# 合格参数教师→受保护单层学生 A3：独立协议冻结

前置：TQ3教师源语义/严格忠实/联合资格3/3，已完整公开并验证远端/匿名访问。H固定8token/top16/forward-KL配方已经结束。本轮检验另一项完整机制：独立参数教师 + 学生完整on-policy轨迹 + 全词表student→teacher reverseKL。它是AlignEdit启发的医疗VLM单层保护适配，多项实现随机制共同改变，不宣称H单因素消融、忠实完整AlignEdit、或某一组件单独造成收益。

## 数据、对照、终点

E01/E02/E03三源固定顺序，每例学生独立原W0；不看T1G/T2G/评估QA或locality答案改任何教师/目标/停止。旧N三例T0 1/3（仅E03对）及同六题面板损伤1/18直接复用，不重建/重生成/重判旧N。旧N实际成本从既有N3/F3账本引用，不能用3×512冒充实际1350。

每例最终最近合法学生端点一次完整T0+同已冻结6个Base正确无关面板输出，总21个端点。主指标源语义/3、完整缺失和上下界，损伤/18及逐检查对集合；近边界0 NA，18非独立病例。预设有用性：保留旧N正确的E03，至少E01/E02之一新增正确，且实际损伤集合不比旧N增加；缺分不能判成功。最终评分不反馈训练、不选中间端点。

## 教师构建和身份

为本轮蒸馏必需，逐例按TQ3原冻结fit代码重建layer6单down_proj教师：Adam .0005/betas .9,.999/eps1e-8/wd0、最多25步/loss<.01停、原chat完整源NLL、选择矩阵FP32/native提交。W0重新开始，核原基座损失以及TQ3该例固定fit forward/更新步数和两个native损失精确至1e-5，否则停整轮不重试。只重建参数教师，不重复teacher完整生成或资格评分；重建所有实际forward/backward算入该例512编辑上限，同时单列成本。不能因不同重建结果换步数/层/学习率或改资格。

仅保留当前教师layer6矩阵的GPU内存快照，不落checkpoint；其他教师权重均W0。学生仅原layer0 down_proj可写，其他参数包括layer6为W0。teacher forward时临时恢复student layer0到W0、安装合格teacher layer6；同原问题/图像及同后缀，不输入参考提示。finally同时恢复两个矩阵与原生dtype，CPU正常/异常swap及真实绑定须通过。teacher logits在学生梯度图建立前no_grad读取，完全停止梯度；不得带着editable_fp32的输入cast hooks将teacher半精度权重混入前向。

## 固定学生更新

每macro step从当前学生原问题/图像生成一次完整greedy自然回答，使用原官方generation_config/max_new_tokens1024，每次128forward上限不降低配置；使用官方返回的完整raw_token_ids，不裁8token、不选token、不拼正确后缀。完整轨迹超请求预算则保留缺失/停止该臂，不用partial trajectory或追加预算；从最近合法端点评估仍须原总预算足够。

教师在同一完整学生后缀上一次完整prefill给每位置全词表Q，学生P。KL=mean_t sum_v P(v)[logP(v)−logQ(v)]，source平均NLL系数固定.02，KL系数1。轨迹及Q停止梯度；不是top16、不是teacher→student，贪心rollout是相对论文采样设置的明确适配。无跨步/跨模型KV复用。完整rKL、因果索引和梯度方向必须先CPU可验证。

联合梯度用原旧F0/null投影解析方向，cost_cap5.368434395105091e-13、最多64 macro step、原1/.5/.25三尺度首合格、固定原控制器半径规则。margin只诊断；原源full-NLL预测/实际下降>1e-5、rho_NLL≥.1，联合J同样下降/rho守卫，任一不通过不接受。原native visual/null相对最初W0 abs≤.01、trust .001、旧F0/secant不变。主收益与trust过而null拒绝才旧secant≤1/提案，补偿后失败结束臂，无追加救援。不换目标权重/前缀长度/温度/门/教师追分。完整源NLL守卫与联合目标冲突时接受科学停止。

## 硬预算和停止

每例512编辑forward包括教师重建、所有学生rollout、teacher/student KL、梯度、保护、补偿和复核；每例64macro上限为从属限制，预算更紧，不把teacher构建藏到免费预处理。完整21端点评估各最多128forward，全轮总5000forward/工作4800；GPU工作1740秒、独立guard1770 TERM/1800 KILL，恢复余量200forward。整个独立实验首次CPU实际执行开始4小时含失败/评分/等待，重试子任务共享原时钟，重建不得重置。内部每服务器256MiB（root240MiB+临时/评分/源码/日志及余量16MiB），预计实际<10MiB，不存teacher/student checkpoint、不下载新模型/数据、不删历史。每服务器新增严格<50GB、总原始历时≤24h上位限制保留。

固定pro5000物理GPU2及原UUID，≥48GiB可用余量；源代码先代理推送、CPU测试全过、存储挂载/余量/写读探针和真实基座/教师准入通过后才学生更新。中性主子argv、独立PID出生tick guard、ps+nvidia-smi核查。任何绑定/数值/恢复/全局预算异常停整轮留现场无重试；科学无合格提案或每例预算停臂，完整分母保留。

终态恢复退出后评分预约去重；原source-agreement/Astra medium/工具0隔离，一批≤21payload、每条一次/600秒，所有相同question/reference/完整输出严格缓存复用、冲突停不挑分，缺失保留；无新teacher评分。无需用新增评价题或Judge迭代teacher。完成匿名代码/标量轨迹/全部拒绝失败缺失和成本/中文报告代理公开，核远端SHA/匿名HTTP200；私有QA/参考/回答token/向量/权重/私有路径和第三方源码全文不公开。唯一pure-w持续，无subagent。
