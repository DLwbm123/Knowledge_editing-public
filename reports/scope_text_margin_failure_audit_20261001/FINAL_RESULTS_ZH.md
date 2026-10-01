# TXT margin失败机制审阅

完整2047底层消费者、三个点6141个缓存路由全部匹配；两个非零点的变化数精确重现前轮。无新模型/Judge/训练/生成，CPU墙钟0.253秒；跨prefix身份与缺失保留自检PASS。

REG +0.1的bank12/bank24 T1G两个已知damage出现来自同一个输入/同一个R0 winner，不是两个独立病例。它们的q均为0.08310286477018478：原tau0保留，+0.1在无严格词面保护时转OFF，缓存答案从正确变错误。该问句相对于winner的native/S_fit严格归一集合为unmatched。来源检查same_image=false、same_source=false；这些身份字段不能证明患者独立或真实临床IN。此处“positive damage”只表示既有T1G任务上的正确性损失，不是新医学scope标注。

DEV两个扰动和REG -0.1均未发现T0/T1G/T2G已知correct→wrong。完整变化、其他任务的damage/rescue/both/missing与margin范围保留在MECHANISM_AGGREGATES.json；已知正确仅取已有绑定cache，不重判missing。不用正向Pressure收益掩盖REG T1G失败。

结论：原tau0通过可复现；固定更严格点的失败由无词面保护的低margin输入误拒引起。本轮不挑更优tau、不追加参数或修改原准入。未证明所有连续margin不可用，也未证明医学scope真正可分。17个M3Bench提案仍UNKNOWN，现有20同anchor原QA也仅条件待审，不能作为新CAL。

接下来保持每小时自主研究维护，核对真实来源与scope审核证据。只有来源/角色资格合格或新增明确未测机制证据时，才在读新输出前冻结新阶段与预算；不以相同暴露题上的连续调参冒充独立确认，不空转GPU。多seed需真实随机训练/采样机制，当前确定性router重复seed没有验证价值。真实签核依赖保持显式，自动结构审计可持续，不能替人类临床审核填真值。

公开仅冻结源码/计划、自检、匿名汇总及限制；私有逐条机制、QA、图像、Judge keys、tokens与权重留服务器。运行未发生技术失败；发布前本地自检曾发现sys导入遗漏并修复，未启动服务器任务或读取其私有输入。
