# TXT三点margin稳健性独立计划

用户2026-10-01明确授权满意结果的有限参数验证。本计划在读取本轮参数输出前冻结。TXT原tau0在DEV与暴露REG均通过原joint，本轮回答通过是否依赖精确margin点，不能补充独立医学scope。TXT为确定性且无新训练，改变无效seed不构成多种子复核。

固定tau=-0.1、0、+0.1三个对称局部扰动；范围是有界q∈[-1,1]的固定0.1工程扰动，无CAL拟合或输出驱动选取。并非部署阈值推荐。原R0 winner、原radius、原NEG正负支持、native+4S_fit减U_bg严格词面保护全部固定；winner在原R0 radius内，q>=tau或词面保护才接受，否则OFF，无fallback/扩大radius。三个点全部报告，不据结果挑最优tau、不改旧门槛。

只读已完成scope-text-guard-20261001/run的强绑定消费者、原H/Base输出keys、原score协议和永久missing锁、BASE_MASKS、闭合joint支持资格。逐消费者以R0的H或Base现有响应/scorekey构造缓存反事实。完整989DEV+1058REG（各single、bank4/8/12/24、old47）均执行；REG仍为EXPOSED回归，不是新准入或独立CONFIRM。复用原decision.joint、exact和Judge cache binding检查，不做新请求。tau0全部2047路由/key和完整joint必须精确重现原TXT，否则机械FAIL，不接受其余点结论。

科学判定只用原各prefix门槛：T0/T1G新增错误上界0，T2G与Base合格T2L上界1，old35各prefix上界1；bank24 Pressure已知增益至少3且shared-key missing精确差下界>=0，原negative support资格不变。完整分母与missing保留。已有41永久missing不重判。多prefix、多tau不是独立样本数。

预先固定结果分支：全部三点两个面板PASS只能称局部margin稳健；任一点FAIL/INCONCLUSIVE则公开敏感项与最早失败门槛，原tau0科学结果保留。不从本轮选阈值，不追加中间点或扩大范围。若局部敏感，下一轮仅在原输出基础上预先冻结失败路由机制诊断；若全部PASS，则转向真实来源scope审核/资格，只有合法数据与角色证据到达才能冻结临床/新数据验证。训练多seed待有真实随机机制与独立预算的新计划，不空转GPU。

首次墙钟1小时，单次CPU任务上限60秒，存储32MiB；GPU/Judge/训练/生成均0，累计Judge7472不重置。中性入口、CUDA_VISIBLE_DEVICES为空，所有大文件留/data/bmw。任何机械失败保存源码/回执/日志/原时钟和累计成本，修复先定位绑定根因，无盲重启。现有历史任务/权重/数据/旧PR只读。完成后审阅并代理发布源码、配置、自检、匿名结果与限制到新draft PR，attach并验证远端SHA/匿名访问；更新小时交接持续推进。
