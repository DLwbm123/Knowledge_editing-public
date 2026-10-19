# V16 来源核查与完整连通组重划分

本轮按用户新授权在全图像连接组上重划分，解决了V15原官方train/eval共享组直接禁止产生的0编辑候选问题。新冻结8个EDIT_EVAL组、32个RETENTION_TRAIN来源与32个RETENTION_EVAL来源，80张实际审查代表图、111条所选来源成员记录，native/cross连接组和PMC来源在新角色间冲突0。所有旧轮次（包括V15零模型预候选）涉及的历史图像组/论文来源仍排除；不挑Base正确或方法损伤来决定新划分。

这只是新协议的结构性候选，不是官方benchmark原划分结果，更不是80张合格CT或8个合格编辑任务。实际图像模态、题图/面板、reference、对比选项、native/cross同语义和Base原正确/双错均未审查。按冻结协议，先过来源严格独立性门槛才进入这些消费者。

## 上游核查结果

固定DataP现有24份来源文件检查完成：GMAI-MMBench part1..11 TSV和PMC-VQA五CSV只读表头，共16个header；七份既有下载/提取/schema/版本报告与一份processed映射共8个JSON。没有新下载或图像blob/像素读取。

GMAI十一分片均包含index、question、选项、image、category、clinical VQA task、department、perceptual granularity、modality、original task，未出现patient/subject/study/case身份列。PMC-VQA表头为Figure_path、Question、Answer和选项/标签，同样无这些身份列。八份JSON映射也未发现对应身份键。当前可追溯到GMAI分片/行位置以及PMC图像/论文来源，但没有逐图可验证的患者或研究组绑定。index/part行号是图像记录标识，category/department是任务类别，PMC是论文来源；均未提升为患者id。

因此严格患者/研究独立性仍UNKNOWN/未通过。不能因为完整图像连接组重划分没有交叉，就宣称不同患者；无法用这种推断满足原冻结来源标准。B图像/源审查、C Base和D公平比较均未启动，模型调用/训练/生成/Judge/Fisher为0、GPU小时0，无checkpoint或模型成本。

## 已冻结的条件执行

条件Base上限88个真实候选回答，硬cap96生成/100000语言前向、0训练、2h墙钟4GPU进程小时。只有至少8正确train/8正确eval和2不同组native/cross双错CT编辑对、以及来源/图像门槛全部通过，才准入方法比较。方法比较固定SOURCE/CE/KL/RESERVE各160步、两版BalancEdit各50步、2任务、1480正式更新；包括Base在内总800000语言前向、40000NNback、520生成、12h墙钟24GPU进程小时、2GiB资产，外部保护1e-4和原内部5e-5不变，原版和同支持增强BE单列。新GPU任务当前不具准入，不能把预算冻结说成已启动。

当前需要补齐可核验的上游患者/研究分组映射才能保持这项严格确认标准进入模型实验。现有文件未提供这种映射，不继续反复扫同样数据或重跑旧编辑。图像/论文组独立但患者未知的benchmark研究是另一种证据等级；如改用该等级，必须明确修改准入/声明并另行登记，不能暗中作为本轮严格独立性PASS。V14已证明的源编辑能力和无可识别保护增益均保持。
