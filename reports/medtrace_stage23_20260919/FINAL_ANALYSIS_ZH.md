# Stage23 结果、机制与证据边界

## 完成范围与版本

Stage23执行代码为 `d1ebfc9af388b088fd79bf66e3a64a9f253f4d30`，使用冻结最终锁 `6d93cd1bc574866ab1659185608044a754ccf63c8dbdcb68ef11469291e5dec9`。GPU正常退出0；控制器为 `23_COMPLETE_SCORED`，最高共同完整评分端点N=45。E0/E1复用经兼容审计的完整45 bank；E2复用前19，20–45从原CP-W0新增26个writer，未从已学patch继续训练。全部bank均实际重新生成插入及11/19/32/45前缀，未拼接历史single分数。启动时三条Base原始token及45个native路由key机械回放通过。

本目录 `closeout.py` 只消费已完成原始输出及冻结评分，验证909条唯一student输出和5条CONFIRM Base全部有语义评分，不调用GPU或Judge。历史Stage20、21A/B文件、输出和标签不修改。Stage21B的L31 down对L31 up仅同Transformer block，不是严格writer模块匹配；此限制不因本轮完成而解除。

## 主结果与分母

[导师摘要](ADVISOR_UPDATE_ZH.md)列出最高共同端点。`RESULTS.json`、`RESULTS.csv`同时交付所有有效11/19/32/45结果、QA与来源等权结果及leave-one-source-out描述，不按成绩选取停止前缀。Retention分母是方法无关Base原正确项；Fix是Base原错误项；未支持的指标为NA/null，不补零。

E2在24问已查看H上的正确率14/24、保持11/18，高于E0的12/24与7/18；但H Fix为3/6，低于E0的5/6。E1为11/24、8/18、3/6。E2和E1旧H均4/6、保持2/4，E0为2/6、1/4；旧U严格保持则E0为6/7，E1/E2仅3/7。已查看新U严格保持三组均7/8，而固定面板保持为10/13、12/13、12/13；不能混用严格无关子集与固定面板分母。改写E2为10/11，对照均11/11；正向DEV三组均1/4。

Stage22B用于选参的是16问H面板，Stage23最终回归的已查看新H为24问，不能把两个不同面板的数字直接写成规模增加引起的变化。旧new_H/U均是已查看DEV。旧6H仅2来源；`ANCHOR_PAIR_CORRECT.json`保留11锚点计权口径，N45 E0=50.00%、E1/E2=72.73%，它不能替代unique-QA与来源等权报告。少量来源的leave-one-source-out只是敏感性描述，不是临床人群区间或显著性检验。

## 封存探针：不足与负结果同时保留

原候选来源先保护，参考/关系审阅7项完成后封存合法面板，随后冻结单一候选；完成全45回归之后才一次性生成CONFIRM。实际合法面板为H1问/1来源、U4问/4来源、positive0，资格绑定见 `FINAL_METHOD_LOCK.json`。该面板最多支持有限source-held-out probe描述，不代表患者独立、未见编辑或完整独立确认。

NO_H在H探针1/1正确，两种H配置均0/1；E2相对NO_H的H来源等权保持差为−100个百分点。在仅1个H来源下不能作人群推断，但必须报告这个观察。U三组均3/4正确、严格保持3/3，Base原错1项均未修复。正向确认无支持，NA。官方状态保持 `INSUFFICIENT_CONFIRM_SUPPORT`，不修改为通过；`CONFIRM_REQUIREMENT_AUDIT.json`单列已可判断的H退化与未满足条件，避免用支持不足掩盖负结果。没有再次调参、换样本或重判。

## 自然路由、输出差异与梯度

113问最终回归三组全部激活专家、实际路由完全相同；E2相对E0/E1分别37/14条原始token输出不同。5问CONFIRM相对两对照分别2/1条输出不同，路由均未切换。完整配对正确性转移在 `PAIRED_SOURCE_RESULTS.json`，不通过条件激活筛掉主评价输入。

`ROUTING_PREFIX_TRANSITIONS.json`记录匿名逻辑顺序号、实际与最近邻专家、方法臂、L30 writer与L31 router feature，以及同一输入跨前缀的对错转移。11→19旧H发生专家切换：E0有4条对→错和1条错→对；E2有1条对→错和1条错→对，因此相同总分可能隐藏抵消。19→32旧U：E0有2条随切换错→对，E2有1条随切换对→错。32→45的58个共同输入没有路由/正确性变化；新增终点评价来源不能被描述为该前缀的纵向变化。

最终专家13承接113问中的33问，提示评价输入集中落入少数专家；这与参数更新影响共同出现，不构成路由竞争的独立因果证明。没有强制专家结果替换主评价，也没有改阈值恢复方法差异。

26个新E2 writer均记录320步非零H梯度，H目标token合计27200；U188480、native30080、fit30080。1/80/160/320步稀疏梯度余弦保存在 `RESOURCE_AND_ARTIFACT_PROFILE.json`，不是从旧梯度范数推算，也不改变优化更新。不同支持的token长度和实际计算不能据步数相同宣称严格等算量。当前证据不足以将整体差异单独归因于H或HSIC；历史45/45选L30仍不支持动态选层优势。

## 账单、存储与交付

| 阶段 | GPU进程墙钟秒 | 新增语义判定 |
|---|---:|---:|
|22A|46.925882|7|
|22B|7634.923701|14|
|22C|4053.965610|14|
|23|2993.366888|16|
|后续共享累计|14729.182081 / 28800|51 / 2000|

所有加载、初始化、教师、训练和生成均按共同进程墙钟账本计入，各session正常退出。Stage21另账11083.196193/12600秒、38/800项，不合并或继承余额。Stage23峰值torch allocated14.514GiB，不等于整卡显存占用。Judge固定隔离gpt-6-astra/high source-agreement，新增判定按项目计数；1599次精确复用为所有后续消费者累计。供应方usage记录input137621/output3373/reasoning437，reasoning可能是output子项，不能直接相加。接口未暴露货币实付账单，实际费用保持null/unavailable，不能用推算价冒充实付。

最终bank与共享旧writer依赖保留，135个引用可能共享文件，不能把引用数视为新训练writer数。CPU读取bank metadata，并核查引用存在非空；不声称所有张量逐字节验证。磁盘余11.87GiB，不做无谓清理，未动原始数据、共享模型、历史正式工件或无关任务。`CHECKPOINT_CONSUMERS.json`记录已完成消费者和保留理由。原始QA、图像、输出、token序列、权重及身份映射仅私有保存；公开仅本目录源码与脱敏汇总，发布回执另在private保存。

## 证据索引

- `RESULTS.json/csv`：全部端点/面板，coverage、QA与来源等权分母。
- `PAIRED_SOURCE_RESULTS.json`、`ROUTING_PREFIX_TRANSITIONS.json`：配对对错/输出差异和路由转移。
- `FINAL_METHOD_LOCK.json`、`CONFIRM_QUALIFICATION_SUMMARY.json`、`CONFIRM_REQUIREMENT_AUDIT.json`：冻结候选、资格与门槛。
- `GPU_BUDGET_LEDGER.json`、`JUDGE_BUDGET_LEDGER.json`、`RESOURCE_AND_ARTIFACT_PROFILE.json`：跨阶段耗费、复用与实际训练诊断。
- `CHECKPOINT_CONSUMERS.json`、`DELIVERY_STATUS.json`：保留和完成覆盖。
- Stage22的选参、来源建设和覆盖全臂结果保留在相邻Stage22目录；本轮不重发或覆盖Stage21/20材料。
