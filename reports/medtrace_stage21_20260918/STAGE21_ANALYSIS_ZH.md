# Stage21 A机制分析（B待澄清）

## Q1 NO_H在45后是否仍≥FACT

**NOT SUPPORTED** — 并非逐指标成立。native同为45/45；旧H保持 FACT 2/4 (50.0%) vs NO_H 1/4 (25.0%)，旧U则 FACT 3/7 (42.9%) vs NO_H 6/7 (85.7%)。

## Q2 H是否提高native编辑成功

**NOT SUPPORTED** — 两者native和当次写入均45/45；本流没有可观察的增益，不能外推为H在任何场景都无效。

## Q3 H对新H/U是否稳定正贡献

**NOT SUPPORTED** — 新H保持仅多1/18，严格新U相同；新H总体正确性 FACT 11/24 (45.8%) vs NO_H 12/24 (50.0%)。参考来源宏平均、LOO和描述性区间，不能称稳定共同增益。

## Q4 同L21下FACT是否优于BE

**INCONCLUSIVE** — B未执行，且实际BE为L31 up_proj，原问题的同L21前提不成立。

## Q5 FACT-L30与FACT-L21差多少

**INCONCLUSIVE** — 未取得冻结层控制结果，保留NA。

## Q6 优势来自layer还是writer

**INCONCLUSIVE** — H有局部取舍证据；层/模块/容量/优化步数差异尚未被充分控制，不能单独归因。

## Q7 HSIC有input-dependent层选择吗

**NOT SUPPORTED** — CURRENT HSIC DYNAMIC-LAYER CLAIM NOT SUPPORTED。旧45与本轮重新记录45均L30，选层熵0；分数随输入变化不等于所选层具有输入差异。

## Q8 相同routing switch下谁更易True→False

**SUPPORTED_DESCRIPTIVE** — 三种已完成方法四个端点的自然路由逐查询一致，共15个相邻前缀切换事件。TF/各方法切换前正确数：FACT 0/5，NO_H 4/8，BE 3/3。这是不同既有正确集合上的描述性结果，事件重复、样本很小，不做独立推断。差异出现在相同路由对应的专家载荷；不排除未控layer/module差异。

## Q9 低秩patch实际storage ratio

**SUPPORTED** — 实测文件 BE/FACT 209.53，纯张量 796.44。前者含不同resume包装，不是严格等价部署格式。BE张量结构仅CPU读取一个兼容样本，45个文件均量取实际文件大小。

## Q10 支持long-sequence claim吗

**NOT SUPPORTED** — 仍是固定45条小规模DEV纯流、单一顺序和有限评价来源；不支持100/146长流、患者独立或临床确认。

EXTRA_FULL11_UNSUPPORTED 保持；没有改样本、标签、tokenization、路由阈值或强制专家。后续实验在B问题明确且本阶段收口后再预注册，不能按现有分数挑好前缀停止。
