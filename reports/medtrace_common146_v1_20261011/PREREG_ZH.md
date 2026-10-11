# 原146三方法统一Astra medium重评

2026-10-11，用户明确授权：先完整绑定校验，复用已有TT88 W0+R0、BalancEdit锁定适配版、BELoRA V2 effect-repaired输出；新的共同Astra medium评分，最多3146次尝试；不补训练、不补生成、不影响正在运行的在线2×2实验。独立运行目录、账本、代码工作树与评分worker，不使用2×2训练预算，不修改历史原始输出和评分。

范围固定为原146项、原顺序、single146、146次插入T0及prefix1/50/100/146。三方法各5722评价记录，总17166；完整输入/原图身份/prompt与attention/runtime/generation/raw文本及tokens/Judge协议完全相同时共享一个判定，合计3146个独立载荷。BELoRA的重复single-native记录去重；不引入历史额外T4G查询。

主表固定原Base资格掩码，T0、T1G、T2G、T1L、T2L全部报告，宏平均按编辑、微平均按probe。不重新评分Base、不改变原掩码；不新增数据、seed、顺序、阈值或checkpoint选择。原L面板仅历史诊断，不是独立合格医学保护评估。

评分为gpt-6-astra medium，沿用原source-agreement提示和四字段文本输入、原严格格式/顺序校验。新epoch MEDTRACE_TT88_BASELINES_COMMON_MEDIUM_20261011_V1，旧成功分数不混入。4个隔离worker、每批最多50条、每项一次attempt，失败保留missing且不隐式重试；shared missing精确界不删分母。无可绑定immutable Judge snapshot，不宣称跨时间版本不可变。

预指定主要对比：最终bank146 T2G，TT88分别减BalancEdit与BELoRA；其余全部五任务、全部前缀及single为完整配套面板。沿用10000次paired edit/source bootstrap与seed20260912；不因看到新评分改变主终点。仅开发/回顾性单顺序证据，不能称独立SOTA确认。参数和算法预算不等；基线保留适配/修复标签。

硬上限新增3146 Judge输入尝试，0 GPU进程小时、0模型加载、0训练、0生成。费用按实际调用账本；不编造美元价格。评分故障全部计数，无自动预算扩展。独立评分进程限定中性argv、共享临时根的跨worker隔离；远程只在/data/bmw下保存私有绑定和评分证据。没有新增模型checkpoint可清理。

完成后核对全局覆盖、失败与预算、评分器退出和配对分母，输出全部75面板、50项TT对基线配对对比、插入保持与评分敏感性说明。公开仅代码、匿名聚合和中文报告，经代理推送GitHub并核实远程提交与匿名可读性；原医疗文本/token、逐题身份评分及凭据仅私有保存。
