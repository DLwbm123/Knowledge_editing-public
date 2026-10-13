# TT-only 预算与 H 更新 V1 冻结执行

完整科学协议见 USER_PROTOCOL_ZH.md，机器配置见 PLAN_CONFIG.json。固定参考PR26 1b33de09237e62c4d655f42b459e5537149ae293。原数据/源/评分不变。

96次原生TT暖启动、240条320次尝试的续训；30个结构/条件/seed消费者臂，每臂8编辑。P0只做有界临时机械更新，不冒充正式续训。TT44/88及机制对照全部三seed优先，之后84/48；每GPU固定2编辑，各结构/seed共享真实W0克隆。全局phase barrier保证优先级不依赖结果。

只更新并部署G1-G4；中间A/B临时可微计算，不注册/保存/自由优化。保留96W0、240TTfinal、8路由；native/A2临时权重最后消费者后列项清理。

诊断固定训练激活为每native/fit/U原行前至多16个assistant predictor；相同W0比较共享该激活集合。W0和1/20/80/160/320记录完整训练项；NO_H额外H诊断保存并恢复梯度、RNG和hook。每次尝试累计core与固定激活函数位移。保护阈值仅来自完整训练native/4fit/U，不用H作为约束、不看评测。

GUARDED一次原H1 Adam候选，alpha1直接使用原候选比特；其余alpha按核增量缩放；接受沿用候选Adam状态一次，拒绝恢复所有核/Adam。样本位置总推进。NaN/Inf立即工程失败。小学习率组固定0.25倍。

三seed先在每edit内平均，再10000配对edit bootstrap和来源连通组敏感性。全原mask分母/永久missing精确界；不把多seed当新增病例。原同完整Qwen键继承，不重判；新键<=6000。

本新阶段硬上限48累计GPUh、24h firstclock、2GiB ownedweights、8GiB余量，允许GPU3/4/5/6最多四卡；PR26的12GPUh是已结束旧阶段，不改旧账本。控制器到限保留状态并停止本轮，完整配对块与缺口分别登记，不延预算。预计按优先组执行，可见进程全部中性入口。

后台控制器P0→训练/生成→Qwen32→CPU报告，不开启新小时监测，不自动新轮。完整结束后科学/隐私审阅与新draft PR交付；源、聚合及中文报告公开，QA/图像/逐题分数/keys/tokens/权重私有。
