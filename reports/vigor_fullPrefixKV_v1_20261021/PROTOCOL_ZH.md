# 完整视觉前缀无缓存接口修复资格

旧 existingDev 烟测已结束失败：累计61计数forward/4 backward，原生权重恢复。旧结果、成本和时钟不改写。本次仅检验已观察到的 Transformers 4.51.3 inputs_embeds 强制 use_cache=True 缺陷修复，不调整编辑方法或追分。

复用既有已授权E01开发编辑源、精确图像/问题/参考/输入、原F0绑定与校准排除；不找新数据、不改变保留角色、不声称独立泛化。

仍调用官方LLaVA视觉预处理和官方greedy生成循环，仅在无缓存请求临时替换实例输入准备函数：每一步拼接完整原视觉前缀与全部已生成token嵌入，按完整mask计算position_ids；实际forward必须use_cache=False、past=None。拒绝视觉前缀替换、非空初始continuation、非逐token增长和mask不匹配。finally恢复实例方法；缓存请求不修改。

固定完整烟测：当前源梯度/系数导数/native FP16写入、cache True/False两次完整EOS生成；全token精确相等，逐步全词表末位置logits maxabs<=.05、relativeRMS<=.001；实际缓存单token增量/无缓存全序列增长观察，完整权重/flags/modes/hooks/config/W0 NLL恢复。不得放宽容差或降低max_new_tokens=1024。失败停止此资格，不重试。

首次实际运行起共享320forward（工作300、恢复20）、4 backward、每生成128forward；工作240秒/独立TERM270/KILL300秒。固定GPU UUID、48GiB空闲入场、12GiB加载后余量、owned峰值36GiB；输出256MiB上限（远低于50GB），不下载/保存checkpoint。部署目录初始0，预计<256MiB。旧实验成本另列。

针对缺陷的CPU检查已通过：前缀保留、三步全序列增长、位置/缓存标记、异常后实例方法恢复。它不代表真实GPU资格。仅真实完整PASS后，接原冻结IND2/JOINT2，不改参数；双成功是结果而非前提。
