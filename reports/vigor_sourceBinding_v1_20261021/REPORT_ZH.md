# 来源身份连接与访问／保留绑定补全

本轮完成 **VQA-RAD 元数据身份连接**，但尚无合法源及完整私有输入清单，终态 **VQA_ID_JOIN_CLOSED_SOURCE_ADMISSION_GAPS_NO_KV_LAUNCH**。没有启动真实VLM/KV/IND2/JOINT2，没有改保留角色或扩toy。新证据关闭旧507189a0报告中的image_name连接缺口，不改写其历史状态。

## VQA-RAD：连接已闭合，候选不等于准入

只读取原JSON的身份/phrase字段投影，与已缓存的指定VQA角色身份记录连接。原2248个qid均唯一，角色2248个qid全部精确一对一命中；原image_name去扩展名后的stem与role image_id逐条相等，缺失、重复qid、图像身份不符与stem对应多个原始文件名均0。共314个图像组。原文件名含扩展名、角色ID不含，之前未完成的连接不是数据不存在或神秘哈希映射。

这是同一qid及文件名规则的**元数据身份认证**，没有读像素、计算新hash，也不声称病人或派生图身份已认证。基准metadata说明曾称image_id直接为文件名，不能代替本次逐条连接结果；公开的bind_identities.py拒绝重复/缺失qid、未知phrase及身份冲突，只输出匿名计数，不输出QA或ID。

按图像组聚合全部问题及角色后，原非test独有111图中：4图涉及FORMAL_OR_RESERVED_SOURCE/CAL_CHECK_SUPPORT_RESERVED，13图涉及FROZEN_FIT_SOURCE，且这4图均包含在13图内。只排保留剩107图；同时排除test图、保留和fit暴露剩 **98图、628条记录**。未按Base/编辑成绩挑题。98/628是指定原始JSON与角色账本上的候选界，不是已允许源、完整暴露史或新医学正确性证明。

当前原始用途依据仍未从可访问的权威端点核实：原官方OSF页面先前受限，本轮官方API端点也未返回可用内容。[OSF原发布入口](https://osf.io/89kps/)仍作为待核证据定位；benchmark下载说明的research-use文字只是二级说明。没有因为连接成功就将这些98图写成合法源，没有提取或使用其答案、也未读取图像来构造烟测输入。

## PadChest-GR：公开用途与本地访问／项目排除分开

此前已找到[官方GR页面](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)及其链接的[研究使用协议](https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/)，研究用途依据保留，不能重新称完全无许可。本轮只查指定数据/metadata目录和两级直接获取来源目录的可见小型用途／访问回执文件名，未找到访问或获取回执；没有扫描图像、访问其他数据集内容、发访问申请或下载数据。该结论仅限所查位置，不证明回执在任何地方都不存在。

项目角色账本无PadChest记录，缺少该账本覆盖全部项目使用的依据，仍不能据缺席认证未保留。旧DATA_ACCESS_POLICY只允许既有native、S_fit、inherited U和audited U_bg，并要求明确purpose及input/source排除；它不是本轮新源授权。PadChest自己的benchmark=True也不是项目保留角色，未将整个train集改禁。此前已核train/validation/test图像和病人不交叠的证据继续保留，但数据集划分互斥不能替代项目用途与保留清单连接。

## 私有输入与真实KV

私有身份映射已保存，只含原文件名/qid/phrase/角色映射，不公开。没有已准入的图像—问题—参考—预处理完整清单，因此没有选择“首个合法源”或拿旧保留源顶替。实际official_native加载类、固定官方代码/模型/视觉版本及cache_position兼容风险沿用已有LOADER_INPUT_BINDING与eca4f530方案，不重复读源码或跑toy。

合法源及私有输入闭合后执行原冻结真实KV：layer6 nativeFP16；零优化；相同原生矩阵cache=True/False各一次完整greedy，1024上限不缩，每请求128F；必须观察prefill、非空past与单token decode/无缓存序列增长、完整token及终止精确一致；logits maxabs≤.05、relativeRMS≤.001；W0 NLL恢复绝对误差≤1e-5，矩阵及flags等恢复。总320F/4B，工作240秒、270TERM/300KILL、20F恢复预留，GPU2原UUID和显存/存储/独立guard准入保持。没有合法输入时不launch。

烟测通过后才另冻真实IND2/JOINT2的相同预算及超参，双成功是后续H1结果，不是启动条件；不复制toy参数、不给native距离附会行为收益。本轮结束后不重复身份连接或普查；后继只接具体权威用途、访问来源、项目排除证据与合法输入绑定。

## 全轮累计读取

本轮在首次研究读取前冻结32MiB累计界，不以followup重置。两次只读SSH均带30秒remote/45秒client限，无重试或远端写入。计入本机旧私有研究缓冲的重复读取、远端完整原文件字节及返回文本、源码脚本；另为工具展示文本及失败API读取保守预扣64KiB。管理指令、Git/交接控制读取单列，不算研究数据。实际计账及原始时钟见READ_BUDGET/CLOSURE，不再追加读取额度。元数据连接PASS不代表来源或模型准入PASS。
