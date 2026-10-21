# VQA-RAD / PadChest-GR 用途、划分、保留排除与真实加载链定向审查

本轮终态 **NAMED_SOURCE_AND_PRIVATE_INPUT_BINDING_GAPS_NO_LAUNCH**。两个指定来源的划分与配置加载链已有具体证据，尚未形成可启动的私有源输入绑定。真实VLM/GPU/KV生成/IND2/JOINT2均0；没有扩展toy、重跑201/38普查或改变保留角色。双成功仍是后续H1结果，绝非启动前提。

## 来源与划分

VQA-RAD 原JSON共2248问题。phrase_type的freeform/para合1797条，test_*合451条；前者313图、后者203图，交叠202图，非test独有图仅111。这是本地元数据约定，不能把“非test问题”直接称为独立训练源。evaluation字段也不是训练划分或许可。官方OSF页面本次403，未核得当前原始用途条款，不将可读取原数据当用途许可。

指定角色账本含2248条VQA-RAD记录、314个image_id组：297条带FROZEN_FIT_SOURCE、116条FORMAL_OR_RESERVED_SOURCE、76条CAL_CHECK_SUPPORT_RESERVED；这些为重叠记录计数。图像组聚合有14组涉及保留、37组涉及fit，277组无这两类标记。277仅是限定账本的角色缺席，不是允许源数；111也仅是排除test图后的上界。原image_name与角色image_id的认证连接及完整暴露/保留排除未完成，不将两个数字合并推算合法候选。

PadChest-GR 的master_table共有8787行，train6153、validation872、test1762，属于表行而非独立病例。对应图像/Study分别3185/455/915，病人3123/443/893；在此表内，三划分的图像、Study、病人两两交叠均0。grounded JSON为4555条Study记录及findings，既有结构不是编辑QA对。

表内study_is_benchmark和patient_is_benchmark全部True，study_is_validation全部False。它们是数据集内部字段，不能直接等同本项目保留角色，也不能据此把整个训练集判为禁止。私有诊断中“全部benchmark均排除”的假设会得到0剩余，那只是条件算术，**不是最终排除规则或准入计数**。本轮不按此假设改角色。指定项目账本无PadChest匹配记录，不能证明项目从未保留或暴露；其跨项目保留排除仍NOT_CERTIFIED。

[官方PadChest-GR页面](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)明确将其定位为grounded报告生成，要求正式申请完整临床库访问，并链接研究使用协议。[该协议](https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/)提供研究用途许可，限制数据再分发、重新识别和临床使用，非学术用途另须审批。因此公开研究用途依据已找到；不能继续笼统称“完全没有许可”。本地批次的正式访问来源记录及本项目用途/排除绑定尚缺。官方GR用途也不自动形成当前编辑QA的输入与目标协议；本轮没有生成新医学标签。[作者论文](https://arxiv.org/abs/2411.05085)题为PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation，旧下载说明的其他标题不沿用。

## 实际加载类与输入绑定

沿已登记launch配置只读核对，official_native由LlavaMedEditorRuntime进入load_frozen_backbone与官方load_pretrained_model。launch的PYTHONPATH提供固定official_llava，实际配置类为LlavaMistralForCausalLM，继承MistralForCausalLM/LlavaMetaForCausalLM，model_type=llava_mistral。模型、视觉快照与官方代码的既有登记ID见LOADER_INPUT_BINDING.json；没有加载权重或重新哈希。裸CPU环境find_spec不存在不等于实际launch缺包。

checkpoint配置标记bfloat16，冻结runtime却为FP16；以未来实际实例参数核验为准，不能用配置标签覆盖历史实际dtype。已安装transformers4.51.3的Cache路径与旧自定义forward间有兼容包装，包装会丢弃cache_position/logits_to_keep。此处既不能静态宣布不兼容，也不能宣布KV通过，必须在准入后观察实际prefill/decode/cache对象及输出。

已明确输入契约：官方tokenizer/图像预处理，mistral_instruct，仅图像+问题，不向prompt加入目标；参考仅用于完整源NLL监督。具体图像ID/私有路径/问题/参考、排除绑定、special-image-token、attention/padding与目标边界清单仍UNBOUND。没有用旧保留源代替，也未伪造某个“已选首源”。

## 冻结烟测与停止

复用既有REAL_VLM_COEFFICIENT_KV_SMOKE_V1，未修改原配置：layer6 down_proj原生FP16，诊断rank1从当前源梯度确定，非零系数1e-4乘W0 Frobenius范数；它只是接口方向，不是后续编辑rank/lr。零优化、零teacher、零Judge、零checkpoint。相同原生矩阵按cache=True/False完成两次官方greedy生成，max_new_tokens1024不缩减，每次最多128F。完整token及终止状态须一致；aligned logits最大绝对误差≤.05、相对RMS≤.001。须观测非空past与单token缓存decode、无缓存增长序列；自然无decode则CACHE_NOT_EXERCISED停止，不换prompt。

总320F/4B，工作300F、恢复20F；GPU工作240秒、TERM270/KILL300秒，整体1h；本轮烟测内部256MiB，GPU2固定UUID、空余≥48GiB、owned≤36GiB并保留12GiB工作余量。零重试/缩尺度，非有限、未知override、输入/来源未绑定、预算/恢复失败均停止。正常/异常恢复权重、dtype/flags/modes/hooks/grad及生成配置，W0 NLL绝对容差1e-5。现阶段缺源与输入绑定，**没有launch**。

## 后继判断

下一步仅补具体访问来源/用途记录、认证图像或病人到项目保留清单的排除连接，以及合法源的私有输入清单；不是重新扫描库存。来源与接口准入后执行原冻结真实KV烟测，再独立冻结IND2/JOINT2相同预算与真实参数。双成功及联合额外价值均待测；不照搬toy，也不将native不同当行为收益。完整关系包尚缺时可另冻源端H1，不能声称H2/H3保护。

## 成本与边界偏差

5次只读SSH、0远端新增；原始整体时钟未重置，关闭约598秒，公开材料与私有缓冲合计小于16MiB本机存储限。初始两次读取后，另有三个先冻结的精确followup；第2至5阶段读取计账已达17,909,652字节（含压缩及解压内存计账，首阶段另计）。因此若按初始16MiB作为全轮累计输入上限，本轮超出该读入界，不能声称全轮输入预算合规；分别预冻followup不消除此偏差。到此停止读取，保留原冻结与成本，不据此获得任何模型准入。12个离线标量/公开边界检查PASS只核对汇总，不覆盖这一偏差。
