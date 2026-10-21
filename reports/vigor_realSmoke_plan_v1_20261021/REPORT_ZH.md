# 新来源定向核实与真实VLM/KV烟测配置

**当前无已准入新源；烟测配置和停止条件已具体化，GPU未启动。** 这不是双端点失败：真实双成功仍是后续H1待测结果。

首轮一次有界只读SSH读取已登记runtime配置及2份运行源码，共54,175源码字节；只列已有数据根的直接目录与用途文档名称，未读QA/图像/题目成绩，不重扫201/38。除SLAKE还见VQA-RAD、PadChest-GR和benchmark目录；它们只是库存线索，未有当前用途、split与排除绑定，不能按目录名准入。发现精确README和adapter路径后，先单独冻结TARGETED_FOLLOWUP，再一次定向读取adapter/安装版Cache与Mistral/该README，181,459字节。README说明benchmark评价和任务构建、底层数据需单独获取，不提供本轮编辑源批次及用途肯定依据。没有下载新数据、改保留角色、选择容易题或延旧训练。用途明确的新来源仍缺可定位依据，不宣称潜在新来源不存在。

真实源码确认训练批use_cache=False，官方runtime.generate支持use_cache覆盖并返回完整token与sequence contract。定向读取已核adapter确实透传use_cache，安装版transformers4.51.3的Mistral构造DynamicCache且支持get_seq_length。实际加载LLaVA类及其prepare_inputs_for_generation覆盖、输入绑定和设备行为仍未知，不能冒称全链验证。

配置见SMOKE_CONFIG.json，执行细节见PROTOCOL_ZH.md：固定layer6/native FP16、一个当前源梯度诊断方向、A=0导数对照、固定非零1e-4||W0||F诊断幅度；同native矩阵cached/uncached各一次完整官方greedy，不训练。必须观察真实past与decode、完整token精确一致和对齐logits容差；无decode记未触发、分歧/折叠/异常停止，不扫尺度追一致性。320总F/4B/2完整生成、1024 tokens不降低、每请求128F、GPU硬300秒/整体1h、内部256MiB，启动显存48GiB及12GiB余量；原生恢复与成本完整计账。诊断方向及幅度不是编辑超参数。

当前source_binding=null，实际generation binding及加载类/Cache覆盖绑定缺失，配置不可执行。没有新增模型/GPU/Judge/toy或运行旧进程查询。后续只核具体用途记录或具体新来源登记；数据+真实设备接口通过才另冻IND2/JOINT2，不把双成功前置。原全部负结果与唯一pure-w保持。
