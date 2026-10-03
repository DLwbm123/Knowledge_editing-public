# H 原始来源补充：固定 CPU 数据审计

用户回复“暂无，你能否帮我补充”后新增此数据阶段。保留原 146 编辑、原评测与全部旧 P0 结果；不调整训练方法、损失、门槛或预算。本文件先于本次补充输出冻结。所有成本沿用 RUN_MANIFEST 的首次时钟。

复用 M3Bench 已有原始 train QA、SLAKE 图像目录中的原始注释和作者公开的知识图谱。先按图像汇总全部历史角色；任一 QA 含 protected/formal/reserved/CAL/fit 角色，则整个源图像不能作为新 H 来源。再排除整个 Stage17 formal/probe/reserved 图像，以及 VQA-RAD 已知相同 case URL。患者身份未知仍记 UNKNOWN，不能据文件名或不同病例页面推断患者独立。

对角色合格的 SLAKE train 图像，纳入原始中文 QA，保留原问句、原答案与源行；不能把中文翻译或程序改写冒充原始注释。作者 KG 的 vhead/ktail 占位不提供实体签核，不据此补临床标签。只下载小型 KG/train 元数据，不重复下载图像，不读取 validation/test 答案。

同命题检索仅使用先固定的明确模板：原问句相等、无额外限定的成像 modality、全图病变描述、无位置限定的 lesion/mass 定位、digestive-system 器官列表，以及同一器官/组织与完整视觉限定词的 function/effect。保留左右、上下、颜色、器官名、数量、单复数和 region 限定。不做 embedding、模糊分数或阈值搜索。每 edit 按来源 ID 稳定选择最多 4 项，同一图像最多 1 项；来源隔离与模板键一致后才能检索。

本阶段只允许一种直接可验证 H 证据：原始合格 QA 明确标注成像大类，native target 与候选原答案分别明确落在不同且互斥的大类 CT/MRI/X-ray。源 QA 须确实问 modality，VQA-RAD 须为 evaluated，精度/contrast 差异不当成大类冲突。此证据确立的是图像条件下的 benchmark modality 保护关系，验证方式为原注释及互斥类型检查，不伪称临床医师审核。必须重读原始 source row 确认答案/问题/图像绑定；不存在、重复 ID 或不同标注冲突时拒绝。

其他模板命中全部为 UNKNOWN 待审：答案字符串不同、列表部分重叠、diagnosis vs finding、不同位置或不同疾病均不得自动升格。输出完整候选、未覆盖编辑和逐项证据清单，医疗 QA/标识/图像路径留服务器私有。公开只有聚合、代码与限制。全 146 未通过前不启动 GPU/Judge，不改成 NO_H 或缩小队列。

modality 家族先筛掉大类相同、未知或多义答案，再按稳定 ID 选择最多 4 个明确互斥的源图像；这是在输出前指定的负例定义，不依据学生或 Judge 表现选择。

边界：最多 64 MiB 输入，60 秒 CPU、32 MiB 新输出，小型作者元数据下载上限 8 MiB，零 GPU/生成/Judge。记录失败和消耗，不覆盖旧证据；一轮固定审计后仅根据明确来源证据做关系审阅，不无限重扫。

作者原始说明：https://www.med-vqa.com/slake/；作者镜像：https://huggingface.co/datasets/BoKelvin/SLAKE。镜像有 cleaned-version 差异，不能静默替换原队列或源标签；本次不读取其 protected splits。
