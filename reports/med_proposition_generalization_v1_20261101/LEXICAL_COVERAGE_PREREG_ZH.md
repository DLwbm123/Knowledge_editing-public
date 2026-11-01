# 训练目标中的医学跨度：保守词面补充检查

首次形式检查结果已知：16个目标均至少三词。它不能说明医学内容或命题覆盖，因此另登记一次来源内词面检查，作为标注可行性的下界线索，不用来挑训练或评价样本。仅检查同一已准入16个source目标，FIT目标逐项相同；不读评价题、图像或模型，不打印原始文本。

固定、非穷尽词表：解剖类 brain, lung, lungs, heart, liver, kidney, kidneys, spleen, pancreas, stomach, intestine, bowel, bladder, breast, chest, abdomen, abdominal, pelvis, pelvic, spine, spinal, head, neck；成像类 x-ray, xray, radiograph, ct, mri, ultrasound, pet；观察类 lesion, mass, nodule, tumor, fracture, opacity, effusion, hemorrhage, bleeding, infection；限定类 left, right, bilateral, upper, lower, anterior, posterior, no, not, without, absent, possible, suspected, uncertain。按大小写无关、词边界精确匹配，每个编辑每类至多记一次；统计各类覆盖、同时覆盖至少两类及所有类均无命中。单纯类别共现不是事实关系或可靠命题标注；词表漏词或歧义可能产生假阴性或假阳性，不称为医学标注通过。仅将原答案直接出现的文本视为词面命中，不推断图像真值和关系。

与首次检查相同16编辑、已批准训练切片和80条绑定，只输出聚合；独立60秒父进程和10MB单文件限制，0模型/图像/评测问答/新增评分。依据结果决定是否有足够可核验的源内结构及严格配对通用跨度可行，不以覆盖不足强造事实或放宽训练角色。
