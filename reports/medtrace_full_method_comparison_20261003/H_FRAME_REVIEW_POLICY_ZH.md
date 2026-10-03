# 既有候选的原始来源视野审阅

此步不扩展检索，只审阅 final 来源清单的 UNKNOWN 项。原始 VQA-RAD `evaluated/freeform` QA、原 `image_organ` 标签、已通过的测试/角色/病例隔离，以及服务器缓存原图均作为证据。审阅者为 Codex，不能签为新临床医师验证或患者级独立确认。

仅接受以下可直接核对的 scope 关系：问题仍是同一无额外限定的单个 lesion/mass/abnormality 定位；原答案的解剖部位与源图明确的视野互斥；源原答案属于其标注部位；完整原图经实际检查与标签一致。仅接受 HEAD 源图排除 lung/liver target，或明确 ABD 源图排除 sella/suprasellar/choroidal-fissure target。不以 CHEST vs ABD 直接判互斥，因为腹部与胸部图像可能显示跨界结构。原始源 QA 答案照原样使用，不自行诊断病变或生成新答案。

记录私有 reviewer、视觉检查、源行、template、原/源解剖部位及拒绝迁移理由。单纯位置文字不同、同一脑区下不同病变、消化器官列表差异、function/effect 字符串差异均保持 UNKNOWN。源 QA 临床注释的作者证据与本次有限 scope 逻辑审阅分别记录，不混称新临床审核。

最终支持 join 合并此前 modality 证明和此次有限人工来源审阅，生成每个原 146 edit 的完整覆盖/缺口表。零训练/生成/Judge/GPU；原时钟与失败成本不重置。全 146 未通过仍禁止 P1，不因本次增加支持而缩小队列。

VQA-RAD 原始出处：https://www.nature.com/articles/sdata2018251。
