# Lingshu L0 预检

## CPU 确认

固定 Lingshu-7B 仓库提交 `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`。保留发布源的 config、chat template 和 generation config 原文。[固定 Lingshu 快照](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B/tree/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9) · [版本和运行合同](lingshu_l0/runtime_lock.json)

与固定 Qwen2.5-VL-7B-Instruct 对照，双方同为 28 层、隐藏维度 3,584、中间维度 18,944。实际在 pro5000 的 Transformers 4.51.3 中用 meta 设备建模，确认了唯一的第 18 层 `down_proj`、张量索引绑定和 FP32 TT 的 8,256 个参数。[固定 Qwen 快照](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5) · [预检代码](../../experiments/med_scope_subspace_v1/lingshu_l0.py)

## GPU 预检合同

采用 BF16 和 SDPA，视觉 token 256–1,280，贪心解码、重复惩罚 1.05、最多 1,024 个新 token。10 个开发样本来自不同图像，排除阶段 5。逐条检查自然结束。真实模型核对零残差前后的 logits 完全相同，并用非零残差验证第一个答案 token 的提示位置生效。GPU 7 上的 10 个样本均自然结束，未触顶。新 token 数为 10–194。真实模型的零残差 logits 完全一致，非零残差校验确认首次作用于预测首个答案 token 的提示位置。4 份权重共 16,584,414,544 字节。原始问答和图像留在私有运行目录。[L0 聚合结果](lingshu_l0/RESULTS.json)

## L1 数据绑定

已绑定 150 道原问和 140 道冻结官方同义改写，对应图像不含阶段 5。SLAKE 按图像在发布源 train、validation、test 中的真实归属分层，共享图像保留多划分标签。VQA-RAD 的现有公开 JSON 无 split 字段，作为独立无划分层。L1 的 Base 生成排在 S1′ GPU 评测之后。
