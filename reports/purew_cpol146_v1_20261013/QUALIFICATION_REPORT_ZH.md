# 训练改写资格审查结果

状态：**BLOCKED_DATA_QUALIFICATION**。新 GPU 正式更新为0；当前原146连续写入未被修改。

| 改写类别 | 已审查 | 语义通过 |
|---|---:|---:|
| syntax | 146 | 143 |
| terminology | 146 | 135 |
| interrogative | 146 | 140 |
| context | 146 | 141 |

语义通过不等于最终数据准入：还需通过核心问题变化、四种候选互异，以及全评价题的独立重合检查。

```json
{
  "source_review_status": "BLOCKED_SOURCE_QUALIFICATION",
  "semantic_or_style_failures": 30,
  "collision_screen": {
    "candidate_count": 584,
    "collision_candidates": 6,
    "counts": {
      "T2G_exact_text": 5,
      "T2G_normalized_text": 5,
      "T1G_exact_text": 1,
      "T1G_normalized_text": 1,
      "T2G_normalized_full_input": 2
    },
    "screen_all_eligible_and_noneligible": true,
    "source_and_answer_not_changed": true,
    "evaluation_labels_read_for_training": false
  },
  "candidate_binding": "876ee17b250ce8bebd5bdb34c771b8803b3297c394e65a18b00f66a28591da7b",
  "source_binding": "c76886efce873b9860135dccd4f82877bf7670c3394badcfbe1ea4ea847d39b9",
  "review_binding": "4eab17b34e4cf9ead05ae4adf33b1eefe29781e37169aeb80eff381327c842cb",
  "status": "BLOCKED_DATA_QUALIFICATION",
  "candidates_repaired": 0,
  "evaluation_examples_disclosed_to_author": 0,
  "training_launched": false
}
```

共尝试 16 次作者/独立审核请求，全部调用、失败与费用计数保留。资格审查不使用GPU。首批本地事件分类误报已经保留；直接复用原响应，未重请求。

C/P/O/L的机制与逐步范数预算保持冻结。数据版本未通过时不取通过子集、不依据评价重合信息修补候选，不把评价题用于训练，不启动P/O/L。若数据资格通过，仍须完成并验证GPU执行器、评分和交付消费者，再等待原146 GPU完成及新轮真实模型准入；本报告不等于GPU排队成功。

C的11项永久缺评分不重试，最终对比保留共享缺分上下界。保护按逐题零新增损伤判定，不能用平均提升补偿。
