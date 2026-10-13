# 整批训练改写：修订版2资格结果

状态：**BLOCKED_DATA_QUALIFICATION**。新GPU正式更新0；原146连续写入和旧版全部记录保留。

| 类别 | 审查条数 | 语义通过 |
|---|---:|---:|
| syntax | 146 | 146 |
| terminology | 146 | 143 |
| interrogative | 146 | 145 |
| context | 146 | 145 |

语义通过还需核心问句变化、候选互异及独立重合准入。作者/审核者仅见原训练源問答及新候选；没有评价题/答案、旧候选、旧失败或重合项身份。

```json
{
  "source_review_status": "BLOCKED_SOURCE_QUALIFICATION",
  "semantic_or_style_failures": 13,
  "collision_screen": {
    "candidate_count": 584,
    "collision_candidates": 2,
    "counts": {
      "T1G_exact_text": 2,
      "T1G_normalized_text": 2
    },
    "screen_all_eligible_and_noneligible": true,
    "source_and_answer_not_changed": true,
    "evaluation_labels_read_for_training": false
  },
  "candidate_binding": "2623e9d8553ca92cddb6fd3f15409ce760cd6be9f21ccea20a50d5e2292bd1e5",
  "source_binding": "c76886efce873b9860135dccd4f82877bf7670c3394badcfbe1ea4ea847d39b9",
  "review_binding": "78ba9076d4e1cca150fbc873c0f2c65b70c32cf5bc6bbcb348e28b207227d7a6",
  "status": "BLOCKED_DATA_QUALIFICATION",
  "candidates_repaired": 0,
  "evaluation_examples_disclosed_to_author": 0,
  "training_launched": false
}
```

本版尝试16次请求，累计含旧版32次；旧请求不重试，全部成本保留。

只在整批资格PASS后冻结可训练数据绑定。仍须完成GPU执行/评分/交付代码并验证，等待原146GPU全部阶段结束、正常退出和无活动租约，再进行新轮真实模型准入；本报告不是GPU已排队或准入已通过。失败不得依据评价重合反馈改词或选子集。C的既有11项缺评分保留上下界。
