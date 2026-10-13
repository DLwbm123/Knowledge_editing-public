# P0历史证据归因

这里只读已有逐项输出、路由与实际Judge证据，没有新增判分。不是H的匹配因果估计。

{
  "status": "COMPLETE_EXISTING_EVIDENCE_ONLY",
  "occurrences": 7291,
  "categories": {
    "IDENTICAL_PAYLOAD_UNCHANGED_OR_MISSING": 5850,
    "SAME_SELECTION_OUTPUT_CHANGED": 1361,
    "IDENTICAL_FULL_PAYLOAD_VERDICT_CHANGED": 11,
    "SELECTION_OR_ACTIVATION_CHANGED": 69
  },
  "T2G": {
    "single": {
      "probes": 557,
      "old_correct": 512,
      "new_correct": 493,
      "losses": 45,
      "fixes": 26,
      "missing": 0
    },
    "sequential": {
      "probes": 557,
      "old_correct": 504,
      "new_correct": 485,
      "losses": 44,
      "fixes": 25,
      "missing": 0
    }
  },
  "T2G_loss_occurrence_intersection": 43,
  "T1G": {
    "damage_occurrences": 76,
    "unique_queries": 76,
    "edits": 51
  },
  "T2L": {
    "probes": 54,
    "old_correct": 25,
    "new_correct": 26,
    "old_macro": 0.4838709677419355,
    "new_macro": 0.5591397849462365
  },
  "new_Judge_attempts": 0,
  "old_scores_read_only": true
}

T2G single45损失/26修复，final44损失/25修复；损失集交43。两者净少19不能当成同一批19。PR22 T1G退化76次/76不同query/51编辑。T2L微平均25/54→26/54；宏平均48.39%→55.91%，仅净多1probe。

旧路由route_on是R0/RC字典，首版把字典当布尔的解析错误已保留并修复；该修复仅改变P0分类，没有改生成/判分，GPU/Judge成本均0。11次同完整载荷异判显示历史评分漂移，不能把所有变化归因于H。患者独立UNKNOWN、H_eval NA。
