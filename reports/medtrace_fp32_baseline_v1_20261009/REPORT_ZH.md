# FP32 Base回答一致性

{
  "status": "COMPLETE",
  "decision": "BASELINE_REQUALIFICATION_REQUIRED",
  "panels": {
    "BASIS": {
      "queries": 61,
      "text_equal": 60,
      "tokens_equal": 60,
      "changed_previously_correct": 1,
      "at_cap": 0,
      "EOS": 61
    },
    "HELDOUT": {
      "queries": 96,
      "text_equal": 91,
      "tokens_equal": 91,
      "changed_previously_correct": 4,
      "at_cap": 0,
      "EOS": 96
    },
    "PRIMARY63": {
      "queries": 63,
      "text_equal": 59,
      "tokens_equal": 59,
      "changed_previously_correct": 4,
      "at_cap": 0,
      "EOS": 63
    }
  },
  "native_controls": 6,
  "new_generations": 163,
  "LLM_forwards": 3973,
  "backwards": 0,
  "updates": 0,
  "new_Judge": 0,
  "new_checkpoints": 0,
  "changed_texts": 6,
  "no_requalification_or_training_claim": true,
  "resource": {
    "new_GPU_process_hours": 0.08498422278298272,
    "cumulative_GPU_process_hours": 42.59959499352508,
    "cumulative_Judge": 12406
  }
}

答案文本变化不是医学错误判定；变化项须独立资格核验，不过滤失败或改原主面板分母。本阶段没有候选优化或保护评估。
