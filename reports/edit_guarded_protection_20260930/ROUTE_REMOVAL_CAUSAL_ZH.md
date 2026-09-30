# 完整冻结 REG12 路由移除诊断

该块使用真实 R0 与生成；不用于 rho 选择。

```json
{
  "status": "COMPLETE",
  "conditions": [
    "FULL",
    "REMOVE_9",
    "REMOVE_10",
    "REMOVE_11",
    "REMOVE_12"
  ],
  "frozen_bank": "historical REG A0 bank12 original positions25..36",
  "rows_per_condition": 119,
  "non_target_inputs": 47,
  "early_positive_inputs": 72,
  "buckets": {
    "REMOVE_9": {
      "all": {
        "OFF_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "OFF_ON": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_ON_same": {
          "n": 119,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 85,
          "both_wrong": 34,
          "missing": 0
        },
        "ON_ON_different": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        }
      },
      "cohorts": {
        "old47": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 47,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 14,
            "both_wrong": 33,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        },
        "early_positive": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 72,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 71,
            "both_wrong": 1,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        }
      }
    },
    "REMOVE_10": {
      "all": {
        "OFF_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "OFF_ON": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_ON_same": {
          "n": 119,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 85,
          "both_wrong": 34,
          "missing": 0
        },
        "ON_ON_different": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        }
      },
      "cohorts": {
        "old47": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 47,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 14,
            "both_wrong": 33,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        },
        "early_positive": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 72,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 71,
            "both_wrong": 1,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        }
      }
    },
    "REMOVE_11": {
      "all": {
        "OFF_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "OFF_ON": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_ON_same": {
          "n": 119,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 85,
          "both_wrong": 34,
          "missing": 0
        },
        "ON_ON_different": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        }
      },
      "cohorts": {
        "old47": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 47,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 14,
            "both_wrong": 33,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        },
        "early_positive": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 72,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 71,
            "both_wrong": 1,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        }
      }
    },
    "REMOVE_12": {
      "all": {
        "OFF_OFF": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "OFF_ON": {
          "n": 0,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 0,
          "both_wrong": 0,
          "missing": 0
        },
        "ON_OFF": {
          "n": 15,
          "correct_wrong": 0,
          "wrong_correct": 10,
          "both_correct": 4,
          "both_wrong": 1,
          "missing": 0
        },
        "ON_ON_same": {
          "n": 92,
          "correct_wrong": 0,
          "wrong_correct": 0,
          "both_correct": 77,
          "both_wrong": 15,
          "missing": 0
        },
        "ON_ON_different": {
          "n": 12,
          "correct_wrong": 2,
          "wrong_correct": 5,
          "both_correct": 2,
          "both_wrong": 3,
          "missing": 0
        }
      },
      "cohorts": {
        "old47": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 15,
            "correct_wrong": 0,
            "wrong_correct": 10,
            "both_correct": 4,
            "both_wrong": 1,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 20,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 6,
            "both_wrong": 14,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 12,
            "correct_wrong": 2,
            "wrong_correct": 5,
            "both_correct": 2,
            "both_wrong": 3,
            "missing": 0
          }
        },
        "early_positive": {
          "OFF_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "OFF_ON": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_OFF": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          },
          "ON_ON_same": {
            "n": 72,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 71,
            "both_wrong": 1,
            "missing": 0
          },
          "ON_ON_different": {
            "n": 0,
            "correct_wrong": 0,
            "wrong_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "missing": 0
          }
        }
      }
    }
  },
  "causal_diagnostic_only": true,
  "selection_usage": false,
  "answers_from_actual_generation": true
}
```
