# Stage17 main-cohort campaign

Completed phase report: C_NO_H sequential, C_NO_H single, balancedit sequential, balancedit single, belora sequential, belora single, grace sequential, grace single, lora single. Remaining phases are pending. C_NO_H/BE sequential are independent-checkpoint insertion replays with new full-bank generation; other sequential methods update their native state. See CAMPAIGN_RESULTS.json and RESULTS.csv for counts, uncertainty and paired comparisons. This is not a full ten-task or paper-exact result. Unsupported H/G and additional task-specific cohorts are not filled with zeros. Existing BE single and Base Astra verdicts were reused. Historical Stage15/16 and Qwen results were not mixed.

User-approved amendment: LoRA single is FP16; sequential is the BF16 stability replacement. The original FP16 sequential run failed at edit 17, step 4 and is retained as a failure. Single/sequential and C_NO_H/LoRA sequential comparisons are not precision matched.
