# Stage17 main-cohort campaign

Completed N=146 main-cohort single/sequential panels for C_NO_H, BalancEdit, LoRA-Perf-v1, GRACE and BELoRA. C_NO_H/BE sequential are independent-checkpoint insertion replays with new full-bank generation; other sequential methods update their native state. See CAMPAIGN_RESULTS.json and RESULTS.csv for counts, uncertainty and paired comparisons. This is not a full ten-task or paper-exact result. Unsupported H/G and additional task-specific cohorts are not filled with zeros. Existing BE single and Base Astra verdicts were reused. Historical Stage15/16 and Qwen results were not mixed.

User-approved amendment: LoRA single is FP16; sequential is the BF16 stability replacement. The original FP16 sequential run failed at edit 17, step 4 and is retained as a failure. Single/sequential and C_NO_H/LoRA sequential comparisons are not precision matched.
