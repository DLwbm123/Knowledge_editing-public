# Stage24C 固定R2 HSIC纯19对照

历史E0/E2复用权重，新R_H使用修复后的确定性后端；本轮不是完整重训的严格HSIC单因素因果对照。全部结果为已查看DEV；Stage25未启动，不能据此声称独立确认或临床验证。

| 臂 | native | DEV H正确 | DEV H保持 | DEV U严格保持 |
|---|---:|---:|---:|---:|
| E0 | 19/19 | 9/16 | 5/12 | 11/15 |
| E2 | 19/19 | 9/16 | 7/12 | 13/15 |
| R_H | 19/19 | 9/16 | 7/12 | 13/15 |

原6H/9U/11改写与DEV16H/16U/正向面板分开报告。PairCorrect11为编辑计权，不代替H unique-QA或来源宏结果。

工程晋级门槛：{"native19": true, "all_panels_noninferior": true, "strict_improvement": false, "DEV_engineering_qualified": false, "new_CONFIRM": "INSUFFICIENT_NEW_CONFIRM_SUPPORT", "stage25_started": false, "tie_prefers": "E2"}

所有有效臂、正负结果和完整评分覆盖均保留；尺度稳定性检查在1/80/160/320步执行，失败即停。
