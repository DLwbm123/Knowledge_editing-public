# 固定回放轨迹：全部节点读出

以下为固定16专家调用子集；来源CHECK为48题/16组强制激活，不能当作全146库或独立确认。区间为缺失界。

| 步数 | CE T2G | 回放T2G | CE来源CHECK | 回放来源CHECK | 回放空答/二元替代（T2G） |
|---|---|---|---|---|---|
| 0 | 95.833 | 95.833 | 8.333 | 8.333 | 0/0，分母65 |
| 12 | 95.833 | [91.667,93.056] | 8.333 | 14.583 | 1/0，分母65 |
| 48 | 95.833 | [79.167,83.333] | 6.250 | 33.333 | 4/0，分母65 |
| 96 | [94.444,95.833] | [80.556,84.722] | 8.333 | 43.750 | 3/4，分母65 |
| 192 | [83.333,90.278] | [84.722,88.889] | 6.250 | 47.917 | 2/3，分母65 |

描述性窗口：[{'node': 12, 'tests': {'T0': 'PASS', 'T1G': 'PASS', 'T2G': 'FAIL', 'CHECK_accuracy': 'PASS', 'CHECK_retention': 'PASS'}, 'status': 'NO_WINDOW_AT_NODE', 'independent_confirmation': False}, {'node': 48, 'tests': {'T0': 'PASS', 'T1G': 'FAIL', 'T2G': 'FAIL', 'CHECK_accuracy': 'PASS', 'CHECK_retention': 'PASS'}, 'status': 'NO_WINDOW_AT_NODE', 'independent_confirmation': False}, {'node': 96, 'tests': {'T0': 'PASS', 'T1G': 'PASS', 'T2G': 'FAIL', 'CHECK_accuracy': 'PASS', 'CHECK_retention': 'PASS'}, 'status': 'NO_WINDOW_AT_NODE', 'independent_confirmation': False}, {'node': 192, 'tests': {'T0': 'PASS', 'T1G': 'PASS', 'T2G': 'FAIL', 'CHECK_accuracy': 'PASS', 'CHECK_retention': 'PASS'}, 'status': 'NO_WINDOW_AT_NODE', 'independent_confirmation': False}]

评分有效594/610，缺失16；累计Judge11908，累计GPU进程小时35.962843。

所有节点、配对区间、T0/T1G/T2G/T1L/T2L、来源保持率和答案类型见TRAJECTORY_RESULTS.json。没有选择或晋级checkpoint；没有追加参数/种子/节点。新增权重在冻结消费者完成且绑定落盘后删除，历史资产保持只读。
